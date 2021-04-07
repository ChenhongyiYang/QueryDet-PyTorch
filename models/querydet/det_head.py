# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
import numpy as np
from typing import List
import torch
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn
import torch.nn.functional as F 

from detectron2.layers import ShapeSpec, batched_nms, cat, Conv2d, get_norm
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from detectron2.modeling.roi_heads.roi_heads import ROIHeads
from detectron2.modeling.poolers import ROIPooler


class RetinaNetHead_3x3(nn.Module):
    def __init__(self, cfg, in_channels, conv_channels, num_convs, num_anchors):
        super().__init__()
        # fmt: off
        num_classes      = cfg.MODEL.RETINANET.NUM_CLASSES
        prior_prob       = cfg.MODEL.RETINANET.PRIOR_PROB
        self.num_convs = num_convs
        # fmt: on

        self.cls_subnet = []
        self.bbox_subnet = []
        channels = in_channels
        for i in range(self.num_convs):
            cls_layer = nn.Conv2d(channels, conv_channels, kernel_size=3, stride=1, padding=1)
            bbox_layer = nn.Conv2d(channels, conv_channels, kernel_size=3, stride=1, padding=1)
            
            torch.nn.init.normal_(cls_layer.weight, mean=0, std=0.01)
            torch.nn.init.normal_(bbox_layer.weight, mean=0, std=0.01)
            
            torch.nn.init.constant_(cls_layer.bias, 0)
            torch.nn.init.constant_(bbox_layer.bias, 0)

            self.add_module('cls_layer_{}'.format(i), cls_layer)
            self.add_module('bbox_layer_{}'.format(i), bbox_layer)

            self.cls_subnet.append(cls_layer)
            self.bbox_subnet.append(bbox_layer)

            channels = conv_channels

        self.cls_score = nn.Conv2d(channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)

        torch.nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.01)

        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, features):
        logits = []
        bbox_reg = []

        for feature in features:
            cls_f  = feature
            bbox_f = feature 
            for i in range(self.num_convs):
                cls_f = F.relu(self.cls_subnet[i](cls_f))
                bbox_f = F.relu(self.bbox_subnet[i](bbox_f))

            logits.append(self.cls_score(cls_f))
            bbox_reg.append(self.bbox_pred(bbox_f))

        return logits, bbox_reg
    
    def get_params(self):
        cls_weights = [x.weight for x in self.cls_subnet] + [self.cls_score.weight.data]
        cls_biases = [x.bias for x in self.cls_subnet] + [self.cls_score.bias.data]

        bbox_weights = [x.weight for x in self.bbox_subnet] + [self.bbox_pred.weight.data]
        bbox_biases = [x.bias for x in self.bbox_subnet] + [self.bbox_pred.bias.data]
        return cls_weights, cls_biases, bbox_weights, bbox_biases
        

class Head_3x3(nn.Module):
    def __init__(self, in_channels, conv_channels, num_convs, pred_channels, pred_prior=None):
        super().__init__()
        self.num_convs = num_convs

        self.subnet = []
        channels = in_channels
        for i in range(self.num_convs):
            layer = nn.Conv2d(channels, conv_channels, kernel_size=3, stride=1, padding=1)
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.constant_(layer.bias, 0)
            self.add_module('layer_{}'.format(i), layer)
            self.subnet.append(layer)
            channels = conv_channels

        self.pred_net = nn.Conv2d(channels, pred_channels, kernel_size=3, stride=1, padding=1)

        torch.nn.init.xavier_normal_(self.pred_net.weight)
        if pred_prior is not None:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.pred_net.bias, bias_value)
        else:
            torch.nn.init.constant_(self.pred_net.bias, 0)

    def forward(self, features):
        preds = []
        for feature in features:
            x = feature
            for i in range(self.num_convs):
                x = F.relu(self.subnet[i](x))
            preds.append(self.pred_net(x))
        return preds

    def get_params(self):
        weights = [x.weight for x in self.subnet] + [self.pred_net.weight]
        biases = [x.bias for x in self.subnet] + [self.pred_net.bias]
        return weights, biases


from utils.merged_sync_bn import MergedSyncBatchNorm

class RetinaNetHead_3x3_MergeBN(nn.Module):
    def __init__(self, cfg, in_channels, conv_channels, num_convs, num_anchors):
        super().__init__()
        # fmt: off
        num_classes      = cfg.MODEL.RETINANET.NUM_CLASSES
        prior_prob       = cfg.MODEL.RETINANET.PRIOR_PROB
        num_anchors      = 1
        self.num_convs   = num_convs
        self.bn_converted = False
        # fmt: on

        self.cls_subnet = []
        self.bbox_subnet = []
        self.cls_bns = []
        self.bbox_bns = []

        channels = in_channels
        for i in range(self.num_convs):
            cls_layer = Conv2d(channels, conv_channels, kernel_size=3, stride=1, padding=1, bias=False, activation=None, norm=None)
            bbox_layer = Conv2d(channels, conv_channels, kernel_size=3, stride=1, padding=1, bias=False, activation=None, norm=None)
            torch.nn.init.normal_(cls_layer.weight, mean=0, std=0.01)
            torch.nn.init.normal_(bbox_layer.weight, mean=0, std=0.01)

            cls_bn = MergedSyncBatchNorm(conv_channels) 
            bbox_bn = MergedSyncBatchNorm(conv_channels)

            self.add_module('cls_layer_{}'.format(i), cls_layer)
            self.add_module('bbox_layer_{}'.format(i), bbox_layer)
            self.add_module('cls_bn_{}'.format(i), cls_bn)
            self.add_module('bbox_bn_{}'.format(i), bbox_bn)

            self.cls_subnet.append(cls_layer)
            self.bbox_subnet.append(bbox_layer)
            self.cls_bns.append(cls_bn)
            self.bbox_bns.append(bbox_bn)

            channels = conv_channels

        self.cls_score = nn.Conv2d(channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)

        torch.nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.01)

        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_score.bias, bias_value)


    def forward(self, features, lvl_start):
        if self.training:
            return self._forward_train(features, lvl_start)
        else:
            return self._forward_eval(features, lvl_start)

    def _forward_train(self, features, lvl_start):
        cls_features = features
        bbox_features = features
        len_feats = len(features)

        for i in range(self.num_convs):          
            cls_features = [self.cls_subnet[i](x) for x in cls_features]
            bbox_features = [self.bbox_subnet[i](x) for x in bbox_features]

            cls_features = self.cls_bns[i](cls_features)
            bbox_features = self.bbox_bns[i](bbox_features)
            
            cls_features = [F.relu(x) for x in cls_features]
            bbox_features = [F.relu(x) for x in bbox_features]
        
        logits = [self.cls_score(x) for x in cls_features]
        bbox_pred = [self.bbox_pred(x) for x in bbox_features]
        return logits, bbox_pred
    

    def _forward_eval(self, features, lvl_start):
        if not self.bn_converted:
            self._bn_convert()
    
        cls_features = features
        bbox_features = features
        len_feats = len(features)

        for i in range(self.num_convs):
            cls_features = [F.relu(self.cls_subnet[i](x)) for x in cls_features]
            bbox_features = [F.relu(self.bbox_subnet[i](x)) for x in bbox_features]
        
        logits     = [self.cls_score(x) for x in cls_features]
        bbox_pred  = [self.bbox_pred(x) for x in bbox_features]

        return logits, bbox_pred, centerness

    def _bn_convert(self):
        # merge BN into head weights
        assert not self.training 
        if self.bn_converted:
            return

        for i in range(self.num_convs):
            cls_running_mean = self.cls_bns[i].running_mean.data
            cls_running_var = self.cls_bns[i].running_var.data
            cls_gamma = self.cls_bns[i].weight.data
            cls_beta  = self.cls_bns[i].bias.data 

            bbox_running_mean = self.bbox_bns[i].running_mean.data
            bbox_running_var = self.bbox_bns[i].running_var.data
            bbox_gamma = self.bbox_bns[i].weight.data
            bbox_beta  = self.bbox_bns[i].bias.data

            cls_bn_scale = cls_gamma * torch.rsqrt(cls_running_var + 1e-10)
            cls_bn_bias  = cls_beta - cls_bn_scale * cls_running_mean

            bbox_bn_scale = bbox_gamma * torch.rsqrt(bbox_running_var + 1e-10)
            bbox_bn_bias  = bbox_beta - bbox_bn_scale * bbox_running_mean

            self.cls_subnet[i].weight.data  = self.cls_subnet[i].weight.data * cls_bn_scale.view(-1, 1, 1, 1)
            self.cls_subnet[i].bias    = torch.nn.Parameter(cls_bn_bias)
            self.bbox_subnet[i].weight.data = self.bbox_subnet[i].weight.data * bbox_bn_scale.view(-1, 1, 1, 1)
            self.bbox_subnet[i].bias   = torch.nn.Parameter(bbox_bn_bias)

        self.bn_converted = True

    def get_params(self):
        if not self.bn_converted:
            self._bn_convert()

        cls_ws = [x.weight.data for x in self.cls_subnet] + [self.cls_score.weight.data]
        bbox_ws = [x.weight.data for x in self.bbox_subnet] + [self.bbox_pred.weight.data]

        cls_bs = [x.bias.data for x in self.cls_subnet] + [self.bbox_pred.weight.data]
        bbox_bs = [x.bias.data for x in self.bbox_subnet] + [self.bbox_pred.bias.data]

        return cls_ws, cls_bs, bbox_ws, bbox_bs


class Head_3x3_MergeBN(nn.Module):
    def __init__(self, in_channels, conv_channels, num_convs, pred_channels, pred_prior=None):
        super().__init__()
        self.num_convs = num_convs
        self.bn_converted = False

        self.subnet = []
        self.bns    = []
        
        channels = in_channels
        for i in range(self.num_convs):
            layer = Conv2d(channels, conv_channels, kernel_size=3, stride=1, padding=1, bias=False, activation=None, norm=None)
            torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
            bn = MergedSyncBatchNorm(conv_channels) 

            self.add_module('layer_{}'.format(i), layer)
            self.add_module('bn_{}'.format(i), bn)

            self.subnet.append(layer)
            self.bns.append(bn)

            channels = conv_channels

        self.pred_net = nn.Conv2d(channels, pred_channels, kernel_size=3, stride=1, padding=1)
        
        torch.nn.init.normal_(self.pred_net.weight, mean=0, std=0.01)
        if pred_prior is not None:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.pred_net.bias, bias_value)
        else:
            torch.nn.init.constant_(self.pred_net.bias, 0)

    def forward(self, features):
        if self.training:
            return self._forward_train(features)
        else:
            return self._forward_eval(features)
    
    def _forward_train(self, features):
        for i in range(self.num_convs):
            features = [self.subnet[i](x) for x in features]
            features = self.bns[i](features)
            features = [F.relu(x) for x in features]
        preds = [self.pred_net(x) for x in features]
        return preds
    
    def _forward_eval(self, features):
        if not self.bn_converted:
            self._bn_convert()

        for i in range(self.num_convs):
            features = [F.relu(self.subnet[i](x)) for x in features]
    
        preds = [self.pred_net(x) for x in features]
        return preds
    
    def _bn_convert(self):
        # merge BN into head weights
        assert not self.training 
        if self.bn_converted:
            return
        for i in range(self.num_convs):
            running_mean = self.bns[i].running_mean.data
            running_var = self.bns[i].running_var.data
            gamma = self.bns[i].weight.data
            beta  = self.bns[i].bias.data 
            bn_scale = gamma * torch.rsqrt(running_var + 1e-10)
            bn_bias  = beta - bn_scale * running_mean
            self.subnet[i].weight.data  = self.subnet[i].weight.data * bn_scale.view(-1, 1, 1, 1)
            self.subnet[i].bias    = torch.nn.Parameter(bn_bias)
        self.bn_converted = True

    def get_params(self):
        if not self.bn_converted:
            self._bn_convert()
        weights = [x.weight.data for x in self.subnet] + [self.pred_net.weight.data]
        biases  = [x.bias.data for x in self.subnet] + [self.pred_net.bias.data]
        return weights, biases

