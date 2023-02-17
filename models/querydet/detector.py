# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import sys
import time
from pathlib import Path
sys.path.append(os.path.abspath(Path(__file__).parent.parent))

import logging
import math
import numpy as np
from typing import List
import torch
import torch.nn.functional as F 
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss, sigmoid_focal_loss, giou_loss
from torch import nn

from detectron2.layers import ShapeSpec, batched_nms, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY


from torch.cuda import Event
###########################################################################################
from utils.utils import *
from utils.loop_matcher import LoopMatcher
from utils.soft_nms import SoftNMSer
from utils.anchor_gen import AnchorGeneratorWithCenter
from utils.gradient_checkpoint import checkpoint
import models.querydet.det_head as dh
import models.querydet.qinfer as qf

from torch.cuda.amp import autocast

__all__ = ["RetinaNetQueryDet"]


def permute_to_N_HWA_K(tensor, K):
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


def permute_all_cls_and_box_to_N_HWA_K_and_concat(box_cls, box_delta, num_classes=80):
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    return box_cls, box_delta


def permute_all_to_NHWA_K_not_concat(box_cls, box_delta, num_classes=80):
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes).reshape(-1, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4).reshape(-1, 4) for x in box_delta]
    return box_cls_flattened, box_delta_flattened


@META_ARCH_REGISTRY.register()
class RetinaNetQueryDet(nn.Module):
    """
    Implement Our QueryDet
    """
    def __init__(self, cfg):
        super().__init__()

        # fmt: off
        self.num_classes              = cfg.MODEL.RETINANET.NUM_CLASSES
        self.in_features              = cfg.MODEL.RETINANET.IN_FEATURES   
        self.query_layer_train        = cfg.MODEL.QUERY.Q_FEATURE_TRAIN    
        self.layers_whole_test        = cfg.MODEL.QUERY.FEATURES_WHOLE_TEST        
        self.layers_value_test        = cfg.MODEL.QUERY.FEATURES_VALUE_TEST             
        self.query_layer_test         = cfg.MODEL.QUERY.Q_FEATURE_TEST   
        # Loss parameters:
        self.focal_loss_alpha         = cfg.MODEL.CUSTOM.FOCAL_LOSS_ALPHAS
        self.focal_loss_gamma         = cfg.MODEL.CUSTOM.FOCAL_LOSS_GAMMAS
        self.smooth_l1_loss_beta      = cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA
        self.use_giou_loss            = cfg.MODEL.CUSTOM.GIOU_LOSS  
        self.cls_weights              = cfg.MODEL.CUSTOM.CLS_WEIGHTS
        self.reg_weights              = cfg.MODEL.CUSTOM.REG_WEIGHTS
        # training query head
        self.small_obj_scale          = cfg.MODEL.QUERY.ENCODE_SMALL_OBJ_SCALE
        self.query_loss_weights       = cfg.MODEL.QUERY.QUERY_LOSS_WEIGHT
        self.query_loss_gammas        = cfg.MODEL.QUERY.QUERY_LOSS_GAMMA
        self.small_center_dis_coeff   = cfg.MODEL.QUERY.ENCODE_CENTER_DIS_COEFF
        # Inference parameters:
        self.score_threshold          = cfg.MODEL.RETINANET.SCORE_THRESH_TEST
        self.topk_candidates          = cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST
        self.use_soft_nms             = cfg.MODEL.CUSTOM.USE_SOFT_NMS 
        self.nms_threshold            = cfg.MODEL.RETINANET.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # query inference
        self.query_infer              = cfg.MODEL.QUERY.QUERY_INFER
        self.query_threshold          = cfg.MODEL.QUERY.THRESHOLD
        self.query_context            = cfg.MODEL.QUERY.CONTEXT
        # other settings
        self.clear_cuda_cache         = cfg.MODEL.CUSTOM.CLEAR_CUDA_CACHE
        self.anchor_num               = len(cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS[0]) * \
                                        len(cfg.MODEL.ANCHOR_GENERATOR.SIZES[0])
        self.with_cp                  = cfg.MODEL.CUSTOM.GRADIENT_CHECKPOINT
        # fmt: on
        assert 'p2' in self.in_features

        self.backbone   = build_backbone(cfg)
        if cfg.MODEL.CUSTOM.HEAD_BN:
            self.det_head   = dh.RetinaNetHead_3x3_MergeBN(cfg, 256, 256, 4, self.anchor_num)
            self.query_head = dh.Head_3x3_MergeBN(256, 256, 4, 1)
        else:
            self.det_head   = dh.RetinaNetHead_3x3(cfg, 256, 256, 4, self.anchor_num)
            self.query_head = dh.Head_3x3(256, 256, 4, 1)

        self.qInfer = qf.QueryInfer(9, self.num_classes, self.query_threshold, self.query_context) 

        backbone_shape = self.backbone.output_shape()        
        all_det_feature_shapes = [backbone_shape[f] for f in self.in_features]
        
        self.anchor_generator = build_anchor_generator(cfg, all_det_feature_shapes)
        self.query_anchor_generator = AnchorGeneratorWithCenter(sizes=[128], aspect_ratios=[1.0], 
                                                                strides=[2**(x+2) for x in self.query_layer_train], offset=0.5)    
        # Matching and loss
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)

        self.soft_nmser = SoftNMSer(
            cfg.MODEL.CUSTOM.SOFT_NMS_METHOD,
            cfg.MODEL.CUSTOM.SOFT_NMS_SIGMA,
            cfg.MODEL.CUSTOM.SOFT_NMS_THRESHOLD,
            cfg.MODEL.CUSTOM.SOFT_NMS_PRUND
        )

        if cfg.MODEL.CUSTOM.USE_LOOP_MATCHER:
            self.matcher = LoopMatcher(
                cfg.MODEL.RETINANET.IOU_THRESHOLDS,
                cfg.MODEL.RETINANET.IOU_LABELS,
                allow_low_quality_matches=True,
            )
        else:
            self.matcher = Matcher(
                cfg.MODEL.RETINANET.IOU_THRESHOLDS,
                cfg.MODEL.RETINANET.IOU_LABELS,
                allow_low_quality_matches=True,
            )

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        # initialize with any reasonable #fg that's not too small
        self.loss_normalizer = 100 
        self.loss_normalizer_momentum = 0.9

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, just_forward=False):
        if self.training:
            return self.train_forward(batched_inputs, just_forward)
        else:
            return self.test(batched_inputs)

    def train_forward(self, batched_inputs, just_forward=False):
        if self.clear_cuda_cache:
            torch.cuda.empty_cache()
        
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        all_features = [features[f] for f in self.in_features] 
        all_anchors, all_centers = self.anchor_generator(all_features)

        query_feature  = [all_features[x] for x in self.query_layer_train]  
        _, query_centers = self.query_anchor_generator(query_feature)   
    
        # make prediction
        det_cls, det_delta = self.det_head(all_features)
        query_logits = self.query_head(query_feature)  
        
        if just_forward:
            return None

        gt_classes, gt_reg_targets = self.get_det_gt(all_anchors, gt_instances)
        losses = self.det_loss(gt_classes, gt_reg_targets, det_cls, det_delta, all_anchors)
        
        # query loss
        gt_query = self.get_query_gt(query_centers, gt_instances)
        query_forgrounds = [gt.sum().item() for gt in gt_query]
        _query_loss = self.query_loss(gt_query, query_logits, self.query_loss_gammas, self.query_loss_weights)
        losses.update(_query_loss)
        return losses

    def test(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        results, total_time = self.test_forward(images)  # normal test
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r, 'time':total_time})
        return processed_results
    
    def test_forward(self, images):
        start_event = Event(enable_timing=True)
        end_event = Event(enable_timing=True)

        start_event.record()
        features = self.backbone(images.tensor[:, :, :])

        all_features = [features[f] for f in self.in_features] 
        
        all_anchors, all_centers = self.anchor_generator(all_features)

        features_whole = [all_features[x] for x in self.layers_whole_test]   
        features_value = [all_features[x] for x in self.layers_value_test]  
        features_key   = [all_features[x] for x in self.query_layer_test]

        anchors_whole  = [all_anchors[x] for x in self.layers_whole_test]  
        anchors_value  = [all_anchors[x] for x in self.layers_value_test]
        
        det_cls_whole, det_delta_whole = self.det_head(features_whole)
        
  
        if not self.query_infer:
            det_cls_query, det_bbox_query = self.det_head(features_value)
            det_cls_query = [permute_to_N_HWA_K(x, self.num_classes) for x in det_cls_query]
            det_bbox_query = [permute_to_N_HWA_K(x, 4) for x in det_bbox_query]
            query_anchors = anchors_value
        else:
            if not self.qInfer.initialized:
                cls_weights, cls_biases, bbox_weights, bbox_biases = self.det_head.get_params()
                qcls_weights, qcls_bias = self.query_head.get_params()
                params = [cls_weights, cls_biases, bbox_weights, bbox_biases, qcls_weights, qcls_bias]
            else:
                params = None

            det_cls_query, det_bbox_query, query_anchors = self.qInfer.run_qinfer(params, features_key, features_value, anchors_value)

        results = self.inference(det_cls_whole, det_delta_whole, anchors_whole, 
                                 det_cls_query, det_bbox_query, query_anchors,
                                 images.image_sizes)
        
        end_event.record()
        torch.cuda.synchronize()
        total_time = start_event.elapsed_time(end_event)
        return results, total_time

    # @float_function
    def _giou_loss(self, pred_deltas, anchors, gt_boxes):
        with autocast(False):
            pred_boxes = self.box2box_transform.apply_deltas(pred_deltas, anchors)
            loss = giou_loss(pred_boxes, gt_boxes, reduction='sum')
            return loss


    def det_loss(self, gt_classes, gt_anchors_targets, pred_logits, pred_deltas, all_anchors):
        def convert_gt_cls(logits, gt_class, f_idxs):
            gt_classes_target = torch.zeros_like(logits)
            gt_classes_target[f_idxs, gt_class[f_idxs]] = 1
            return gt_classes_target
        
        alphas = self.focal_loss_alpha
        gammas = self.focal_loss_gamma
        cls_weights = self.cls_weights
        reg_weights = self.reg_weights
        
        assert len(cls_weights) == len(pred_logits)
        assert len(cls_weights) == len(reg_weights)

        batch_size = pred_logits[0].size(0)
        pred_logits, pred_deltas = permute_all_to_NHWA_K_not_concat(pred_logits, pred_deltas, self.num_classes)
        
        lengths = [x.shape[0] for x in pred_logits]
        start_inds = [0] + [sum(lengths[:i]) for i in range(1, len(lengths))]
        end_inds = [sum(lengths[:i+1]) for i in range(len(lengths))]
        
        gt_classes = gt_classes.flatten()
        gt_anchors_targets = gt_anchors_targets.view(-1, 4)

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum().item()
        get_event_storage().put_scalar("num_foreground", num_foreground)
        self.loss_normalizer = (
            self.loss_normalizer_momentum * self.loss_normalizer
            + (1 - self.loss_normalizer_momentum) * num_foreground
        )
        all_anchor_lists = [torch.cat([x.tensor.reshape(-1, 4) for _ in range(batch_size)]) for x in all_anchors]
        gt_clsses_list = [gt_classes[s:e] for s, e in zip(start_inds, end_inds)]
        gt_anchors_targets_list = [gt_anchors_targets[s:e] for s, e in zip(start_inds, end_inds)]
        valid_idxs_list = [valid_idxs[s:e] for s, e in zip(start_inds, end_inds)]
        foreground_idxs_list = [foreground_idxs[s:e] for s, e in zip(start_inds, end_inds)]

        loss_cls = [
            w * sigmoid_focal_loss_jit(
                x[v],
                convert_gt_cls(x, g, f)[v].detach(),
                alpha=alpha,
                gamma=gamma,
                reduction="sum"
            ) 
            for w, x, g, v, f, alpha, gamma in zip(cls_weights, pred_logits, gt_clsses_list, valid_idxs_list, foreground_idxs_list, alphas, gammas)
        ]

        if self.use_giou_loss:
            loss_box_reg = [
                w * self._giou_loss(
                        x[f],
                        a[f].detach(),
                        g[f].detach(),
                    )
                for w, x, a, g, f in zip(reg_weights, pred_deltas, all_anchor_lists, gt_anchors_targets_list, foreground_idxs_list)
            ] 
        else:
            loss_box_reg = [
                w * smooth_l1_loss(
                    x[f], 
                    g[f].detach(),
                    beta=self.smooth_l1_loss_beta,
                    reduction="sum"
                )
                for w, x, g, f in zip(reg_weights, pred_deltas, gt_anchors_targets_list, foreground_idxs_list)
            ]

        loss_cls = sum(loss_cls) / max(1., self.loss_normalizer)
        loss_box_reg = sum(loss_box_reg) / max(1., self.loss_normalizer)
        return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}

    def query_loss(self, gt_small_obj, pred_small_obj, gammas, weights):
        pred_logits = [permute_to_N_HWA_K(x, 1).flatten() for x in pred_small_obj]
        gts = [x.flatten() for x in gt_small_obj]
        loss = sum([sigmoid_focal_loss_jit(x, y, alpha=0.25, gamma=g, reduction="mean") * w for (x, y, g, w) in zip(pred_logits, gts, gammas, weights)]) 
        return {'loss_query': loss}    

    @torch.no_grad()
    def get_det_gt(self, anchors, targets):
        gt_classes = []
        gt_anchors_targets = []
        anchor_layers = len(anchors)
        anchor_lens = [len(x) for x in anchors]
        start_inds = [0] + [sum(anchor_lens[:i]) for i in range(1, len(anchor_lens))]
        end_inds = [sum(anchor_lens[:i+1]) for i in range(len(anchor_lens))]
        all_anchors = Boxes.cat(anchors)  # Rx4

        for targets_per_image in targets:
            
            if type(self.matcher) == Matcher:
                match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes, all_anchors)
                gt_matched_idxs, anchor_labels = self.matcher(match_quality_matrix)
                del(match_quality_matrix)
            elif type(self.matcher) == LoopMatcher:  # for encoding images with lots of gts
                gt_matched_idxs, anchor_labels = self.matcher(targets_per_image.gt_boxes, all_anchors)
            else:
                raise NotImplementedError

            has_gt = len(targets_per_image) > 0
            if has_gt:
                # ground truth box regression
                matched_gt_boxes = targets_per_image.gt_boxes[gt_matched_idxs]

                if not self.use_giou_loss:
                    gt_anchors_reg_targets_i = self.box2box_transform.get_deltas(
                        all_anchors.tensor, matched_gt_boxes.tensor
                    )
                else:
                    gt_anchors_reg_targets_i = matched_gt_boxes.tensor

                gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs]
                # Anchors with label 0 are treated as background.
                gt_classes_i[anchor_labels == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                gt_classes_i[anchor_labels == -1] = -1

            else:
                gt_classes_i = torch.zeros_like(gt_matched_idxs) + self.num_classes
                gt_anchors_reg_targets_i = torch.zeros_like(all_anchors.tensor)

            gt_classes.append([gt_classes_i[s:e] for s, e in zip(start_inds, end_inds)])
            gt_anchors_targets.append([gt_anchors_reg_targets_i[s:e] for s, e in zip(start_inds, end_inds)])
        
        gt_classes = [torch.stack([x[i] for x in gt_classes]) for i in range(anchor_layers)]
        gt_anchors_targets = [torch.stack([x[i] for x in gt_anchors_targets]) for i in range(anchor_layers)]

        gt_classes = torch.cat([x.flatten() for x in gt_classes])
        gt_anchors_targets = torch.cat([x.reshape(-1, 4) for x in gt_anchors_targets])

        return gt_classes, gt_anchors_targets
    
    
    @torch.no_grad()
    def get_query_gt(self, small_anchor_centers, targets):
        small_gt_cls = []
        for lind, anchor_center in enumerate(small_anchor_centers):
            per_layer_small_gt = []
            for target_per_image in targets:
                target_box_scales = get_box_scales(target_per_image.gt_boxes)

                small_inds = (target_box_scales < self.small_obj_scale[lind][1]) & (target_box_scales >= self.small_obj_scale[lind][0])               
                small_boxes = target_per_image[small_inds]
                center_dis, minarg = get_anchor_center_min_dis(small_boxes.gt_boxes.get_centers(), anchor_center)
                small_obj_target = torch.zeros_like(center_dis)
                
                if len(small_boxes) != 0:
                    min_small_target_scale = (target_box_scales[small_inds])[minarg]
                    small_obj_target[center_dis < min_small_target_scale * self.small_center_dis_coeff[lind]] = 1

                per_layer_small_gt.append(small_obj_target)
            small_gt_cls.append(torch.stack(per_layer_small_gt))

        return small_gt_cls


    def inference(self, 
                  retina_box_cls, retina_box_delta, retina_anchors,
                  small_det_logits, small_det_delta, small_det_anchors, 
                  image_sizes
    ):
        results = []

        N, _, _, _ = retina_box_cls[0].size()
        retina_box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in retina_box_cls]
        retina_box_delta = [permute_to_N_HWA_K(x, 4) for x in retina_box_delta]
        small_det_logits = [x.view(N, -1, self.num_classes) for x in small_det_logits]
        small_det_delta = [x.view(N, -1, 4) for x in small_det_delta]

        for img_idx, image_size in enumerate(image_sizes):
            
            retina_box_cls_per_image = [box_cls_per_level[img_idx] for box_cls_per_level in retina_box_cls]
            retina_box_reg_per_image = [box_reg_per_level[img_idx] for box_reg_per_level in retina_box_delta]
            small_det_logits_per_image = [small_det_cls_per_level[img_idx] for small_det_cls_per_level in small_det_logits]
            small_det_reg_per_image = [small_det_reg_per_level[img_idx] for small_det_reg_per_level in small_det_delta]
            
            if len(small_det_anchors) == 0 or type(small_det_anchors[0]) == torch.Tensor:
                small_det_anchor_per_image = [small_det_anchor_per_level[img_idx] for small_det_anchor_per_level in small_det_anchors]
            else:
                small_det_anchor_per_image = small_det_anchors
     
            results_per_img = self.inference_single_image(
                                retina_box_cls_per_image, retina_box_reg_per_image, retina_anchors,
                                small_det_logits_per_image, small_det_reg_per_image, small_det_anchor_per_image,
                                tuple(image_size))
            results.append(results_per_img)

        return results


    def inference_single_image(self, 
                               retina_box_cls, retina_box_delta, retina_anchors, 
                               small_det_logits, small_det_delta, small_det_anchors, 
                               image_size
    ):  
        with autocast(False):
            # small pos cls inference
            all_cls = small_det_logits + retina_box_cls
            all_delta = small_det_delta + retina_box_delta 
            all_anchors = small_det_anchors + retina_anchors

            boxes_all, scores_all, class_idxs_all = self.decode_dets(all_cls, all_delta, all_anchors)
            boxes_all, scores_all, class_idxs_all = [cat(x) for x in [boxes_all, scores_all, class_idxs_all]]
            
            if self.use_soft_nms:  
                keep, soft_nms_scores = self.soft_nmser(boxes_all, scores_all, class_idxs_all)
            else:
                keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_threshold)
            result = Instances(image_size)

            keep = keep[: self.max_detections_per_image]       
            result.pred_boxes = Boxes(boxes_all[keep])
            result.scores = scores_all[keep]
            result.pred_classes = class_idxs_all[keep]
            return result


    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images
    
    def decode_dets(self, cls_results, reg_results, anchors):
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        for cls_i, reg_i, anchors_i in zip(cls_results, reg_results, anchors):
            cls_i = cls_i.view(-1, self.num_classes)
            reg_i = reg_i.view(-1, 4)
        
            cls_i = cls_i.flatten().sigmoid_()  # (HxWxAxK,)
            num_topk = min(self.topk_candidates, reg_i.size(0))
            
            predicted_prob, topk_idxs = cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

        # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes
            predicted_class = classes_idxs

            reg_i = reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]

            if type(anchors_i) != torch.Tensor:
                anchors_i = anchors_i.tensor

            predicted_boxes = self.box2box_transform.apply_deltas(reg_i, anchors_i)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(predicted_class)

        return boxes_all, scores_all, class_idxs_all
    

