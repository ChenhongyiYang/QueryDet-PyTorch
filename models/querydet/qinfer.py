from typing import List
import torch
import torch.nn.functional as F 
import spconv


def permute_to_N_HWA_K(tensor, K):
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor

def run_conv2d(x, weights, bias):
    n_conv = len(weights)
    for i in range(n_conv):
        x = F.conv2d(x, weights[i], bias[i])
        if i != n_conv - 1:
            x = F.relu(x)
    return x


class QueryInfer(object):
    def __init__(self, anchor_num, num_classes, score_th=0.12, context=2):
        
        self.anchor_num  = anchor_num
        self.num_classes = num_classes
        self.score_th    = score_th
        self.context     = context 

        self.initialized = False
        self.cls_spconv  = None 
        self.bbox_spconv = None
        self.qcls_spconv = None
        self.qcls_conv   = None 
        self.n_conv      = None
    
    
    def _make_sparse_tensor(self, query_logits, last_ys, last_xs, anchors, feature_value):
        if last_ys is None:
            N, _, qh, qw = query_logits.size()
            assert N == 1
            prob  = torch.sigmoid_(query_logits).view(-1)
            pidxs = torch.where(prob > self.score_th)[0]# .float()
            y = torch.div(pidxs, qw).int()
            x = torch.remainder(pidxs, qw).int()
        else:
            prob  = torch.sigmoid_(query_logits).view(-1)
            pidxs = prob > self.score_th
            y = last_ys[pidxs]
            x = last_xs[pidxs]
        
        if y.size(0) == 0:
            return None, None, None, None, None, None 

        _, fc, fh, fw = feature_value.shape
        
        ys, xs = [], []
        for i in range(2):
            for j in range(2):
                ys.append(y * 2 + i)
                xs.append(x * 2 + j)

        ys = torch.cat(ys, dim=0)
        xs = torch.cat(xs, dim=0)
        inds = (ys * fw + xs).long()

        sparse_ys = []
        sparse_xs = []
        
        for i in range(-1*self.context, self.context+1):
            for j in range(-1*self.context, self.context+1):
                sparse_ys.append(ys+i)
                sparse_xs.append(xs+j)

        sparse_ys = torch.cat(sparse_ys, dim=0)
        sparse_xs = torch.cat(sparse_xs, dim=0)


        good_idx = (sparse_ys >= 0) & (sparse_ys < fh) & (sparse_xs >= 0)  & (sparse_xs < fw)
        sparse_ys = sparse_ys[good_idx]
        sparse_xs = sparse_xs[good_idx]
        
        sparse_yx = torch.stack((sparse_ys, sparse_xs), dim=0).t()
        sparse_yx = torch.unique(sparse_yx, sorted=False, dim=0)
        
        sparse_ys = sparse_yx[:, 0]
        sparse_xs = sparse_yx[:, 1]

        sparse_inds = (sparse_ys * fw + sparse_xs).long()

        sparse_features = feature_value.view(fc, -1).transpose(0, 1)[sparse_inds].view(-1, fc)
        sparse_indices  = torch.stack((torch.zeros_like(sparse_ys), sparse_ys, sparse_xs), dim=0).t().contiguous()
        
        sparse_tensor = spconv.SparseConvTensor(sparse_features, sparse_indices, (fh, fw), 1)
  
        anchors = anchors.tensor.view(-1, self.anchor_num, 4)
        selected_anchors = anchors[inds].view(1, -1, 4)
        return sparse_tensor, ys, xs, inds, selected_anchors, sparse_indices.size(0)

    def _make_spconv(self, weights, biases):
        nets = []
        for i in range(len(weights)):
            in_channel  = weights[i].shape[0]
            out_channel = weights[i].shape[1]
            k_size      = weights[i].shape[2]
            filter = spconv.SubMConv2d(in_channel, out_channel, k_size, 1, padding=k_size//2, indice_key="asd")
            filter.weight.data = weights[i].transpose(1,2).transpose(0,1).transpose(2,3).transpose(1,2).transpose(2,3)
            filter.bias.data   = biases[i]
            nets.append(filter)
            if i != len(weights) - 1:
                nets.append(torch.nn.ReLU())
        return spconv.SparseSequential(*nets)

    def _make_conv(self, weights, biases):
        nets = []
        for i in range(len(weights)):
            in_channel  = weights[i].shape[0]
            out_channel = weights[i].shape[1]
            k_size      = weights[i].shape[2]
            filter = torch.nn.Conv2d(in_channel, out_channel, k_size, 1, padding=k_size//2)
            filter.weight.data = weights[i]
            filter.bias.data   = biases[i]
            nets.append(filter)
            if i != len(weights) - 1:
                nets.append(torch.nn.ReLU())
        return torch.nn.Sequential(*nets)
    
    def _run_spconvs(self, x, filters):
        y = filters(x)
        return y.dense(channels_first=False)

    def _run_convs(self, x, filters):
        return filters(x)

    def run_qinfer(self, model_params, features_key, features_value, anchors_value):
        
        if not self.initialized:
            cls_weights, cls_biases, bbox_weights, bbox_biases, qcls_weights, qcls_biases = model_params
            assert len(cls_weights) == len(qcls_weights)
            self.n_conv = len(cls_weights)
            self.cls_spconv  = self._make_spconv(cls_weights, cls_biases)
            self.bbox_spconv = self._make_spconv(bbox_weights, bbox_biases)
            self.qcls_spconv = self._make_spconv(qcls_weights, qcls_biases)
            self.qcls_conv   = self._make_conv(qcls_weights, qcls_biases)
            self.initialized  = True

        last_ys, last_xs = None, None 
        query_logits = self._run_convs(features_key[-1], self.qcls_conv)
        det_cls_query, det_bbox_query, query_anchors = [], [], []
        
        n_inds_all = []

        for i in range(len(features_value)-1, -1, -1):
            x, last_ys, last_xs, inds, selected_anchors, n_inds = self._make_sparse_tensor(query_logits, last_ys, last_xs, anchors_value[i], features_value[i])
            n_inds_all.append(n_inds)
            if x == None:
                break
            cls_result   = self._run_spconvs(x, self.cls_spconv).view(-1, self.anchor_num*self.num_classes)[inds]
            bbox_result  = self._run_spconvs(x, self.bbox_spconv).view(-1, self.anchor_num*4)[inds]
            query_logits = self._run_spconvs(x, self.qcls_spconv).view(-1)[inds]
            
            query_anchors.append(selected_anchors)
            det_cls_query.append(torch.unsqueeze(cls_result, 0))
            det_bbox_query.append(torch.unsqueeze(bbox_result, 0))

        return det_cls_query, det_bbox_query, query_anchors


