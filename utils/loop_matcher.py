# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List
import torch

# useful when there are huge number of gt boxes
class LoopMatcher(object):
    def __init__(
        self, thresholds: List[float], labels: List[int], allow_low_quality_matches: bool = False
    ):
        thresholds = thresholds[:]
        assert thresholds[0] > 0
        thresholds.insert(0, -float("inf"))
        thresholds.append(float("inf"))
        assert all(low <= high for (low, high) in zip(thresholds[:-1], thresholds[1:]))
        assert all(l in [-1, 0, 1] for l in labels)
        assert len(labels) == len(thresholds) - 1

        self.low_quality_thrshold = 0.3
        self.thresholds = thresholds
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches


    def _iou(self, boxes, box):
        iw = torch.clamp(boxes[:, 2], max=box[2]) - torch.clamp(boxes[:, 0], min=box[0])
        ih = torch.clamp(boxes[:, 3], max=box[3]) - torch.clamp(boxes[:, 1], min=box[1])
        
        inter = torch.clamp(iw, min=0) * torch.clamp(ih, min=0)

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        area = (box[2] - box[0]) * (box[3] - box[1])

        iou = inter / (areas + area - inter)
        return iou 

    def __call__(self, gt_boxes, anchors):
        if len(gt_boxes) == 0:
            default_matches = torch.zeros((len(anchors)), dtype=torch.int64).to(anchors.tensor.device)
            default_match_labels = torch.zeros((len(anchors)), dtype=torch.int8).to(anchors.tensor.device) + self.labels[0]
            return default_matches, default_match_labels

        gt_boxes_tensor = gt_boxes.tensor
        anchors_tensor  = anchors.tensor

        max_ious = torch.zeros((len(anchors))).to(anchors_tensor.device)
        matched_inds = torch.zeros((len(anchors)), dtype=torch.long).to(anchors_tensor.device)
        gt_ious  = torch.zeros((len(gt_boxes))).to(anchors_tensor.device)

        for i in range(len(gt_boxes)):
            ious = self._iou(anchors_tensor,  gt_boxes_tensor[i])
            gt_ious[i] = ious.max()
            matched_inds = torch.where(ious > max_ious, torch.zeros(1, dtype=torch.long, device=matched_inds.device)+i, matched_inds)
            max_ious = torch.max(ious, max_ious)
            del(ious)

        matched_vals = max_ious
        matches = matched_inds

        match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)

        for (l, low, high) in zip(self.labels, self.thresholds[:-1], self.thresholds[1:]):
            low_high = (matched_vals >= low) & (matched_vals < high)
            match_labels[low_high] = l

        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(match_labels, matched_vals, matches, gt_ious)

        return matches, match_labels

    def set_low_quality_matches_(self, match_labels, matched_vals, matches, gt_ious):
        for i in range(len(gt_ious)):
            match_labels[(matched_vals==gt_ious[i]) & (matches==i)] = 1
        
