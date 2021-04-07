import torch

from detectron2.modeling.anchor_generator import DefaultAnchorGenerator, _create_grid_offsets
from detectron2.modeling import ANCHOR_GENERATOR_REGISTRY
from detectron2.structures import Boxes
import math 
import detectron2.utils.comm as comm


@ANCHOR_GENERATOR_REGISTRY.register()
class AnchorGeneratorWithCenter(DefaultAnchorGenerator):

    def _grid_anchors(self, grid_sizes):
        anchors = []
        centers = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides, self.cell_anchors):
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, base_anchors.device)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            center = torch.stack((shift_x, shift_y), dim=1)

            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))
            centers.append(center.view(-1, 2))
        return anchors, centers

    def forward(self, features):
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps, centers_over_all_feature_maps = self._grid_anchors(grid_sizes)
        anchor_boxes = [Boxes(x) for x in anchors_over_all_feature_maps]

        return anchor_boxes, centers_over_all_feature_maps