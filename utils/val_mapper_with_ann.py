# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import torch
from fvcore.common.file_io import PathManager

from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T



class ValMapper(object):
    """
    COCO validation mapper, with annotations
    """

    def __init__(self, cfg):
        self.is_train = False

        self.tfm_gens = utils.build_transform_gen(cfg, self.is_train)

        self.img_format     = cfg.INPUT.FORMAT
        assert not cfg.MODEL.LOAD_PROPOSALS, "not supported yet"
    
        

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        for anno in dataset_dict["annotations"]:
            anno.pop("segmentation", None)
            anno.pop("keypoints", None)

        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=None
                )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]

        instances = utils.annotations_to_instances(annos, image_shape)

        dataset_dict["instances"] = instances[instances.gt_boxes.nonempty()]
        return dataset_dict

  