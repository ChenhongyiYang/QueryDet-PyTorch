# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper"]


class Mapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True):

        self.tfm_gens = build_transform_gen(cfg, is_train)
        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = False
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.keypoint_on    = False
        self.load_proposals = False
        self.keypoint_hflip_indices = None
        # fmt: on
        
        self.is_train = is_train

    def __call__(self, dataset_dict):

        dataset_dict = copy.deepcopy(dataset_dict)  
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # h, w

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))


        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict


def build_transform_gen(cfg, is_train):
    if is_train:
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        sample_style = 'choice'
    
    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip(horizontal=True, vertical=False))
        tfm_gens.append(T.ResizeShortestEdge(short_edge_length=cfg.VISDRONE.SHORT_LENGTH, max_size=cfg.VISDRONE.MAX_LENGTH, sample_style=sample_style))
    else:
        tfm_gens.append(T.ResizeShortestEdge(short_edge_length=[cfg.VISDRONE.TEST_LENGTH], max_size=cfg.VISDRONE.TEST_LENGTH, sample_style=sample_style))
        
    return tfm_gens


























