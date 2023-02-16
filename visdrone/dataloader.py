import os
import json
import bisect
import copy
import itertools
import logging
import numpy as np
import operator
import pickle
import torch.utils.data
from fvcore.common.file_io import PathManager
from tabulate import tabulate
from termcolor import colored

from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size
from detectron2.utils.env import seed_all_rng
from detectron2.utils.logger import log_first_n

from detectron2.structures.boxes import BoxMode
from detectron2.data import samplers
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.common import AspectRatioGroupedDataset, DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import check_metadata_consistency


from visdrone.mapper import Mapper


def get_train_data_dicts(json_file, img_root, filter_empty=False):
    data = json.load(open(json_file))
    
    images = {x['id']: {'file': x['file_name'], 'height':x['height'], 'width':x['width']} for x in data['images']}
        
    annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations.keys():
            annotations[img_id] = []
        annotations[img_id].append({'bbox': ann['bbox'], 'category_id': ann['category_id']-1, 'iscrowd': ann['iscrowd'], 'area': ann['area']})
    
    for img_id in images.keys():
        if img_id not in annotations.keys():
            annotations[img_id] = []
    
    data_dicts = []
    for img_id in images.keys():
        if filter_empty and len(annotations[img_id]) == 0:
            continue
        data_dict = {}
        data_dict['file_name'] = str(os.path.join(img_root, images[img_id]['file']))
        data_dict['height'] = images[img_id]['height']
        data_dict['width'] = images[img_id]['width']
        data_dict['image_id'] = img_id
        data_dict['annotations'] = []
        for ann in annotations[img_id]:
            data_dict['annotations'].append({'bbox': ann['bbox'], 'iscrowd': ann['iscrowd'], 'category_id': ann['category_id'], 'bbox_mode': BoxMode.XYWH_ABS})
        data_dicts.append(data_dict)
    return data_dicts


def get_test_data_dicts(json_file, img_root):
    data = json.load(open(json_file))
    images = {x['id']: {'file': x['file_name'], 'height':x['height'], 'width':x['width']} for x in data['images']}
    
    data_dicts = []
    for img_id in images.keys():
        data_dict = {}
        data_dict['file_name'] = str(os.path.join(img_root, images[img_id]['file']))
        data_dict['height'] = images[img_id]['height']
        data_dict['width'] = images[img_id]['width']
        data_dict['image_id'] = img_id
        data_dict['annotations'] = []
        data_dicts.append(data_dict)
    return data_dicts


def build_train_loader(cfg):
    num_workers = get_world_size()
    images_per_batch = cfg.SOLVER.IMS_PER_BATCH

    assert (
        images_per_batch % num_workers == 0
    ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    assert (
        images_per_batch >= num_workers
    ), "SOLVER.IMS_PER_BATCH ({}) must be larger than the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    images_per_worker = images_per_batch // num_workers

    dataset_dicts = get_train_data_dicts(cfg.VISDRONE.TRAIN_JSON, cfg.VISDRONE.TRING_IMG_ROOT)
    dataset = DatasetFromList(dataset_dicts, copy=False)
    mapper = Mapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    
    if sampler_name == "TrainingSampler":
        sampler = samplers.TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        sampler = samplers.RepeatFactorTrainingSampler(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_worker, drop_last=True
    )
    # drop_last so the batch always have the same size
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,
    )
    return data_loader


def build_test_loader(cfg):

    dataset_dicts = get_test_data_dicts(cfg.VISDRONE.TEST_JSON, cfg.VISDRONE.TEST_IMG_ROOT)

    dataset = DatasetFromList(dataset_dicts)
    mapper = Mapper(cfg, False)
    dataset = MapDataset(dataset, mapper)

    sampler = samplers.InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)


def trivial_batch_collator(batch):
    return batch













