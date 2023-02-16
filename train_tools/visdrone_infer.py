import logging
import sys
import os
from collections import OrderedDict
import torch
import argparse

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation.evaluator import inference_on_dataset


from utils.val_mapper_with_ann import ValMapper
from utils.anchor_gen import AnchorGeneratorWithCenter
from utils.coco_eval_fpn import COCOEvaluatorFPN
from utils.json_evaluator import JsonEvaluator
from utils.time_evaluator import GPUTimeEvaluator

from visdrone.dataloader import build_train_loader, build_test_loader

# from models.backbone import build
from configs.custom_config import add_custom_config

from models.retinanet.retinanet import RetinaNet_D2
from models.querydet.detector import RetinaNetQueryDet



class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_list.append(JsonEvaluator(os.path.join(cfg.OUTPUT_DIR, 'visdrone_infer.json'), class_add_1=True))
        if cfg.META_INFO.EVAL_GPU_TIME:
            evaluator_list.append(GPUTimeEvaluator(True, 'minisecond'))
        return DatasetEvaluators(evaluator_list)
    
    @classmethod
    def build_train_loader(cls, cfg):
        return build_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_test_loader(cfg)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        logger = logging.getLogger(__name__)
        dataset_name = 'VisDrone2018'

        data_loader = cls.build_test_loader(cfg, dataset_name)
        evaluator = cls.build_evaluator(cfg, dataset_name)
        result = inference_on_dataset(model, data_loader, evaluator)
        if comm.is_main_process():
            assert isinstance(
                result, dict
            ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                result
            )
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(result)

        if len(result) == 1:
            result = list(result.values())[0]
        return result


def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
        Examples:

        Run on single machine:
            $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth

        Run on multiple machines:
            (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
            (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--no-pretrain", action="store_true", help="whether to load pretrained model")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_custom_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def start_train(args):
    cfg = setup(args)
    
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    if not args.no_pretrain:
        trainer.resume_or_load(resume=args.resume)
    return trainer.train()
