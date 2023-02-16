import sys

from detectron2.engine import launch
from train_tools.visdrone_infer import default_argument_parser, start_train

import logging

from models.retinanet.retinanet import RetinaNet_D2
from models.querydet.detector import RetinaNetQueryDet



if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        start_train,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )