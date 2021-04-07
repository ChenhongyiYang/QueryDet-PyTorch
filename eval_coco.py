from coco import coco
from coco import cocoeval
import json

import argparse
parser = argparse.ArgumentParser(description='Evaluation Arguments')
parser.add_argument('--result', required=True, type=str, help='coco results json')
args = parser.parse_args()


result_file = args.result
ann_file = 'path/to/instances_val2017.json'

def main():
    coco_gt = coco.COCO(ann_file)
    coco_dt = coco_gt.loadRes(result_file)

    coco_eval = cocoeval.COCOeval(coco_gt, coco_dt, 'bbox', eval_pkl=None, vis_path=None)

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    main()