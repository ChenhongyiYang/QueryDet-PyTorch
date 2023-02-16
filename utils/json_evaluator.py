import os
import csv 
import json
import torch
import logging
import itertools
import numpy as np 

from detectron2.evaluation.evaluator import DatasetEvaluator
import detectron2.utils.comm as comm
import itertools
from collections import OrderedDict 
from detectron2.evaluation.coco_evaluation import instances_to_coco_json

import numpy as np 

class JsonEvaluator(DatasetEvaluator):
    def __init__(self, out_json, distributed=True, class_add_1=True):
        self._out_json = out_json
        self.class_add_1 = class_add_1

        self._distributed = distributed
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._predictions = []

        self.reset()

    
    def reset(self):
        self._predictions = []


    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            img_name = os.path.split(input['file_name'])[-1].split('.')[0] 
            if "instances" in output:
                prediction = {"img_name": img_name}
                instances = output["instances"].to(self._cpu_device)
                if self.class_add_1:
                    instances.pred_classes += 1
                prediction["instances"] = instances_to_coco_json(instances, input['image_id'])
                self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))
            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions
        
        if len(predictions) == 0:
            return {}
        
        det_preds = []
        for pred in predictions:
            det_preds = det_preds + pred['instances']

        with open(self._out_json, "w") as f:
            f.write(json.dumps(det_preds))
            f.flush()

        return {}


