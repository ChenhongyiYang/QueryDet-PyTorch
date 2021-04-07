import time
from detectron2.evaluation.evaluator import DatasetEvaluator
import detectron2.utils.comm as comm
import itertools
from collections import OrderedDict 

import numpy as np 

class GPUTimeEvaluator(DatasetEvaluator):
    def __init__(self, distributed, unit, out_file=None):
        self.distributed = distributed
        self.all_time = []
        self.unit = unit
        self.out_file = out_file
        if unit not in {'minisecond', 'second'}:
            raise NotImplementedError('Unsupported time unit %s'%unit)
        self.reset()
    
    def reset(self):
        self.all_time = []

    def process(self, inputs, outputs):
        for output in outputs:
            if 'time' in output.keys():
                self.all_time.append(output['time'])
        return

    def evaluate(self):
        if self.distributed:
            comm.synchronize()
            all_time = comm.gather(self.all_time, dst=0)
            all_time = list(itertools.chain(*all_time))
            
            if not comm.is_main_process():
                return {}
        else:
            all_time = self.all_time

        if len(all_time) == 0:
            return {'GPU_Speed': 0}
        
        all_time = np.array(all_time)
        
        speeds = 1. / all_time
        if self.unit == 'minisecond':
            speeds *= 1000

        mean_speed = speeds.mean() 
        std_speed = speeds.std()
        max_speed = speeds.max()
        min_speed = speeds.min()
        mid_speed = np.median(speeds)

        if self.out_file is not None:
            f = open(self.out_file, 'a')
            curr_time = time.strftime('%Y/%m/%d,%H:%M:%S', time.localtime())
            f.write('%s\t%.2f\n'%(curr_time, mean_speed))
            f.close()

        ret_dict = {'Mean_FPS': mean_speed, 'Std_FPS': std_speed, 'Max_FPS': max_speed, 'Min_FPS': min_speed, 'Mid_FPS': mid_speed}   

        return {'GPU_Speed': ret_dict}