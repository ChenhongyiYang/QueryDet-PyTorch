# QueryDet-PyTorch
This repository is the official implementation of our paper: [QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection](https://arxiv.org/abs/2103.09136)



## Requirement

a. Install [Pytorch 1.4](https://pytorch.org/).

b. Install [APEX](https://github.com/NVIDIA/apex) for mixed precision training.

c. Install our Pytorch based [sparse convolution toolkit](https://github.com/traveller59/spconv).

d. Install the [detectron2 toolkit](https://github.com/facebookresearch/detectron2). Note we build our approach based on version 0.2.1, you may follow the instructions to set environment configs.

e. Install the [Detectron2_Backbone](https://github.com/sxhxliang/detectron2_backbone) for usage of MobileNet and ShuffleNet.

f. Clone our repository and have fun with it!

## Usage

### 1. Data preparation

a. To prepare MS-COCO, you may follow the instructions of Detectron2

b. We provide the data preprocessing code for VisDrone2018. You need to first download dataset from [here](http://aiskyeye.com/) 

c. Check visdrone/data_prepare.py to process the dataset

### 2. Training

```shell
% train coco RetinaNet baseline
python train_coco.py --config-file models/retinanet/configs/coco/train.yaml --num-gpu 8 OUTPUT_DIR /path/to/workdir

% train coco QueryDet 
python train_coco.py --config-file models/querydet/configs/coco/train.yaml --num-gpu 8 OUTPUT_DIR /path/to/workdir

% train VisDrone RetinaNet baseline
python train_visdrone.py --config-file models/retinanet/configs/visdrone/train.yaml --num-gpu 8 OUTPUT_DIR /path/to/workdir

% train VisDrone QueryDet
python train_visdrone.py --config-file models/querydet/configs/visdrone/train.yaml --num-gpu 8 OUTPUT_DIR /path/to/workdir
```

### 3. Test

```shell
% test coco RetinaNet baseline
python infer_coco.py --config-file models/retinanet/configs/coco/test.yaml --num-gpu 8 --eval-only MODEL.WEIGHTS /path/to/workdir/model_final.pth

% test coco QueryDet with Dense Inference
python infer_coco.py --config-file models/querydet/configs/coco/test.yaml --num-gpu 8 --eval-only MODEL.WEIGHTS /path/to/workdir/model_final.pth

% test coco QueryDet with CSQ
python infer_coco.py --config-file models/querydet/configs/coco/test.yaml --num-gpu 8 --eval-only MODEL.WEIGHTS /path/to/workdir/model_final.pth MODEL.QUERY.QUERY_INFER True
```



## 

