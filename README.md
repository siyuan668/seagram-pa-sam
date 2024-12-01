# Fine Tune PA-SAM for Seagrass Semantic Segmentation
 Fine Tune PA-SAM for Seagrass Semantic Segmentation

## Introduction

The Segment Anything Model (SAM) has exhibited outstanding performance in various image segmentation tasks. Despite being trained with over a billion masks, SAM faces challenges in mask prediction quality in numerous scenarios, especially in real-world contexts. In this repo, we fine tune Prompt Adapter Segment Anything Model (PA-SAM), aiming to enhance the segmentation mask quality of the UAV seagrass images. 


The architecture of the prompt adapter, which achieves adaptive detail enhancement using a consistent representation module (CRM) and token-to-image attention, and implements hard point mining using the Gumbel top-k point sampler.


## Setup and Installation

The code package can be cloned from the git repository using:

```bash
> git clone https://github.com/siyuan668/seagram-pa-sam.git
```

### Anaconda Environment Setup

The conda environment for the code can be created using the `envname.yaml` file provided with the scripts.

```bash
> cd pa-sam
> conda env create --file envname.yml
> conda activate pasam
```

## Getting Started

### Training

```
sh train.sh
```

or

```
python -m torch.distributed.launch --nproc_per_node=<num_gpus> train.py --checkpoint <path/to/checkpoint> --model-type <model_type> --output <path/to/output>
```

Example PA-SAM-L training script

```
python -m torch.distributed.launch --nproc_per_node=8 train.py --checkpoint ./pretrained_checkpoint/sam_vit_l_0b3195.pth --model-type vit_l --output work_dirs/pa_sam_l
```

### Evaluation

```
sh eval.sh
```

or

```
python -m torch.distributed.launch --nproc_per_node=<num_gpus> train.py --checkpoint <path/to/checkpoint> --model-type <model_type> --output <path/to/output> --eval --restore-model <path/to/training_checkpoint>
```

Example PA-SAM-L evaluation script, and you can download the checkpoint from [here](https://pan.baidu.com/s/1PGfooGqweEPeXWvA5c55EA?pwd=wr97)

```
python -m torch.distributed.launch --nproc_per_node=1 train.py --checkpoint ./pretrained_checkpoint/sam_vit_l_0b3195.pth --model-type vit_l --output work_dirs/pa_sam_l --eval --restore-model work_dirs/pa_sam_l/epoch_20.pth
```


## Acknowledgement

The repo is based on the official implementation: https://github.com/xzz2/pa-sam: 
 [&#39;PA-SAM: Prompt Adapter SAM for High-quality Image Segmentation&#39;](https://arxiv.org/abs/2401.13051).

This repo benefits from [Segment Anything in High Quality](https://github.com/SysCV/sam-hq). Thanks for their wonderful works.
