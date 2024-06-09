## Introduction

This repo contains the PyTorch implementation of our paper
[TKR-FSOD: Fetal Anatomical Structure Few-Shot Detection Utilizing Topological Knowledge Reasoning]

## Quick Start

**1. Check Requirements**
* Linux
* Python == 3.7
* PyTorch == 1.7.1
* torch-geometric==1.5.0
* CUDA 11.0
* GCC >= 4.9

**2. Build DeFRCN**
* Clone Code
  ```angular2html
  git clone https://github.com/lixi92/TKR-FSOD.git
  cd TKR-FSOD
  ```

* Install PyTorch 1.7.1 with CUDA 11.0 
  ```shell
  pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
  ```
* Install Detectron2
  ```angular2html
  python -m pip install detectron2==0.3 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
  ```

* Install Detectron2
  ```angular2html
  python -m pip install detectron2==0.3 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
  ```

* Install torch-geometric, check more details: https://github.com/pyg-team/pytorch_geometric/tree/1.5.0?tab=readme-ov-file
  ```angular2html
  pip install torch-geometric==1.5.0
  pip install torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
  pip install torch-sparse==0.6.8 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
  pip install torch-cluster==1.5.8 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
  pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
  ```

* Install other requirements. 
  ```angular2html
  python3 -m pip install -r requirements.txt
  ```

**3. Prepare Data and Weights**
* same with DeFRCN(https://github.com/er-muyue/DeFRCN), for dataset, we use PASCAL_VOC format.

**4. Training and Evaluation**

* To train model, `EXP_NAME` can be any string
  ```angular2html
  bash run.sh EXP_NAME
  ```

## Acknowledgement
This repo is developed based on [DeFRCN](https://github.com/er-muyue/DeFRCN) and [Detectron2](https://github.com/facebookresearch/detectron2). Please check them for more details and features.


