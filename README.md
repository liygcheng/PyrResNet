# PyrResNet
For cvpr2018 paper ["Intrinsic Image Transformation Via Scale Space Decomposition"](https://arxiv.org/pdf/1805.10253.pdf)
Some of the implementations are redundant and will be optimized in the future.

## Prerequisites
- Linux(16.04)
- Python 2 or 3
- NVIDIA GPU(TiTan Xp) + CUDA CuDNN

## Getting Started
### Installation
- Install PyTorch(0.3.0) and torchvision from http://pytorch.org and other dependencies.
You can install all the dependencies by
```bash
pip install -r requirements.txt
```
**Note**: The current software does not update with the newest PyTorch version, some warnings may exist.

- Clone this repo:
```bash
git clone https://github.com/liygcheng/PyrResNet.git
cd PyrResNet
```

- Download Sintel Dataset and MIT Dataset( and additional dataset)
```
https://drive.google.com/open?id=1gcNSwkDSQCwr8CezgRvL0WOX9wetlFIk
```
###  train/test
- Download dataset (e.g. sintel):
- Train a model (e.g. Scene Split):
```bash
python PyrResNet_Joint_MPI.py --cuda --niter=1000 
```
- To view training results and loss plots, change directory to `PyrResNet/Results/` and start
tensorboard as:
```bash
tensorboard --logdir=LossVis --port=10240
```
open the URL http://localhost:10240 and you will get the visualization results.


## Citation
If you use this code for your research, please cite our papers.
```
@InProceedings{Cheng_2018_CVPR,
author = {Cheng, Lechao and Zhang, Chengyi and Liao, Zicheng},
title = {Intrinsic Image Transformation via Scale Space Decomposition},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}




