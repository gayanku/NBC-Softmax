# NBC-Softmax :  Darkweb Author fingerprinting and migration tracking
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the PyTorch implementation of "NBC-Softmax", an auxiliary block contrastive loss which only uses the negative samples, to improve traditional softmax. This has achieved state of the art in author style detection. This is the official implementation for the results and work described in our paper [NBC-Softmax](https://arxiv.org/abs/2212.08184)

TLDR; A simple negative block contrastive loss addition for softmax.


The paper describes the loss function NBC-Softmax. This needs to be used with a dataset and network for deep metric learning. 
In this repo we only show the NBC-Softmax code. Portions of the contrastive learning code is from [PAMC](https://github.com/gayanku/PAMC). 
The data and network, network $f(\theta)$. as mentioned in the paper, is from [SYSML](https://github.com/pranavmaneriker/sysml) contains the datsets,  pretrained graph context embeddings and the deep learning.  NBCSAuthor is the dark web author fingerprinting model implementation that uses NBC-softmax loss under the hood.

We acknowledge and thank the authors of these works for sharing their code and data.

![NBC-softmax theory](https://github.com/gayanku/NBC-Softmax/blob/main/theory.png?raw=true)

Above figure shows the comparison between the traditional softmax loss (left) with NBC-softmax ( on the right). We use similarity of different classes, represented and managed by $\hat{\mu}$ to force apart the weight vectors $W$, instead of imposing any soft or hard margins.

## Results
![Results](https://github.com/gayanku/NBC-Softmax/blob/main/Results_NBC-softmax.png?raw=true)



## Setup
- Our code was tested on CUDA 11.3.0, python 3.6.9, pytorch 1.3.1. Please note that some, minimal, changes were needed to get SYSML pl-lightning code to run on the current version.

## Usage
- All parameters are defined in SYSML. additionally we use the following to define the NBS-softmax hyperparameters for --model_params_classwise
```
SingleDatasetModel
--batch_size 2048   
--model_params_classwise "model_type='COMBO2'|model1_type='sm'|model2_type='proj_contrastiveBC1'|model2_ratio=0.5|proj_dim=0|NOTE='singleW2_0.01_G1_0.5_000_TTC_L5_NEG_0.20_z2048'" 

MultiDatasetModel
--batch_size 2048
--model_params_cross "model_type='COMBO2'|model1_type='sm'|model2_type='proj_contrastiveBC1'|model2_ratio=0.5|proj_dim=0|NOTE='mutiW2_0.01_G1_0.5_000_TTC_L5_NEG_0.30_z2048'"
```

## Citation
```
@article{kulatilleke2022__________________________________________________________,
  title={Efficient block contrastive learning via parameter-free meta-node approximation}, 
  author={Kulatilleke, Gayan K and Portmann, Marius and Chandra, Shekhar S},
  journal={arXiv preprint arXiv:2209.14067},
  year={2022}
}
```
