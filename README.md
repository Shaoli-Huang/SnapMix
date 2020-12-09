# SnapMix: Semantically Proportional Mixing for Augmenting Fine-grained Data (AAAI 2021)

PyTorch implementation of SnapMix | [paper](https://)

## Method Overview

![SnapMix](./imgs/overview.jpg)

## Cite
```
@inproceedings{huang2021snapmix,
    title={SnapMix: Semantically Proportional Mixing for Augmenting Fine-grained Data},
    author={Shaoli Huang, Xinchao Wang, and Dacheng Dao},
    year={2021},
    booktitle={AAAI Conference on Artificial Intelligence},
}
```

## Setup
### Install Package Dependencies
```
torch
torchvision 
PyYAML
easydict
tqdm
scikit-learn
pandas
opencv
```
### Datasets
***create a soft link to the dataset directory***

CUB dataset
```
ln -s /your-path-to/CUB-dataset data/cub
```
Car dataset
```
ln -s /your-path-to/Car-dataset data/car
```
Aircraft dataset
```
ln -s /your-path-to/Aircraft-dataset data/aircraft
```

## Training

### Training with Imagenet pre-trained weights


***1. Baseline and Baseline+***

To train a model on CUB dataset using the Resnet-50 backbone, 

``` python main.py ```   # baseline

``` python main.py --midlevel```  # baseline+

To train model on other datasets using other network backbones, you can specify the following arguments: 

``` --net_type: name of network type. {inceptionv3_ft,resnet_ft,densenet_ft} ```

``` --depth: network depth,e.g., {18,34,50,101} for resnet architecture ```

``` --dataset: dataset name```

For example, 

``` python main.py --depth 18 --dataset car ```   # using the Resnet-18 backbone on Car dataset

``` python main.py --net_type inceptionv3_ft --dataset aircraft ```  # using the inceptionV3 backbone on Aircraft dataset


***2. Training with mixing augmentation***

Applying SnapMix in training ( we used the hyperparameter values (prob=1., beta=5) for SnapMix in most of the experiments.):

```python main.py --mixmethod snapmix --beta 5 --depth 50 --dataset cub ``` # baseline 

```python main.py --mixmethod snapmix --beta 5 --depth 50 --dataset cub --midlevel ``` # baseline+ 

Applying other augmentation methods (currently support cutmix,cutout,and mixup) in training:

```python main.py --mixmethod cutmix --beta 3 --depth 50 --dataset cub ```   # training with CutMix

```python main.py --mixmethod mixup --prob 0.5 --depth 50 --dataset cub ```  # training with MixUp

***3. Results***

|  Backbone | Method | CUB   | Car    |   Aircraft |  
|:--------|:--------|--------:|------:|--------:|
|Resnet-18 | Baseline| 82.35% |  91.15% | 87.80% |  
|Resnet-18 | Baseline + SnapMix| 84.29% |  93.12% | 90.17% |
|Resnet-34 | Baseline| 84.98% |  92.02% | 89.92% |  
|Resnet-34 | Baseline + SnapMix| 87.06% |  93.95% | 92.36% |
|Resnet-50 | Baseline| 85.49% |  93.04% | 91.07% |  
|Resnet-50 | Baseline + SnapMix| 87.75% |  94.30% | 92.08% |
|Resnet-101 | Baseline| 85.62% |  93.09% | 91.59% |  
|Resnet-101 | Baseline + SnapMix| 88.45% |  94.44% | 93.74% |
|Resnet-50 | Baseline+| 87.13% |  93.80% | 91.68% |  
|Resnet-50 | Baseline+ + SnapMix| 88.70% |  95.00% | 93.24% |
|Resnet-101 | Baseline+| 87.81% |  93.94% | 91.85% |  
|Resnet-101 | Baseline+ + SnapMix| 89.32% |  94.84% | 94.05% |

|  Backbone | Method | CUB   | 
|:--------|:--------|--------:|
|InceptionV3 | Baseline| 82.22% |
|InceptionV3 | Baseline + SnapMix| 85.54%|
|DenseNet121 | Baseline| 84.23% |  
|DenseNet121| Baseline + SnapMix| 87.42%|

### Training from scratch

To train a model without using ImageNet pretrained weights:

```python main.py --mixmethod snapmix --prob 0.5 --depth 18 --dataset cub --pretrained 0``` # resnet-18 backbone

```python main.py --mixmethod snapmix --prob 0.5 --depth 50 --dataset cub --pretrained 0 ``` # resnet-50 backbone

***2. Results***

|  Backbone | Method | CUB   | 
|:--------|:--------|--------:|
|Resnet-18 | Baseline| 64.98% |
|Resnet-18 | Baseline + SnapMix| 70.31%|
|Resnet-50 | Baseline| 66.92% |  
|Resnet-50| Baseline + SnapMix| 72.17%|

