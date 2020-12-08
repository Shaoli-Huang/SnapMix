# SnapMix: Semantically Proportional Mixing for Augmenting Fine-grained Data (AAAI 2021)

Official PyTorch implementation of SnapMix | [paper](https://)

## Method Overview

![SnapMix](./imgs/overview.jpg)

## How to cite
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
Python Environment: >= 3.6
torch = 1.2.0
torchvision = 0.4.0
scikit-learn >= 0.2
tensorbard >= 2.0.0
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

Applying SnapMix in training:

```python main.py --mixmethod snapmix --beta 5 --depth 50 --dataset cub ``` # baseline 

```python main.py --mixmethod snapmix --beta 5 --depth 50 --dataset cub --midlevel ``` # baseline+ 

Applying other augmentation methods (currently support cutmix,cutout,and mixup) in training:

```python main.py --mixmethod cutmix --beta 3 --depth 50 --dataset cub ```   # training with CutMix

```python main.py --mixmethod mixup --prob 0.5 --depth 50 --dataset cub ```  # training with MixUp

### Training from scratch
```python main.py --mixmethod snapmix --prob 0.5 --depth 18 --dataset cub --pretrained 0``` # resnet-18 backbone

```python main.py --mixmethod snapmix --prob 0.5 --depth 50 --dataset cub --pretrained 0 ``` # resnet-50 backbone

