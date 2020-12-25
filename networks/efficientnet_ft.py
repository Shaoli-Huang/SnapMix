import os
import sys
import numpy as np
from math import floor
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import cv2
from efficientnet_pytorch import EfficientNet

import torchvision
from networks.resnet import *
import pdb

dimdict={'efficientnet-b0':1536 ,'efficientnet-b1':1536 ,'efficientnet-b2':1536 ,'efficientnet-b3':1536 ,'efficientnet-b4':1792,'efficientnet-b5':2048,'efficientnet-b6':2304,'efficientnet-b7':2560}

class ENet(nn.Module):

    def __init__(self,conf):
        super(ENet, self).__init__()
        self.basemodel = EfficientNet.from_pretrained(conf.netname)
        feadim=dimdict[conf.netname]
        self.classifier = nn.Linear(feadim, conf.num_class)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self._dropout = nn.Dropout(0.5)


    def forward(self, x):
        x = self.basemodel.extract_features(x)
        fea_pool = self.avg_pool(x).view(x.size(0), -1)
        logits = self.classifier(fea_pool)
        return logits,x.detach(),None


    def get_params(self, param_name):
        ftlayer_params = list(self.basemodel.parameters())
        ftlayer_params_ids = list(map(id, ftlayer_params))
        freshlayer_params = filter(lambda p: id(p) not in ftlayer_params_ids, self.parameters())

        return eval(param_name+'_params')


def get_net(conf):
    return ENet(conf)
