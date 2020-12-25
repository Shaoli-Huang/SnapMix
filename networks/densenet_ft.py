import os
import sys
import numpy as np
from math import floor
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import cv2

import torchvision
from torchvision import models
import pdb

dimdict = {'densenet121':1024,'densenet201':1920}

class DenseNet(nn.Module):

    def __init__(self,conf):
        super(DenseNet, self).__init__()
        basenet = eval('models.'+conf.netname)(pretrained=conf.pretrained)
        self.feature = nn.Sequential(*list(basenet.children())[:-1])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        indim = dimdict[conf.netname]
        self.classifier = nn.Linear(indim, conf.num_class)

    def set_detach(self,isdetach):
        pass


    def forward(self, x):
        x = self.feature(x)
        x = F.relu(x, inplace=True)
        fea_pool = self.avg_pool(x).view(x.size(0), -1)
        logits = self.classifier(fea_pool)
        return logits,x.detach(),None

        #results = {'logit': [logits]}
        #return results

    def _init_weight(self, block):
        for m in block.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_params(self, param_name):
        ftlayer_params = list(self.feature.parameters())
        ftlayer_params_ids = list(map(id, ftlayer_params))
        freshlayer_params = filter(lambda p: id(p) not in ftlayer_params_ids, self.parameters())

        return eval(param_name+'_params')


def get_net(conf):
    return DenseNet(conf)
