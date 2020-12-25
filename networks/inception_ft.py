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
from networks.inception import inception_v3
import pdb


def get_net(conf):
    return inception_v3(pretrained=conf.pretrained)
