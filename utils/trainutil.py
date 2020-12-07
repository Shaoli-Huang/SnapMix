
import torch
import torch.nn as nn
import imp
import numpy as np
import utils
import os
import torch.nn.functional as F
import random
import copy


def get_sgd(params,conf):
    return torch.optim.SGD(params,conf.lr,momentum=conf.momentum,\
                weight_decay=conf.weight_decay,nesterov=True)


# criterion

def get_criterion(conf):
    reduction = 'mean'
    if 'reduction' in conf:
        reduction = conf.reduction
    return eval('nn.'+conf.criterion)(reduction=reduction).cuda()

# LR scheduler
def get_multisteplr(optim,conf):

    return torch.optim.lr_scheduler.MultiStepLR(optim, \
            milestones=conf.lrstep, gamma=conf.lrgamma, last_epoch=-1)

def get_proc(conf):

    if 'train_proc' not in conf:
        train_proc = conf.net_type
    else:
        train_proc = conf.train_proc

    if 'test_proc' not in conf:
        test_proc = conf.net_type
    else:
        test_proc = conf.test_proc

    trainfile = '{}_train.py'.format(train_proc)
    testfile = '{}_test.py'.format(test_proc)
    trainpy= os.path.join('trainer',trainfile)
    testpy = os.path.join('trainer',testfile)
    train = imp.load_source('train',trainpy).train
    validate = imp.load_source('validate',testpy).validate

    return train,validate



# parameters

def get_params(model,conf=None):

    if conf is not None and 'prams_group' in  conf:
        prams_group = conf.prams_group
        lr_group = conf.lr_group
        params = []
        for pram,lr in zip(prams_group,lr_group):
            params.append({'params':model.module.get_params(pram),'lr': lr})

        return params

    return model.parameters()


def get_train_setting(model,conf):

    optim = 'sgd'
    criterion = 'cross_entropy'
    lrscheduler = 'multisteplr'

    if 'optim' in conf:
        optim = conf.optim

    if 'criterion' in conf:
        criterion = conf.criterion

    if 'lrscheduler' in conf:
        lrscheduler = conf.lrscheduler


    criterion = get_criterion(conf)
    optim = eval('get_'+optim)(get_params(model,conf),conf)
    lrscheduler = eval('get_'+lrscheduler)(optim,conf)

    return criterion,optim,lrscheduler





