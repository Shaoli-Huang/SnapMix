import os
import sys
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import time
import logging
from utils import *



def train(train_loader, model, criterion, optimizer, conf,wmodel=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageAccMeter()
    end = time.time()
    model.train()

    time_start = time.time()
    pbar = tqdm(train_loader, dynamic_ncols=True, total=len(train_loader))
    mixmethod = None
    clsw = None
    if 'mixmethod' in conf:
        if 'baseline' not in conf.mixmethod:
            mixmethod = conf.mixmethod
            if wmodel is None:
                wmodel = model

    for idx, (input, target) in enumerate(pbar):

        # measure data loading time
        data_time.add(time.time() - end)
        input = input.cuda()
        target = target.cuda()

        if 'baseline' not in conf.mixmethod:
            input,target_a,target_b,lam_a,lam_b = eval(mixmethod)(input,target,conf,wmodel)

            output,_,moutput = model(input)

            loss_a = criterion(output, target_a)
            loss_b = criterion(output, target_b)
            loss = torch.mean(loss_a* lam_a + loss_b* lam_b)

            if 'inception' in conf.net_type:
                loss1_a = criterion(moutput, target_a)
                loss1_b = criterion(moutput, target_b)
                loss1 = torch.mean(loss1_a* lam_a + loss1_b* lam_b)
                loss += 0.4*loss1

            if 'midlevel' in conf:
                if conf.midlevel:
                    loss_ma = criterion(moutput, target_a)
                    loss_mb = criterion(moutput, target_b)
                    loss += torch.mean(loss_ma* lam_a + loss_mb* lam_b)
        else:
            output,_,moutput = model(input)
            loss = torch.mean(criterion(output, target))

            if 'inception' in conf.net_type:
                loss += 0.4*torch.mean(criterion(moutput,target))

            if 'midlevel' in conf:
                if conf.midlevel:
                    loss += torch.mean(criterion(moutput,target))

        # measure accuracy and record loss
        losses.add(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.add(time.time() - end)
        end = time.time()

        pbar.set_postfix(batch_time=batch_time.value(), data_time=data_time.value(), loss=losses.value(), score=0)



    return losses.value()
