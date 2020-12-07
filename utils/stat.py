import torch
import pdb
import math
from sklearn.metrics import average_precision_score as aps
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def add(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def value(self):
        return self.sum / self.count

class AverageAccMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def add(self, output,target):
        n = output.size(0)
        self.val = self.accuracy(output,target).item()
        self.sum += self.val * n
        self.count += n

    def value(self):
        if self.sum == 0:
            return 0
        else:
            return self.sum / self.count

    def accuracy(self,output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
            #wrong_k = batch_size - correct_k
            #res.append(wrong_k.mul_(100.0 / batch_size))

        return res[0]

