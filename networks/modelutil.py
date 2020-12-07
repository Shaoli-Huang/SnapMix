
import imp
import os
import torch.nn as nn

def get_model(conf):

    src_file = os.path.join('networks',conf.net_type+'.py')
    netimp = imp.load_source('networks',src_file)
    net = netimp.get_net(conf)
    return net

def count_params(net):
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))










