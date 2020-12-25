import torch
import os
import os.path as path
from datetime import datetime
import shutil
from tqdm import tqdm
import math
from urllib.request import urlretrieve


# ---------------load checkpoint--------------------
def load_checkpoint(model, pth_file):
    print('==> Reading from model checkpoint..')
    assert os.path.isfile(pth_file), 'Error: no model checkpoint directory found!'
    checkpoint = torch.load(pth_file)

    pretrained_dict = checkpoint['state_dict']
    model_dict = model.module.state_dict()
    model_dict.update(pretrained_dict)

    model.module.load_state_dict(model_dict)
    print("=> loaded model checkpoint '{}' (epoch {})"
            .format(pth_file, checkpoint['epoch']))

    return checkpoint


# ---------------save checkpoint--------------------
def save_checkpoint(state, is_best=False, outdir='checkpoint', filename='checkpoint.pth',iteral=50):

    epochnum = state['epoch']
    filepath = os.path.join(outdir, filename)
    epochpath =  str(epochnum)+'_'+filename
    epochpath = os.path.join(outdir, epochpath)
    if epochnum % iteral == 0:
        savepath = epochpath
    else:
        savepath = filepath
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(outdir, 'model_best.pth.tar'))



def set_outdir(conf):

    default_outdir = 'results'
    if 'timedir' in conf:
        timestr = datetime.now().strftime('%d-%m-%Y_%I_%M-%S_%p')
        outdir = os.path.join(default_outdir,conf.exp_name, \
            conf.net_type+'_'+conf.dataset,timestr)
    else:
        outdir = os.path.join(default_outdir,conf.exp_name, \
            conf.netname+'_'+conf.dataset)

        prefix = 'bs_'+str(conf.batch_size)+'seed_'+str(conf.seed)

        if conf.weightfile:
            prefix = 'ft_'+prefix

        if not conf.pretrained:
            prefix = 'scratch_'+prefix

        if 'midlevel' in conf:
            if conf.midlevel:
                prefix += 'mid_'
        if 'mixmethod' in conf:
            if isinstance(conf.mixmethod,list):
                prefix += '_'.join(conf.mixmethod)
            else:
                prefix += conf.mixmethod+'_'
        if 'prob' in conf:
            prefix += '_p'+str(conf.prob)
        if 'beta' in conf:
            prefix += '_b'+str(conf.beta)

        outdir = os.path.join(outdir,prefix)
    ensure_dir(outdir)
    conf['outdir'] = outdir

    return conf



# check if dir exist, if not create new folder
def ensure_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print('{} is created'.format(dir_name))


def ensure_file(file_path):

    newpath = file_path
    if os.path.exists(file_path):
        timestr = datetime.now().strftime('%d-%m-%Y_%I_%M-%S_%p_')
        newpath = path.join(path.dirname(file_path),timestr + path.basename(file_path))
    return newpath


