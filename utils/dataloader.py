
from torch.utils import data
import imp
import os


def get_dataloader(conf):

    src_file = os.path.join('datasets',conf.dataset+'.py')
    dataimp = imp.load_source('loader',src_file)
    ds_train,ds_test = dataimp.get_dataset(conf)
    if 'trainshuffle' in conf:
        trainshuffle = conf.trainshuffle
    else:
        trainshuffle = True

    print('train shuffle:',trainshuffle)
    train_loader = data.DataLoader(ds_train, batch_size=conf.batch_size, shuffle= trainshuffle, num_workers=conf.workers, pin_memory=True)
    val_loader =data.DataLoader(ds_test, batch_size=conf.batch_size, shuffle= False, num_workers=conf.workers, pin_memory=True)

    return train_loader,val_loader









