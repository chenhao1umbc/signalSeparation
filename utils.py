#@title Adding all the packages
import os
import h5py 
import numpy as np
import scipy.io as sio
from scipy.signal import stft 
from scipy.signal import istft 
import itertools
import norbert

import torch
from torch import nn
import torch.nn.functional as Func
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
plt.rcParams['figure.dpi'] = 100

from unet.unet_model import UNet
import torch_optimizer as optim

"make the result reproducible"
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Opt():
    def __init__(self, ifgendata=False) -> None:
        self.ifgendata = ifgendata
        self.n_epochs = 101
        self.lr = 1e-3
        self.train_model = False


def label_gen(n):
    """This function generates labels in the mixture

    Parameters
    ----------
    n : [int]
        how many components in the mixture
    """
    lb_idx = np.array(list(itertools.combinations([0,1,2,3,4,5], n)))
    label_n = np.zeros( (lb_idx.shape[0], 6) )
    for i in range(lb_idx.shape[0]):
        label_n[i, lb_idx[i]] = 1
    return label_n


def mix_data_torch(x, labels):
    """This functin will mix the data according the to labels

    Parameters
    ----------
    x : [tensor of complex]
        [data with shape of [n_classes, n_samples, time_len]]
    labels : [matrix of int]
        [maxtrix of [n_samples, n_classes]]

    Returns
    -------
    [complex pytorch]
        [mixture data with shape of [n_samples, time_len] ]
    """
    n = labels.shape[0]
    output = np.zeros( (n, x.shape[1], x.shape[2]) ).astype('complex64')
    for i1 in range(n):
        s = 0
        for i2 in range(6):
            if labels[i1, i2] == 1:
                s = s + x[i2]
            else:
                pass
        output[i1] = s
    return torch.tensor(output), torch.tensor(labels).to(torch.float)


def save_mix(x, lb1, lb2, lb3, lb4, lb5, lb6, pre='_'):
    mix_1, label1 = mix_data_torch(x, lb1)  # output is in pytorch tensor
    mix_2, label2 = mix_data_torch(x, lb2)
    mix_3, label3 = mix_data_torch(x, lb3)
    mix_4, label4 = mix_data_torch(x, lb4)
    mix_5, label5 = mix_data_torch(x, lb5)
    mix_6, label6 = mix_data_torch(x, lb6)

    torch.save({'data':mix_1, 'label':label1}, pre+'dict_mix_1.pt')
    torch.save({'data':mix_2, 'label':label2}, pre+'dict_mix_2.pt')
    torch.save({'data':mix_3, 'label':label3}, pre+'dict_mix_3.pt')
    torch.save({'data':mix_4, 'label':label4}, pre+'dict_mix_4.pt')
    torch.save({'data':mix_5, 'label':label5}, pre+'dict_mix_5.pt')
    torch.save({'data':mix_6, 'label':label6}, pre+'dict_mix_6.pt')


def get_label(lb, shape):
    """repeat the labels for the shape of mixture data

    Parameters
    ----------
    lb : [torch.float matrix]
        [matrix of labels]
    shape : [tuple int]
        [data shape]]

    Returns
    -------
    [labels]
        [large matrix]
    """
    n_comb, n_sample = shape
    label = np.repeat(lb, n_sample, axis=0).reshape(n_comb, n_sample, 6 )
    return label


def get_mixdata_label(mix=1, pre='train_'):
    """loading mixture data and prepare labels

    Parameters
    ----------
    mix : int, optional
        [how many components in the mixture], by default 1

    Returns
    -------
    [data, label]
    """
    dict = torch.load('../data/data_ss/'+pre+'dict_mix_'+str(mix)+'.pt')
    label = get_label(dict['label'], dict['data'].shape[:2])
    return dict['data'], label


def get_Unet_input(x, l, y, which_class=0, tr_va_te='_tr', n_batch=30, shuffle=True):
    class_names = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']
    n_sample, t_len = x.shape[1:]
    x = x.reshape(-1, t_len)
    l = l.reshape(-1, 6)

    ind = l[:, which_class]==1.0  # find the index, which belonged to this class
    ltr = l[ind]  # find labels

    "get the stft with low freq. in the center"
    f_bins = 200
    f, t, Z = stft(x[ind], fs=4e7, nperseg=f_bins, boundary=None)
    xtr = torch.tensor(np.log(abs(np.roll(Z, f_bins//2, axis=1))))

    "get the cleaned source as the ground-truth"
    f, t, Z = stft(y[which_class], fs=4e7, nperseg=f_bins, boundary=None)
    temp = torch.tensor(np.log(abs(np.roll(Z, f_bins//2, axis=1))))
    n_tile = int(xtr.shape[0]/n_sample)
    ytr = torch.tensor(np.tile(temp, (n_tile, 1,1)))

    data = Data.TensorDataset(xtr, ytr, ltr)
    data = Data.DataLoader(data, batch_size=n_batch, shuffle=shuffle)

    torch.save(data, class_names[which_class]+tr_va_te+'.pt') 
    print('saved '+class_names[which_class]+tr_va_te)   


def awgn(x, snr=20):
    """
    This function is adding white guassian noise to the given signal
    :param x: the given signal with shape of [...,, T], could be complex64
    :param snr: a float number
    :return:
    """
    x_norm_2 = (abs(x)**2).sum()
    Esym = x_norm_2/ x.numel()
    SNR = 10 ** (snr / 10.0)
    N0 = (Esym / SNR).item()
    noise = torch.tensor(np.sqrt(N0) * np.random.normal(0, 1, x.shape), device=x.device)
    return x+noise.to(x.dtype)
