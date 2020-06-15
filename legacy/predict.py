import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

import pickle
import glob
from unet import UNet, InitNet
from utils.dataset import PsdDatasetWithClass
from torch.utils.data import DataLoader, random_split


dir_mixture = '../datasets/dataset_0426_14000_128x20/mixture_dataset_multiple/mixture_data_14000.pickle'
dir_list_label = glob.glob('datasets/dataset_0426_14000_128x20/component/*.pickle')
pickle_file_path = '../val_output_pickle_file_fcn'
dir_train_sample_pickle = '../datasets/dataset_0426_14000_128x20/train_set.pickle'
dir_val_sample_pickle = '../datasets/dataset_0426_14000_128x20/val_set.pickle'
val_percent = 0.1
gamma = 0.5

train_set_file_path = '../train_set_visualization.pickle'
val_set_root_path = '../val_output_pickle_file_fcn'


def predict_img(net,
                device):
    net.eval()

    print(f'Load dataset from previous generation!\n'
          f'training set path:{dir_train_sample_pickle}\n'
          f'validation set path:{dir_val_sample_pickle}')
    train_sample_file = open(dir_train_sample_pickle, 'rb')
    val_sample_file = open(dir_val_sample_pickle, 'rb')
    val_loader = pickle.load(val_sample_file)
    train_sample_file.close()
    val_sample_file.close()

    criterion_class = nn.BCEWithLogitsLoss()
    criterion_component = nn.MSELoss()

    for batch in val_loader:

        train_set_file = open(train_set_file_path, 'rb')
        train_set_info = pickle.load(train_set_file)
        mixture = train_set_info['mixture']
        print(train_set_info['component_label'].shape)
        component_label = train_set_info['component_label'][:, 1, :, :].unsqueeze(1)
        print('component label shape:', component_label.shape)

        mixture = mixture.to(device=device, dtype=torch.float32)
        component_label = component_label.to(device=device, dtype=torch.float32)
        #class_label = class_label.to(device=device, dtype=torch.float32)

        component_output, class_output = net(mixture)

        try:
            #classify_loss = criterion_class(class_output, class_label)
            classify_loss = 0
            component_loss = criterion_component(component_output, component_label)
            total_loss = gamma * classify_loss + (1 - gamma) * component_loss
        except RuntimeError:
            print(f'input_mixture:{mixture.shape},\n'
                  f' class_output:{class_output.shape},\n'
                  f' class_label:{None},\n'
                  f' component_output:{component_output.shape},\n'
                  f'component_label:{component_label.shape}')
            raise Exception

        pickle_file = open(pickle_file_path, 'wb')
        pickle.dump({'component_output': component_output,
                     'class_output': class_output,
                     'mixture': mixture,
                     'component_label': component_label,
                     'class_label': class_label,
                     'classify_loss': classify_loss,
                     'component_loss': component_loss}, pickle_file)
        pickle_file.close()
        break

    print(f'classify_loss:{classify_loss}, \n'
          f'component_loss:{component_loss}, \n'
          f'total_loss:{total_loss}')


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")

    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    net = InitNet(n_classes=1)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    mask = predict_img(net=net,
                       device=device)

