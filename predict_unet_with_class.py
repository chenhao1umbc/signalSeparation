import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

import pickle
import glob
from unet import UNetWithClass

pickle_file_path = './val_output_pickle_file'
val_percent = 0.1
gamma = 0.1
threshold = 0.5

train_set_file_path = 'train_set_visualization.pickle'


def symbol_accuracy(class_label, class_output, batch_size):
    TP = 0
    for item_index in range(batch_size):
        output = int((1 - threshold) + class_output[item_index, 0].cpu().detach().numpy())
        label = int((1 - threshold) + class_label[item_index, 0].cpu().detach().numpy())
        #print(output, label)
        if output == label:
            TP += 1
    return TP / batch_size


def predict_img(net,
                device,
                input_pickle_file_path,
                source_index):
    net.eval()
    criterion_component = nn.MSELoss()
    criterion_class = nn.BCELoss()

    input_pickle_file = open(input_pickle_file_path, 'rb')
    train_set_info = pickle.load(input_pickle_file)
    mixture = train_set_info['mixture']
    component_label = train_set_info['component_label'][:, source_index, :, :].unsqueeze(1)
    class_label = train_set_info['class_label'][:, source_index:source_index + 1]

    mixture = mixture.to(device=device, dtype=torch.float32)
    component_label = component_label.to(device=device, dtype=torch.float32)
    class_label = class_label.to(device=device, dtype=torch.float32)

    component_output, class_output = net(mixture)

    try:
        component_loss = criterion_component(component_output, component_label)
        classify_loss = criterion_class(class_output, class_label)

    except RuntimeError:
        print(f'input_mixture:{mixture.shape},\n'
              f' class_label:{class_label.shape},\n'
              f' component_output:{component_output.shape},\n'
              f'component_label:{component_label.shape}')
        raise Exception

    class_accuracy = symbol_accuracy(class_label=class_label,
                                     class_output=class_output,
                                     batch_size=class_label.shape[0])
    pickle_file = open(pickle_file_path, 'wb')
    pickle.dump({'component_output': component_output,
                 'class_output': class_output,
                 'mixture': mixture,
                 'component_label': component_label,
                 'class_label': class_label,
                 'classify_loss': classify_loss,
                 'component_loss': component_loss}, pickle_file)
    pickle_file.close()

    print(f'classify_loss:{classify_loss}, \n'
          f'component_loss:{component_loss}, \n'
          f'total_loss:{(1-gamma)*component_loss + gamma*classify_loss}, \n'
          f'pickle_file_path:{pickle_file_path},\n'
          f'symbol accuracy:{class_accuracy}')

    return {'component_output': component_output,
            'class_output': class_output,
            'mixture': mixture,
            'component_label': component_label,
            'class_label': class_label,
            'classify_loss': classify_loss,
            'component_loss': component_loss}


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")

    parser.add_argument('--source', '-s', type=int,
                        help="Source index",
                        default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    net = UNetWithClass(n_channels=1, n_classes=1)

    logging.info("Loading model {}".format(args.model))
    logging.info("Source index {}".format(args.source))

    source_index = args.source

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    mask = predict_img(net=net,
                       device=device,
                       input_pickle_file_path=train_set_file_path,
                       source_index=source_index)

