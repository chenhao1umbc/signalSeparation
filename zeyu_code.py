#%% em_process
# Data and module loading
import norbert
import numpy as np
import logging
import argparse
import torch
import glob
import pickle
import torch.nn as nn
from unet import UNet


pickle_file_path = './EM_outputs/EM_output_iter_'
train_set_file_path = './train_set_visualization.pickle'
EM_loss_file_path = './EM_scoreFile.pickle'

# todo: replace static numbers
sample_length = 20
batch_size = 100
freq_bin = 128
# todo: multiple channels
nb_channels = 1
nb_sources = 4


def get_args():
    parser = argparse.ArgumentParser(description='Arguments parser for EM process',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--init_model', '-im', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the initialization model is stored")
    parser.add_argument('--refine_models', '-refms', default='./MODELS',
                        metavar='FILE',
                        help="Specify the folder in which the initialization model is stored")
    parser.add_argument('--class_model', '-cm', default='CLASS_MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the classify model is stored")
    parser.add_argument('--iterations', '-iter', default='1',
                        metavar='FILE',
                        help="Specify the number of EM iterations")

    return parser.parse_args()


class EMCapsule:
    init_nets = None

    def __init__(self, init_model_paths, refine_model_paths, classify_model_paths):
        self.init_model_paths = init_model_paths
        self.refine_model_paths = refine_model_paths
        self.classify_model_paths = classify_model_paths
        self.labels = []
        self.init_nets = dict()
        self.refine_nets = dict()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {self.device}')
        logging.info(f"Loading initialization model dir {init_model_paths}\n"
                     f"refining model dir {refine_model_paths}\n"
                     f"classify model dir {classify_model_paths}")

        for model_path in init_model_paths:
            label = model_path.rstrip('.pth')
            init_net = UNet(n_channels=1, n_classes=1)
            init_net.to(device=self.device)
            init_net.load_state_dict(torch.load(model_path, map_location=self.device))
            self.init_nets[label] = init_net

        for refine_model_path in refine_model_paths:
            refine_net = UNet(n_channels=1, n_classes=1)
            refine_net.to(device=self.device)
            refine_net.load_state_dict(torch.load(refine_model_path, map_location=self.device))
            self.refine_nets[label] = init_net

        logging.info("Model loaded !")

    @staticmethod
    def psd_model(cls, net, psd_mixture):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        psd_mixture = psd_mixture.to(device=device, dtype=torch.float32)
        output = net(psd_mixture)
        return output.cpu().detach().numpy()

    def process(self, iteration, psd_mixture, component_label):
        psd_sources = torch.zeros([batch_size, nb_sources, sample_length, freq_bin])
        psd_sources = psd_sources.to(device=self.device, dtype=torch.float32)
        criterion_component = nn.MSELoss().cuda()
        all_scores = dict()
        for source_index in range(nb_sources):
            all_scores[source_index] = []

        for it in range(int(iteration)):
            print(f"EM iteration {it}/{iteration}")
            source_index = 0
            flatten_sources = np.zeros((sample_length*batch_size, freq_bin, nb_channels, nb_sources))

            # Psd modeling
            for label in self.init_nets:
                if it == 0:
                    psd_source = self.psd_model(self.init_nets[label], psd_mixture)
                else:
                    psd_source = self.psd_model(self.refine_nets[label],
                                                psd_sources[:, source_index:source_index+1, :, :])

                for batch_index in range(batch_size):
                    flatten_sources[batch_index * sample_length:(batch_index + 1) * sample_length, :, 0, source_index] = \
                        psd_source[batch_index, 0, :, :]

                source_index += 1

            # torch to numpy and flatten
            flatten_mixture = np.zeros((sample_length * batch_size, freq_bin, nb_channels))
            for batch_index in range(batch_size):
                flatten_mixture[batch_index * sample_length:(batch_index + 1) * sample_length, :, 0] = \
                    psd_mixture.cpu().detach().numpy()[batch_index, 0, :]

            # db to absolute
            flatten_mixture = np.exp(flatten_mixture)
            flatten_sources = np.exp(flatten_sources)

            # PSD to stft
            stft_mixture = np.sqrt(flatten_mixture)
            stft_sources = np.sqrt(flatten_sources)

            # Wiener filtering
            stft_sources, flatten_sources, cov_matrix = norbert.expectation_maximization(stft_sources, stft_mixture)

            # Reflect back to torch
            for source_index in range(nb_sources):
                for batch_index in range(batch_size):
                    psd_sources[batch_index, source_index, :, :] = \
                        torch.Tensor(np.log(flatten_sources[batch_index * sample_length:(batch_index + 1) * sample_length, :, source_index]))

            # Score Evaluation after each iter
            for source_index in range(nb_sources):
                source_loss = criterion_component(psd_sources[:, source_index, :, :].cuda(),
                                                  component_label[:, source_index, :, :].cuda())
                print(f'MSE loss for current iter, source {source_index}:{source_loss.item()}')
                all_scores[source_index].append(source_loss.item())

            # Store all outputs into pickle file
            pickle_file = open(pickle_file_path + str(it), 'wb')
            pickle.dump({'component_output': psd_sources,
                         'component_label': component_label,
                         'mixture': psd_mixture,
                         'class_label': None,
                         'class_output': None,
                         'classify_loss': None}, pickle_file)
            pickle_file.close()

            # Store all losses into pickle file
            loss_file = open(EM_loss_file_path, 'wb')
            pickle.dump(all_scores, loss_file)
            loss_file.close()

            print(f'component_output:{psd_sources.shape}, \n'
                  f'component_label:{component_label.dtype}, \n'
                  f'mixture:{psd_mixture.shape}\n'
                  f'store path:{pickle_file_path + str(it)}')

        return None


if __name__ == "__main__":

    input_pickle_file_path = open(train_set_file_path, 'rb')
    train_set_info = pickle.load(input_pickle_file_path)
    mixture = train_set_info['mixture']
    component_label = train_set_info['component_label']

    args = get_args()

    # init_model_paths = glob.glob(args.init_model + '*.pth')
    init_model_paths = ['init_models/blt.pth', 'init_models/ZigbeeOQPSK.pth',
                        'init_models/ZigbeeASK.pth', 'init_models/ZigbeeBPSK.pth']
    # refine_model_paths = glob.glob(args.refine_models + '*.pth')
    refine_model_paths = ['init_models/blt.pth', 'init_models/ZigbeeOQPSK.pth',
                          'init_models/ZigbeeASK.pth', 'init_models/ZigbeeBPSK.pth']
    class_model_paths = glob.glob(args.class_model + '*.pth')

    # print(init_model_paths)

    em_capsule = EMCapsule(init_model_paths=init_model_paths,
                           refine_model_paths=refine_model_paths,
                           classify_model_paths=class_model_paths)

    em_capsule.process(iteration=args.iterations, psd_mixture=mixture, component_label=component_label)


#%% em_process_with_class

# Data and module loading
import norbert
import numpy as np
import logging
import argparse
import torch
import glob
import pickle
import torch.nn as nn
from unet import UNet


pickle_file_path = './EM_outputs/EM_output_iter_'
train_set_file_path = './train_set_visualization.pickle'
EM_loss_file_path = './EM_scoreFile.pickle'

# todo: replace static numbers
sample_length = 20
batch_size = 100
freq_bin = 128
# todo: multiple channels
nb_channels = 1
nb_sources = 4


def symbol_accuracy(class_label, class_output, batch_size, threshold=0.5):
    TP = 0
    for item_index in range(batch_size):
        output = int((1 - threshold) + class_output[item_index, 0].cpu().detach().numpy())
        label = int((1 - threshold) + class_label[item_index, 0].cpu().detach().numpy())
        if output == label:
            TP += 1
    return TP / batch_size


def get_args():
    parser = argparse.ArgumentParser(description='Arguments parser for EM process',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--init_model', '-im', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the initialization model is stored")
    parser.add_argument('--refine_models', '-refms', default='./MODELS',
                        metavar='FILE',
                        help="Specify the folder in which the initialization model is stored")
    parser.add_argument('--class_model', '-cm', default='CLASS_MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the classify model is stored")
    parser.add_argument('--iterations', '-iter', default='1',
                        metavar='FILE',
                        help="Specify the number of EM iterations")

    return parser.parse_args()


class EMCapsule:
    init_nets = None

    def __init__(self, init_model_paths, refine_model_paths, classify_model_paths):
        self.init_model_paths = init_model_paths
        self.refine_model_paths = refine_model_paths
        self.classify_model_paths = classify_model_paths
        self.labels = []
        self.init_nets = dict()
        self.refine_nets = dict()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {self.device}')
        logging.info(f"Loading initialization model dir {init_model_paths}\n"
                     f"refining model dir {refine_model_paths}\n"
                     f"classify model dir {classify_model_paths}")

        for model_path in init_model_paths:
            label = model_path.rstrip('.pth')
            init_net = UNet(n_channels=1, n_classes=1)
            init_net.to(device=self.device)
            init_net.load_state_dict(torch.load(model_path, map_location=self.device))
            self.init_nets[label] = init_net

        for refine_model_path in refine_model_paths:
            refine_net = UNet(n_channels=1, n_classes=1)
            refine_net.to(device=self.device)
            refine_net.load_state_dict(torch.load(refine_model_path, map_location=self.device))
            self.refine_nets[label] = init_net

        logging.info("Model loaded !")

    @staticmethod
    def psd_model(net, psd_mixture):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        psd_mixture = psd_mixture.to(device=device, dtype=torch.float32)
        component_output, class_output = net(psd_mixture)
        return component_output.cpu().detach().numpy(), class_output

    def process(self, iteration, psd_mixture, component_label, class_label):
        psd_sources = torch.zeros([batch_size, nb_sources, sample_length, freq_bin])
        psd_sources = psd_sources.to(device=self.device, dtype=torch.float32)
        criterion_component = nn.MSELoss().cuda()
        all_scores = dict()
        for source_index in range(nb_sources):
            all_scores[source_index] = []

        for it in range(int(iteration)):
            print(f"EM iteration {it}/{iteration}")
            source_index = 0
            flatten_sources = np.zeros((sample_length*batch_size, freq_bin, nb_channels, nb_sources))

            # Psd modeling
            for label in self.init_nets:
                if it == 0:
                    psd_source, class_output = self.psd_model(self.init_nets[label], psd_mixture)
                else:
                    psd_source, class_output = self.psd_model(self.refine_nets[label],
                                                              psd_sources[:, source_index:source_index+1, :, :])


                for batch_index in range(batch_size):
                    flatten_sources[batch_index * sample_length:(batch_index + 1) * sample_length, :, 0, source_index] = \
                        psd_source[batch_index, 0, :, :]

                source_index += 1

            # torch to numpy and flatten
            flatten_mixture = np.zeros((sample_length * batch_size, freq_bin, nb_channels))
            for batch_index in range(batch_size):
                flatten_mixture[batch_index * sample_length:(batch_index + 1) * sample_length, :, 0] = \
                    psd_mixture.cpu().detach().numpy()[batch_index, 0, :]

            # db to absolute
            flatten_mixture = np.exp(flatten_mixture)
            flatten_sources = np.exp(flatten_sources)

            # PSD to stft
            stft_mixture = np.sqrt(flatten_mixture)
            stft_sources = np.sqrt(flatten_sources)

            # Wiener filtering
            stft_sources, flatten_sources, cov_matrix = norbert.expectation_maximization(stft_sources, stft_mixture)

            # Reflect back to torch
            for source_index in range(nb_sources):
                for batch_index in range(batch_size):
                    psd_sources[batch_index, source_index, :, :] = \
                        torch.Tensor(np.log(flatten_sources[batch_index * sample_length:(batch_index + 1) * sample_length, :, source_index]))

            # Score Evaluation after each iter
            for source_index in range(nb_sources):
                source_loss = criterion_component(psd_sources[:, source_index, :, :].cuda(),
                                                  component_label[:, source_index, :, :].cuda())
                print(f'MSE loss for current iter, source {source_index}:{source_loss.item()}')
                all_scores[source_index].append(source_loss.item())

            # Store all outputs into pickle file
            pickle_file = open(pickle_file_path + str(it), 'wb')
            pickle.dump({'component_output': psd_sources,
                         'component_label': component_label,
                         'mixture': psd_mixture,
                         'class_label': None,
                         'class_output': None,
                         'classify_loss': None}, pickle_file)
            pickle_file.close()

            # Store all losses into pickle file
            loss_file = open(EM_loss_file_path, 'wb')
            pickle.dump(all_scores, loss_file)
            loss_file.close()

            print(f'component_output:{psd_sources.shape}, \n'
                  f'component_label:{component_label.dtype}, \n'
                  f'mixture:{psd_mixture.shape}\n'
                  f'store path:{pickle_file_path + str(it)}')

        return None


if __name__ == "__main__":

    # input_pickle_file_path = open(train_set_file_path, 'rb')
    # train_set_info = pickle.load(input_pickle_file_path)
    # mixture = train_set_info['mixture']
    # component_label = train_set_info['component_label']
    # class_label = train_set_info['class_label']

    args = get_args()

    # init_model_paths = glob.glob(args.init_model + '*.pth')
    init_model_paths = ['../init_models/blt.pth', '../init_models/ZigbeeOQPSK.pth',
                        '../init_models/ZigbeeASK.pth', '../init_models/ZigbeeBPSK.pth']
    # refine_model_paths = glob.glob(args.refine_models + '*.pth')
    refine_model_paths = ['../init_models/blt.pth', '../init_models/ZigbeeOQPSK.pth',
                          '../init_models/ZigbeeASK.pth', '../init_models/ZigbeeBPSK.pth']
    class_model_paths = glob.glob(args.class_model + '*.pth')

    # print(init_model_paths)

    em_capsule = EMCapsule(init_model_paths=init_model_paths,
                           refine_model_paths=refine_model_paths,
                           classify_model_paths=class_model_paths)

    em_capsule.process(iteration=args.iterations, psd_mixture=mixture, component_label=component_label, class_label=class_label)


#%% eval_fcn
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff


def eval_net(net, loader, device):
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    print(f'Validation sample num:{n_val}')
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs = batch['mixture'].unsqueeze(1)
            true_masks = batch['source_labels'][:, 0, :, :].unsqueeze(1)
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            tot += F.mse_loss(mask_pred, true_masks).item()
            pbar.update()
    print(f'Validation loss:{tot/n_val}')
    return tot / n_val


#%% predict_simple_unet_with_class
import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

import pickle
import glob
from unet import UNetSmallWithClass

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
    net = UNetSmallWithClass(n_channels=1, n_classes=1)

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

#%% eval
import torch
import torch.nn.functional as F
from tqdm import tqdm

gamma = 0.1


def eval_net(net, loader, device, source_index):
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    print(f'Validation sample num:{n_val}')
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs = batch['mixture'].unsqueeze(1)
            true_masks = batch['source_labels'][:, source_index, :, :].unsqueeze(1)
            true_class = batch['class_label'][:, source_index:source_index + 1]
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)
            true_class = true_class.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                mask_pred, class_output = net(imgs)

            tot += (1 - gamma)*F.mse_loss(mask_pred, true_masks).item()
            tot += gamma*F.binary_cross_entropy(class_output, true_class).item()
            pbar.update()
    print(f'Validation loss:{tot/n_val}')
    return tot / n_val


#%% train_simple

import argparse
import logging
import os
import sys
import pickle
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNetSmallWithClass
from utils.dataset import PsdDatasetWithClass
from torch.utils.data import DataLoader, random_split


class NetworkTrainer(object):

    def __init__(self, net, dir_mixture, dir_list_label):
        """
        Initializing dataset
        """
        self.net = net
        self.dir_mixture = dir_mixture
        self.dir_list_label = dir_list_label
        self.dataset = PsdDatasetWithClass(self.dir_mixture, self.dir_list_label)

    def count_parameters(self):
        """
        Counting parameters in the network
        """
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def train_net(self,
                  device,
                  dir_checkpoint,
                  epochs=5,
                  batch_size=10,
                  lr=0.01,
                  val_percent=0.1,
                  gamma=0.1,
                  save_cp=True,
                  store_data_flag=True,
                  store_path="train_set_visualization.pickle"):
        """
        Training network
        """

        n_val = int(len(self.dataset) * val_percent)
        n_train = len(self.dataset) - n_val
        train, val = random_split(self.dataset, [n_train, n_val])
        all_scores_train = []
        all_scores_val = []

        train_loader = DataLoader(train, batch_size=batch_size, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=True)

        logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
        Checkpoints:     {dir_checkpoint}
        Mixture Dir:     {self.dir_mixture}
        Parameter Number:{self.count_parameters()}
        ''')

        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
        class_criterion = nn.BCELoss()

        if net.n_classes > 1:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        batch_index = 0
        for epoch in range(epochs):
            net.train()
            epoch_loss = 0
            batch_num = n_train / batch_size
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='samples') as pbar:
                for batch in train_loader:
                    batch_index += 1

                    imgs = batch['mixture'].unsqueeze(1)
                    true_masks = batch['source_labels'][:, source_index, :, :].unsqueeze(1)
                    true_class = batch['class_label'][:, source_index:source_index + 1]

                    assert imgs.shape[1] == net.n_channels, \
                        f'Network has been defined with {net.n_channels} input channels, ' \
                        f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                    imgs = imgs.to(device=device, dtype=torch.float32)
                    mask_type = torch.float32 if net.n_classes == 1 else torch.long
                    true_masks = true_masks.to(device=device, dtype=mask_type)
                    true_class = true_class.to(device=device, dtype=torch.float32)

                    masks_pred, class_output = net(imgs)

                    loss = criterion(masks_pred, true_masks)
                    class_loss = class_criterion(class_output, true_class)
                    total_loss = (1 - gamma) * loss + gamma * class_loss

                    epoch_loss += total_loss.item()
                    if batch_index == 1 and store_data_flag:
                        self.store_test_data(masks_pred, class_output, imgs, batch, loss, store_path)

                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_value_(net.parameters(), 0.1)
                    optimizer.step()

                    pbar.update(imgs.shape[0])

                val_score = eval_net(net, val_loader, device, source_index=source_index)
                scheduler.step(val_score)
                all_scores_train.append(epoch_loss / batch_num)
                all_scores_val.append(val_score)

                if net.n_classes > 1:
                    logging.info('Validation cross entropy: {}'.format(val_score))
                else:
                    logging.info('Validation Dice Coeff: {}'.format(val_score))

                # Store all training loss
                score_file_train = open(loss_storage_file_path, 'wb')
                score_file_val = open(loss_storage_file_path_val, 'wb')
                pickle.dump(all_scores_train, score_file_train)
                pickle.dump(all_scores_val, score_file_val)
                score_file_train.close()
                score_file_val.close()
                print(f'Training Score file saved! Location:{loss_storage_file_path}\n'
                      f'Validation Score file saved! Location:{loss_storage_file_path_val}')

                if len(all_scores_val) > 3 and all_scores_val[-1] > all_scores_val[-2] > all_scores_val[-3]:
                    logging.info(f'Validation threshold reached! Current val loss:{all_scores_val[-1]}, '
                                 f'current batch num:{len(all_scores_val)}')
                    break

            if save_cp:
                try:
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(net.state_dict(),
                           dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')

    @staticmethod
    def store_test_data(masks_pred, class_output, imgs, batch, loss, training_set_visualization_file_path):
        """
        Store training data for further evaluation
        """
        pickle_file = open(training_set_visualization_file_path, 'wb')
        pickle.dump({'component_output': masks_pred,
                     'class_output': class_output,
                     'mixture': imgs,
                     'component_label': batch['source_labels'],
                     'class_label': batch['class_label'],
                     'classify_loss': None,
                     'component_loss': loss.item()}, pickle_file)
        pickle_file.close()
        logging.info(f'training visualization file stored! File path:{training_set_visualization_file_path}')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet for signal separation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batch_size')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=0.1,
                        help='Percent of the data that is used as validation (0-1)')

    return parser.parse_args()


if __name__ == '__main__':
    # Specify all paths
    source_index = 3
    dir_img = 'data/imgs/'
    dir_mask = 'data/masks/'
    dir_checkpoint = 'checkpoints_simple_with_class_' + str(source_index) + '/'
    dir_mixture = 'datasets/dataset_0426_14000_128x20/mixture_dataset_multiple/mixture_data_14000.pickle'
    dir_list_label = ['datasets/dataset_0426_14000_128x20/component/Blt.mat.pickle',
                      'datasets/dataset_0426_14000_128x20/component/Zigbee.mat.pickle',
                      'datasets/dataset_0426_14000_128x20/component/ZigbeeASK.mat.pickle',
                      'datasets/dataset_0426_14000_128x20/component/ZigbeeBPSK.mat.pickle']
    dir_train_sample_pickle = 'datasets/dataset_0426_14000_128x20/train_set.pickle'
    dir_val_sample_pickle = 'datasets/dataset_0426_14000_128x20/val_set.pickle'
    training_set_visualization_file_path = 'train_set_visualization.pickle'
    loss_storage_file_path = 'scoreFile_' + str(source_index) + '.pickle'
    loss_storage_file_path_val = 'scoreFileVal_' + str(source_index) + '.pickle'

    # Start logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Changed channel number to 1 for STFT images
    net = UNetSmallWithClass(n_channels=1, n_classes=1, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    try:
        networkTrainer = NetworkTrainer(net, dir_list_label=dir_list_label, dir_mixture=dir_mixture)
        networkTrainer.train_net(device=device,
                                 dir_checkpoint=dir_checkpoint,
                                 epochs=args.epochs,
                                 batch_size=args.batch_size,
                                 lr=args.lr,
                                 val_percent=args.val)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

#%% train_fcn_with_class
import argparse
import logging
import os
import sys
import pickle
import torch
import glob
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from unet import InitNet

from utils.dataset import PsdDatasetWithClass
from torch.utils.data import DataLoader, random_split

dir_checkpoint = 'checkpoints_fcn/'
dir_mixture = 'datasets/dataset_0426_14000_128x20/mixture_dataset_multiple/mixture_data_14000.pickle'
dir_list_label = ['datasets/dataset_0426_14000_128x20/component/Blt.mat.pickle',
                  'datasets/dataset_0426_14000_128x20/component/Zigbee.mat.pickle',
                  'datasets/dataset_0426_14000_128x20/component/ZigbeeASK.mat.pickle',
                  'datasets/dataset_0426_14000_128x20/component/ZigbeeBPSK.mat.pickle']
dir_train_sample_pickle = 'datasets/dataset_0426_14000_128x20/train_set.pickle'
dir_val_sample_pickle = 'datasets/dataset_0426_14000_128x20/val_set.pickle'
training_set_visualization_file_path = 'train_set_visualization_fcn.pickle'
loss_storage_file_path = 'scoreFile_fcn.pickle'


def train_net(net,
              device,
              epochs=5,
              batch_size=10,
              lr=0.01,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5,
              gamma=0,
              dir_mixture=dir_mixture,
              dir_list_label=dir_list_label,
              override=True):
    dataset = PsdDatasetWithClass(dir_mixture, dir_list_label)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    if not override:
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    else:
        print(f'Load dataset from previous generation!\n'
              f'training set path:{dir_train_sample_pickle}\n'
              f'validation set path:{dir_val_sample_pickle}')
        train_sample_file = open(dir_train_sample_pickle, 'rb')
        val_sample_file = open(dir_val_sample_pickle, 'rb')
        train_loader = pickle.load(train_sample_file)
        val_loader = pickle.load(val_sample_file)
        train_sample_file.close()
        val_sample_file.close()

    # writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    criterion_class = nn.BCELoss()
    criterion_component = nn.MSELoss()
    batch_index = 0
    allScores = []
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='samples') as pbar:
            for batch in train_loader:
                batch_index += 1
                mixture = batch['mixture']
                class_label = batch['class_label']
                component_label = batch['source_labels'][:, 0:1, :, :]

                mixture = mixture.to(device=device, dtype=torch.float32)
                component_label = component_label.to(device=device, dtype=torch.float32)
                class_label = class_label.to(device=device, dtype=torch.float32)

                component_output, class_output = net(mixture)

                try:
                    #classify_loss = criterion_class(class_output, class_label)
                    classify_loss = 0
                    component_loss = criterion_component(component_output, component_label)
                    total_loss = gamma*classify_loss + (1-gamma)*component_loss
                except RuntimeError:
                    print(f'input_mixture:{mixture.shape},\n'
                          f' class_output:{class_output.shape},\n'
                          f' class_label:{class_label.shape},\n'
                          f' component_output:{component_output.shape},\n'
                          f'component_label:{component_label.shape}')
                    raise Exception

                epoch_loss += total_loss.item()
                # writer.add_scalar('Loss/train', loss.item(), global_step)

                if batch_index == 500:
                    pickle_file = open(training_set_visualization_file_path, 'wb')
                    pickle.dump({'component_output': component_output,
                                 'class_output': class_output,
                                 'mixture': mixture,
                                 'component_label': component_label,
                                 'class_label': class_label,
                                 'classify_loss': None, #classify_loss.item(),
                                 'component_loss': component_loss.item()}, pickle_file)
                    pickle_file.close()
                    print(f'training visualization file stored! File path:{training_set_visualization_file_path}')

                pbar.set_postfix(**{'loss (batch)': total_loss.item()})

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                pbar.update(mixture.shape[0])
                '''
                global_step += 1
                if global_step % (len(dataset) // (1 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        #writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        #writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    #writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        #writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        #writer.add_scalar('Dice/test', val_score, global_step)'''

            allScores.append(epoch_loss)
            # Store all loss
            scoreFile = open(loss_storage_file_path, 'wb')
            pickle.dump(allScores, scoreFile)
            scoreFile.close()
            print(f'Score file saved! Location: {loss_storage_file_path}')

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved ! dir:{dir_checkpoint}')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 4 classes for 4 types of signal
    net = InitNet(n_classes=1)
    logging.info(f'Network:\n'
                 f'\t{net.n_classes} output channels (classes)\n')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


#%% dataset
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import pickle


class PsdDataset(Dataset):
    def __init__(self, mixture_file_path, label_file_path, preprocess_type='real'):
        mixture_file = open(mixture_file_path, 'rb')
        label_file = open(label_file_path, 'rb')
        self.mixture_list = pickle.load(mixture_file)
        self.label_list = pickle.load(label_file)
        self.preprocess_type = preprocess_type
        mixture_file.close()
        label_file.close()
        logging.info(f'Creating dataset with {len(self.mixture_list)} examples')

    def __len__(self):
        return len(self.mixture_list)

    @classmethod
    def preprocess(cls, input_data, type='real'):
        shape_detect = input_data.shape
        while len(shape_detect) < 3:
            input_data = np.expand_dims(input_data, axis=0)
            shape_detect = input_data.shape

        if type == 'real':
            return np.real(input_data)
        elif type == 'imag':
            return np.imag(input_data)

    def __getitem__(self, i):
        mixture = self.preprocess(self.mixture_list[i], self.preprocess_type)
        label = self.preprocess(self.label_list[i], self.preprocess_type)
        return {'mixture': torch.from_numpy(mixture), 'label': torch.from_numpy(label)}


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}


class PsdDatasetWithClass(Dataset):
    def __init__(self, mixture_file_path, label_file_paths, preprocess_type='real'):
        mixture_file = open(mixture_file_path, 'rb')
        self.label_list = dict()
        self.mixture_list = pickle.load(mixture_file)
        mixture_file.close()
        for i in range(len(self.mixture_list[0]['label'])):
            label_file = open(label_file_paths[i], 'rb')
            self.label_list[i] = pickle.load(label_file)
            label_file.close()
        self.preprocess_type = preprocess_type
        logging.info(f'Creating dataset with {len(self.mixture_list)} examples')

    def __len__(self):
        return len(self.mixture_list)

    @classmethod
    def preprocess(cls, input_data, type='real', default_dimension=2):
        shape_detect = input_data.shape
        while len(shape_detect) < default_dimension:
            input_data = np.expand_dims(input_data, axis=0)
            shape_detect = input_data.shape

        if type == 'real':
            return np.real(input_data)
        elif type == 'imag':
            return np.imag(input_data)

    def __getitem__(self, i):
        mixture = self.preprocess(self.mixture_list[i]['mixture'], self.preprocess_type)
        class_label = self.mixture_list[i]['label']
        source_labels = []
        for channel_index in range(len(self.mixture_list[i]['label'])):
            channel = self.mixture_list[i]['label'][channel_index]
            component_sample_num = len(self.label_list[channel_index])
            if channel == 1:
                source_labels.append(self.preprocess(self.label_list[channel_index][i % component_sample_num],
                                                     self.preprocess_type))

            else:
                source_labels.append(self.preprocess(np.zeros(self.label_list[channel_index][i % component_sample_num].shape),
                                                     self.preprocess_type))

        source_labels = np.array(source_labels)
        return {'mixture': torch.from_numpy(mixture), 'class_label': torch.from_numpy(class_label),
                'source_labels': torch.from_numpy(source_labels)}


#%% data_vis
import matplotlib.pyplot as plt

def plot_img_and_mask(img, mask):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()