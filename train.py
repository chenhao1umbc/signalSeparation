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
from unet import UNet

from utils.dataset import PsdDatasetWithClass
from torch.utils.data import DataLoader, random_split, RandomSampler


dir_img = 'data/imgs/'
dir_mask = 'data/masks/'
dir_checkpoint = 'checkpoints/'
dir_mixture = 'datasets/dataset_0426_14000_128x20/mixture_dataset_multiple/mixture_data_14000.pickle'
dir_list_label = ['datasets/dataset_0426_14000_128x20/component/Blt.mat.pickle',
                  'datasets/dataset_0426_14000_128x20/component/Zigbee.mat.pickle',
                  'datasets/dataset_0426_14000_128x20/component/ZigbeeASK.mat.pickle',
                  'datasets/dataset_0426_14000_128x20/component/ZigbeeBPSK.mat.pickle']
dir_train_sample_pickle = 'datasets/dataset_0426_14000_128x20/train_set.pickle'
dir_val_sample_pickle = 'datasets/dataset_0426_14000_128x20/val_set.pickle'
training_set_visualization_file_path = 'train_set_visualization.pickle'
loss_storage_file_path = 'scoreFile.pickle'
loss_storage_file_path_val = 'scoreFileVal.pickle'



def train_net(net,
              device,
              epochs=5,
              batch_size=10,
              lr=0.01,
              val_percent=0.001,
              save_cp=True,
              img_scale=0.5,
              dir_mixture=dir_mixture,
              dir_list_label=dir_list_label,
              override=False):

    dataset = PsdDatasetWithClass(dir_mixture, dir_list_label)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    all_scores_train = []
    all_scores_val = []
    if not override:
        print(f'Regenerate random dataset from files!\n'
              f'mixture file path:{dir_mixture}')
        train_loader = DataLoader(train, batch_size=batch_size, num_workers=8, pin_memory=True,
                                  sampler=RandomSampler(train, replacement=True, num_samples=2000))
        val_loader = DataLoader(val, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=True,
                                sampler=RandomSampler(val,replacement=True, num_samples=200))

        train_sample_file = open(dir_train_sample_pickle, 'wb')
        val_sample_file = open(dir_val_sample_pickle, 'wb')
        pickle.dump(train_loader, train_sample_file)
        pickle.dump(val_loader, val_sample_file)
        train_sample_file.close()
        val_sample_file.close()

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

    #writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    batch_index = 0
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        batch_num = 2000/batch_size
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='samples') as pbar:
            for batch in train_loader:
                batch_index += 1

                imgs = batch['mixture'].unsqueeze(1)
                true_masks = batch['source_labels'][:, 0, :, :].unsqueeze(1)

                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                
                masks_pred = net(imgs)

                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()

                '''
                if batch_index == 500:
                    pickle_file = open(training_set_visualization_file_path, 'wb')
                    pickle.dump({'component_output': masks_pred,
                                 'class_output': None,
                                 'mixture': imgs,
                                 'component_label': true_masks,
                                 'class_label': None,
                                 'classify_loss': None,
                                 'component_loss': loss.item()}, pickle_file)
                    pickle_file.close()
                    print(f'training visualization file stored! File path:{training_set_visualization_file_path}')

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                '''

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])

            val_score = eval_net(net, val_loader, device)
            scheduler.step(val_score)
            all_scores_train.append(epoch_loss/batch_num)
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

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    
    #Changed channel number to 1 for STFT images
    net = UNet(n_channels=1, n_classes=1, bilinear=True)
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
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  override=True)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
