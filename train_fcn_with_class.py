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
