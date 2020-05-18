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
        self.classify_nets = dict()

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

        # todo: refining networks
        '''for i in range(len(refine_model_paths)):
            init_net = UNet(n_channels=1, n_classes=1)
            init_net.to(device=device)
            init_net.load_state_dict(torch.load(args.model, map_location=device))
            refine_model = refine_model_paths[i]'''

        # todo: classify networks

        logging.info("Model loaded !")

    @staticmethod
    def psd_model(net, psd_mixture):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        psd_mixture = psd_mixture.to(device=device, dtype=torch.float32)
        output = net(psd_mixture)
        return output.cpu().detach().numpy()

    def process(self, iteration, psd_mixture, component_label):
        psd_sources = torch.zeros([batch_size, nb_sources, sample_length, freq_bin])
        psd_sources = psd_sources.to(device=self.device, dtype=torch.float32)
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
                    psd_source = psd_sources[:, source_index:source_index+1, :, :]

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
            criterion_component = nn.MSELoss().cuda()
            for source_index in range(nb_sources):
                source_loss = criterion_component(psd_sources[:, source_index:source_index+1, :, :].cuda(),
                                                  component_label[:, source_index:source_index+1, :, :].cuda())
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

    #init_model_paths = glob.glob(args.init_model + '*.pth')
    init_model_paths = ['init_models/blt.pth', 'init_models/ZigbeeOQPSK.pth',
                        'init_models/ZigbeeASK.pth', 'init_models/ZigbeeBPSK.pth']
    refine_model_paths = glob.glob(args.refine_models + '*.pth')
    class_model_paths = glob.glob(args.class_model + '*.pth')

    print(init_model_paths)


    em_capsule = EMCapsule(init_model_paths=init_model_paths,
                           refine_model_paths=refine_model_paths,
                           classify_model_paths=class_model_paths)

    em_capsule.process(iteration=args.iterations, psd_mixture=mixture, component_label=component_label)
