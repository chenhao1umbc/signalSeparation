# Data and module loading
import norbert
import numpy as np
import logging
import argparse
import torch
import glob
import pickle
from unet import UNet

pickle_file_path = 'EM_output_iter_'
train_set_file_path = 'train_set_visualization.pickle'


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
    parser.add_argument('--iterations', '-iter', default='2',
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
        return output.detach().numpy()

    @classmethod
    def process(cls, iteration, psd_mixture, component_label):

        for it in range(iteration):
            logging.info(f"EM iteration {it}/{iteration}")
            source_index = 0
            component_label.detach().numpy().shape
            psd_sources = np.array(component_label.detach().numpy().shape)

            # Psd modeling
            for label in cls.init_nets:
                psd_source = cls.psd_model(cls.init_nets[label], psd_mixture)
                psd_sources[:, :, :, source_index] = psd_source
                source_index += 1

            # Applying Wiener filter
            stft_mixture = np.sqrt(psd_mixture)
            stft_sources = norbert.wiener(psd_sources, stft_mixture)

            # Extracting covariance matrix and psd
            for source_index in stft_sources.shape[:3]:
                psd_source, cov_matrix = norbert.get_local_gaussian_model(stft_sources[:, :, :, source_index])
                psd_sources[:, :, :, source_index] = psd_source

            pickle_file = open(pickle_file_path + str(it), 'wb')
            pickle.dump({'component_output': psd_sources,
                         'component_label:': component_label,
                         'mixture': psd_mixture,
                         'classify_loss': None}, pickle_file)
            pickle_file.close()

        return None


if __name__ == "__main__":

    input_pickle_file_path = open(train_set_file_path, 'rb')
    train_set_info = pickle.load(input_pickle_file_path)
    mixture = train_set_info['mixture']
    component_label = train_set_info['component_label']

    args = get_args()

    init_model_paths = glob.glob(args.init_model + '*.pth')
    refine_model_paths = glob.glob(args.refine_models + '*.pth')
    class_model_paths = glob.glob(args.class_model + '*.pth')

    em_capsule = EMCapsule(init_model_paths=init_model_paths,
                           refine_model_paths=refine_model_paths,
                           classify_model_paths=class_model_paths)

    em_capsule.process(iteration=args.iterations, psd_mixture=mixture, component_label=component_label)
