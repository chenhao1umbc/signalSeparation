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
