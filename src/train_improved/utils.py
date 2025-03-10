"""
Utilities. Partially based on code from
https://github.com/modestyachts/CIFAR-10.1
"""
import io
import json
import os
import pickle

import numpy as np
import pathlib

from models.wideresnet import WideResNet
from models.shake_shake import ShakeNet
from models.cifar_resnet import ResNet

import torch
from torch.nn import Sequential, Module


import torch
import torch.nn as nn
import torchaudio
from torchaudio.transforms import MelSpectrogram

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels) if use_batch_norm else None
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.relu(x)

class AudioCNN(nn.Module):
    def __init__(self, num_classes=12, batch_norm=True):
        super().__init__()
        self.use_batch_norm = batch_norm

        # Ajout du MelSpectrogram directement dans le modèle
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=16000,      
            n_mels=40,              
            n_fft=512,              
            hop_length=160,         
            f_min=40,               
            f_max=8000              
        )

        # Architecture CNN adaptée aux spectrogrammes
        self.conv1 = Conv2dBlock(1, 16, self.use_batch_norm)
        self.conv2 = Conv2dBlock(16, 16, self.use_batch_norm)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = Conv2dBlock(16, 32, self.use_batch_norm)
        self.conv4 = Conv2dBlock(32, 32, self.use_batch_norm)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = Conv2dBlock(32, 64, self.use_batch_norm)
        self.conv6 = Conv2dBlock(64, 64, self.use_batch_norm)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Calcul de la taille après convolution
        self._calculate_conv_output_size()

        # Couches denses
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.conv_output_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def _calculate_conv_output_size(self):
        """Calcul automatique de la taille de sortie après les convolutions"""
        x = torch.zeros(1, 1, 40, 101)  # Taille d'entrée : [batch, channel, n_mels, time]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)
        self.conv_output_size = x.numel()

    def forward(self, x):
        """Appliquer MelSpectrogram puis passer dans le CNN"""
        x = self.mel_spectrogram(x)  # Transformer l’audio en spectrogramme Mel

        # Normalisation et transformation log-mel
        x = torch.log(torch.clamp(x, min=1e-5))
        x = torch.clamp(x, -10.0, 5.0)
        x = (x + 2.5) / 7.5  # Normalisation entre 0 et 1

        # Passage dans le CNN
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)

        # Couches Fully Connected
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

def get_model(model_name, num_classes=12, normalize_input=False):
    """Retourne l’AudioCNN au lieu d’un modèle image classique"""
    return AudioCNN(num_classes=num_classes)


####OLD ###
cifar10_label_names = ['airplane', 'automobile', 'bird',
                       'cat', 'deer', 'dog', 'frog', 'horse',
                       'ship', 'truck']

def get_model_2(name, num_classes=10, normalize_input=False):
    name_parts = name.split('-')
    if name_parts[0] == 'wrn':
        depth = int(name_parts[1])
        widen = int(name_parts[2])
        model = WideResNet(
            depth=depth, num_classes=num_classes, widen_factor=widen)
        
    elif name_parts[0] == 'ss':
        model = ShakeNet(dict(depth=int(name_parts[1]),
                              base_channels=int(name_parts[2]),
                              shake_forward=True, shake_backward=True,
                              shake_image=True, input_shape=(1, 3, 32, 32),
                              n_classes=num_classes,
                              ))
    elif name_parts[0] == 'resnet':
        model = ResNet(num_classes=num_classes, depth=int(name_parts[1]))
    else:
        raise ValueError('Could not parse model name %s' % name)

    if normalize_input:
        model = Sequential(NormalizeInput(), model)

    return model


class NormalizeInput(Module):
    def __init__(self, mean=(0.4914, 0.4822, 0.4465),
                 std=(0.2023, 0.1994, 0.2010)):
        super().__init__()

        self.register_buffer('mean', torch.Tensor(mean).reshape(1, -1, 1, 1))
        self.register_buffer('std', torch.Tensor(std).reshape(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std

# TODO: decide whether to remove all code below this line
# Should we add some description of how these files
# were obtained?
def load_tinyimage_subset(other_data_path,
                          version_string='v7'):
    image_data_filename = 'tinyimage_subset_data'
    if version_string != '':
        image_data_filename += '_' + version_string
    image_data_filename += '.pickle'
    image_data_filepath = os.path.abspath(os.path.join(other_data_path, image_data_filename))
    indices_filename = 'tinyimage_subset_indices'
    if version_string != '':
        indices_filename += '_' + version_string
    indices_filename += '.json'
    indices_filepath = os.path.abspath(os.path.join(other_data_path, indices_filename))
    print('Loading indices from file {}'.format(indices_filepath))
    assert pathlib.Path(indices_filepath).is_file()
    print('Loading image data from file {}'.format(image_data_filepath))
    assert pathlib.Path(image_data_filepath).is_file()
    with open(indices_filepath, 'r') as f:
        indices = json.load(f)
    with open(image_data_filepath, 'rb') as f:
        image_data = pickle.load(f)
    num_entries = 0
    for kw, kw_indices in indices.items():
        for entry in kw_indices:
            assert entry['tinyimage_index'] in image_data
            num_entries += 1
    assert num_entries == len(image_data)
    return indices, image_data


def load_cifar10_by_keyword(unique_keywords=True, version_string='v7'):
    cifar10_keywords = load_cifar10_keywords(unique_keywords=unique_keywords,
                                             lists_for_unique=True,
                                             version_string=version_string)
    cifar10_by_keyword = {}
    for ii, keyword_entries in enumerate(cifar10_keywords):
        for entry in keyword_entries:
            cur_keyword = entry['nn_keyword']
            if not cur_keyword in cifar10_by_keyword:
                cifar10_by_keyword[cur_keyword] = []
            cifar10_by_keyword[cur_keyword].append(ii)
    return cifar10_by_keyword


def load_cifar10_keywords(other_data_path, 
                          unique_keywords=True,
                          lists_for_unique=False,
                          version_string='v7'):
    filename = 'cifar10_keywords'
    if unique_keywords:
        filename += '_unique'
    if version_string != '':
        filename += '_' + version_string
    filename += '.json'
    keywords_filepath = os.path.abspath(os.path.join(other_data_path, filename))
    print('Loading keywords from file {}'.format(keywords_filepath))
    assert pathlib.Path(keywords_filepath).is_file()
    with open(keywords_filepath, 'r') as f:
        cifar10_keywords = json.load(f)
    if unique_keywords and lists_for_unique:
        result = []
        for entry in cifar10_keywords:
            result.append([entry])
    else:
        result = cifar10_keywords
    assert len(result) == 60000
    return result


def load_distances_to_cifar10(version_string='v7'):
    data_path = os.path.join(os.path.dirname(__file__), '../data/')
    filename = 'tinyimage_cifar10_distances'
    if version_string != '':
        filename += '_' + version_string
    filename += '.json'
    filepath = os.path.abspath(os.path.join(data_path, filename))
    print('Loading distances from file {}'.format(filepath))
    assert pathlib.Path(filepath).is_file()
    with open(filepath, 'r') as f:
        tmp = json.load(f)
    if version_string == 'v4':
        assert len(tmp) == 372131
    elif version_string == 'v6':
        assert len(tmp) == 1646248
    elif version_string == 'v7':
        assert len(tmp) == 589711
    result = {}
    for k, v in tmp.items():
        result[int(k)] = v
    return result


def load_new_test_data_indices(version_string='v7'):
    data_path = os.path.join(os.path.dirname(__file__), '../data/')
    filename = 'cifar10.1'
    if version_string == '':
        version_string = 'v7'
    if version_string in ['v4', 'v6', 'v7']:
        filename += '_' + version_string
    else:
        raise ValueError('Unknown dataset version "{}".'.format(version_string))
    ti_indices_data_path = os.path.join(os.path.dirname(__file__), '../data/')
    ti_indices_filename = 'cifar10.1_' + version_string + '_ti_indices.json'
    ti_indices_filepath = os.path.abspath(os.path.join(ti_indices_data_path, ti_indices_filename))
    print('Loading Tiny Image indices from file {}'.format(ti_indices_filepath))
    assert pathlib.Path(ti_indices_filepath).is_file()
    with open(ti_indices_filepath, 'r') as f:
        tinyimage_indices = json.load(f)
    assert type(tinyimage_indices) is list
    if version_string == 'v6' or version_string == 'v7':
        assert len(tinyimage_indices) == 2000
    elif version_string == 'v4':
        assert len(tinyimage_indices) == 2021
    return tinyimage_indices
