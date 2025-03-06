"""
Modèles PyTorch pour la détection de mots-clés
"""
import torch
import torch.nn as nn
from pathlib import Path
from genc import genc_model

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

class CNN(nn.Module):
    def __init__(self, batch_norm=True, dense_final=False, load_genc=False, train_genc=False, ds_type='log-mel'):
        super().__init__()
        self.use_batch_norm = batch_norm
        self.ds_type = ds_type

        if ds_type == 'samples':
            self.genc = genc_model(dim_z=40)
            if load_genc:
                self.genc.load_state_dict(torch.load(Path.cwd() / 'genc.h5'))
            if not train_genc:
                for param in self.genc.parameters():
                    param.requires_grad = False

            self.reshape = lambda x: x.view(-1, 1, 40, x.size(1))
            self.crop = lambda x: x[:, :, 5:, :]
            in_channels = 1
        else:
            in_channels = 1

        # Add MelSpectrogram transformation
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=16000,  # Adjust based on your audio data
            n_mels=40,          # Number of Mel filterbanks
            n_fft=400,          # Size of FFT
            hop_length=160      # Length of hop between STFT windows
        )

        # Architecture CNN
        self.conv1 = Conv2dBlock(in_channels, 16, self.use_batch_norm)
        self.conv2 = Conv2dBlock(16, 16, self.use_batch_norm)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = Conv2dBlock(16, 32, self.use_batch_norm)
        self.conv4 = Conv2dBlock(32, 32, self.use_batch_norm)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = Conv2dBlock(32, 64, self.use_batch_norm)
        self.conv6 = nn.Conv2d(64, 30, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        if self.ds_type == 'samples':
            x = self.genc(x)
            x = self.reshape(x)
            x = self.crop(x)
        else:
            # Apply MelSpectrogram transformation
            x = self.mel_spectrogram(x)
            x = x.unsqueeze(1)  # Add channel dimension

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.global_pool(x)
        return x.squeeze(-1).squeeze(-1)

class TallKernel(nn.Module):
    def __init__(self, batch_norm=True, dense_final=False, load_genc=False, train_genc=False, ds_type='log-mel'):
        super().__init__()
        self.ds_type = ds_type
        
        if ds_type == 'samples':
            self.genc = genc_model(dim_z=40)
            self.reshape = lambda x: x.view(-1, 1, 40, x.size(1))
            self.crop = lambda x: x[:, :, 5:, :]
            in_channels = 1
        else:
            h = 61 if ds_type == 'log-mel' else 63
            in_channels = 1

        # Architecture avec noyaux hauts
        self.conv1 = nn.Conv2d(in_channels, 80, kernel_size=(3, 40), stride=(2, 1))
        self.conv2 = nn.Conv2d(80, 160, kernel_size=(3, 1), stride=(2, 1))
        self.conv3 = nn.Conv2d(160, 160, kernel_size=(3, 1), stride=(2, 1))
        
        if dense_final:
            self.fc1 = nn.Linear(160 * 7 * 1, 90)
            self.fc2 = nn.Linear(90, 30)
            self.relu = nn.ReLU()
        else:
            self.conv4 = nn.Conv2d(160, 30, kernel_size=(3, 1), stride=(2, 1))
            self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        if self.ds_type == 'samples':
            x = self.genc(x)
            x = self.reshape(x)
            x = self.crop(x)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        if hasattr(self, 'fc1'):  # Si dense_final est True
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
        else:
            x = self.conv4(x)
            x = self.global_pool(x)
            x = x.squeeze(-1).squeeze(-1)
        
        return x 