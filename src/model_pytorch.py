"""
Modèles PyTorch pour la détection de mots-clés
"""
import torch
import torch.nn as nn
from pathlib import Path
from genc import genc_model

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
            
            self.reshape = lambda x: x.reshape(-1, 1, 40, x.size(1))
            self.crop = lambda x: x[:, :, 5:, :]
            in_channels = 1
        else:
            in_channels = 1

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
