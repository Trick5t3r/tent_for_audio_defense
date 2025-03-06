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

class CNN(nn.Module):
    def __init__(self, batch_norm=True, num_classes=12):
        super().__init__()
        self.use_batch_norm = batch_norm

        # Add MelSpectrogram transformation
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=16000,      # Sample rate
            n_mels=40,              # Number of mel bands
            n_fft=512,              # FFT size (257 * 2 - 1)
            hop_length=160,         # Hop length
            f_min=40,               # Lower edge of mel bands
            f_max=8000              # Upper edge of mel bands
        )

        # Architecture CNN
        self.conv1 = Conv2dBlock(1, 16, self.use_batch_norm)
        self.conv2 = Conv2dBlock(16, 16, self.use_batch_norm)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = Conv2dBlock(16, 32, self.use_batch_norm)
        self.conv4 = Conv2dBlock(32, 32, self.use_batch_norm)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = Conv2dBlock(32, 64, self.use_batch_norm)
        self.conv6 = Conv2dBlock(64, 64, self.use_batch_norm)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Calculate the size of the output from the convolutional layers
        self._calculate_conv_output_size()

        # Dense layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.conv_output_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def _calculate_conv_output_size(self):
        # Dummy input to calculate the output size
        x = torch.zeros(1, 1, 40, 101)  # Example input size: [batch_size, channels, n_mels, time]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)
        self.conv_output_size = x.numel()  # Flattened size

    def forward(self, x):
        # Apply MelSpectrogram transformation
        x = self.mel_spectrogram(x)

        # Convert to log-mel spectrogram
        x = torch.log(torch.clamp(x, min=1e-5))
        x = torch.clamp(x, -10.0, 5.0)
        x = (x + 2.5) / 7.5

        # CNN layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)

        # Dense layers
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
