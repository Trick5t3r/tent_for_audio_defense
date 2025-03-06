import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    """
    Convolution 1D causale : applique un padding à gauche pour que la sortie 
    à un instant donné ne dépende que des entrées passées et actuelles.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1, bias=True):
        super(CausalConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        # On ne spécifie pas de padding dans Conv1d car il sera appliqué manuellement.
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, dilation=dilation, bias=bias)
        # Initialisation "He uniform" (Kaiming uniform) adaptée à la ReLU.
        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        # Calcul du padding nécessaire : (kernel_size - 1) * dilation
        pad = (self.kernel_size - 1) * self.dilation
        # F.pad applique un padding sous forme (pad_left, pad_right)
        x = F.pad(x, (pad, 0))
        return self.conv(x)


class CPCEncoder(nn.Module):
    """
    Modèle d'encodeur pour le Contrastive Predictive Encoding.
    Il attend une entrée de forme (N, T) et renvoie une séquence encodée de forme (N, T//256, dim_z).
    """
    def __init__(self, dim_z):
        super(CPCEncoder, self).__init__()
        # La première couche attend une entrée avec 1 canal.
        self.conv1 = CausalConv1d(1, 64, kernel_size=8, stride=8)
        self.conv2 = CausalConv1d(64, 64, kernel_size=4, stride=4)
        self.conv3 = CausalConv1d(64, 64, kernel_size=4, stride=4)
        self.conv4 = CausalConv1d(64, dim_z, kernel_size=4, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Si x est de forme (N, T) alors on ajoute la dimension des canaux.
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # Continuer avec les convolutions...
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)  # pas d'activation ici
        x = x.transpose(1, 2)  # pour obtenir la forme (N, T//..., dim_z)
        return x


def genc_model(dim_z):
    return CPCEncoder(dim_z)
