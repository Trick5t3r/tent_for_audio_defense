"""
Construction du dataset PyTorch à partir des données Kaggle KWS traitées.

https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html

Utiliser tfrecords.py pour construire les TFRecords.
"""
import torch
from torch.utils.data import Dataset
import tensorflow as tf
from pathlib import Path
import numpy as np

class SpeechCommandsDataset(Dataset):
    def __init__(self, config, mode):
        self.config = config
        self.mode = mode
        self.filenames = self._build_filelist()
        if mode == 'train':
            np.random.shuffle(self.filenames)
        
        # Création d'un dataset TensorFlow pour la lecture des TFRecords
        self.tf_dataset = tf.data.TFRecordDataset(self.filenames, num_parallel_reads=8)
        self.tf_dataset = self.tf_dataset.map(
            lambda example: self._decode(example, config['ds_type']),
            num_parallel_calls=8
        )
        if mode == 'train':
            self.tf_dataset = self.tf_dataset.shuffle(1024)
        
        # Conversion en liste pour PyTorch
        self.data = []
        self.labels = []
        for x, y in self.tf_dataset:
            self.data.append(x.numpy())
            self.labels.append(y.numpy())
        
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
    
    def _decode(self, example, ds_type):
        """Parse l'exemple depuis `serialized_example`."""
        epsilon = 1e-5
        feature_description = {
            'x': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'y': tf.io.FixedLenFeature([], tf.int64, default_value=0)
        }
        example = tf.io.parse_single_example(example, feature_description)
        x = tf.io.parse_tensor(example['x'], out_type=tf.float32)
        
        if ds_type == 'samples':
            x = tf.reshape(x, (16000,))
            x = x - tf.reduce_mean(x)
            max_x = tf.reduce_max(x)
            min_x = tf.reduce_min(x)
            scale = tf.maximum(-min_x, max_x)
            x = x / (scale + epsilon)
        elif ds_type in ['mfcc', 'log-mel']:
            x = tf.reshape(x, (61, 40, 1))
            # nombres magiques pour approximativement mettre les entrées à l'échelle [-1, 1]
            x = tf.clip_by_value(x, -10.0, 5.0)
            x = (x + 2.5) / 7.5
        elif ds_type == 'cpc-enc':
            # nombre magique pour approximativement mettre les entrées à l'échelle [-1, 1]
            x = tf.reshape(x, (63, 40, 1)) / 4
        else:
            return None
        
        y = tf.cast(example['y'], tf.int32)
        return x, y
    
    def _build_filelist(self):
        """Construit la liste des fichiers TFRecord."""
        path = Path(self.config['data_dir']) / self.config['ds_type']
        filenames = path.rglob(f'{self.mode}_*.tfr')
        return sorted(list(map(str, filenames)))  # tri déterministe
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx]).float()
        y = torch.from_numpy(self.labels[idx]).long()
        
        # Conversion des dimensions pour PyTorch
        if self.config['ds_type'] == 'samples':
            x = x.view(16000)
        elif self.config['ds_type'] in ['mfcc', 'log-mel']:
            x = x.view(61, 40, 1)
        elif self.config['ds_type'] == 'cpc-enc':
            x = x.view(63, 40, 1)
        
        return x, y

def build_dataset(config, mode):
    """Construit le dataset PyTorch."""
    return SpeechCommandsDataset(config, mode) 