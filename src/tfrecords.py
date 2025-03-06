#! /usr/bin/env python
"""
Build training and validation sets in TFRecord format.

30 classes (keywords), no unknown or background_noise class for now.
"""
import tensorflow as tf
import hashlib
import os
import re
import random
from pathlib import Path
import argparse
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# from: https://www.tensorflow.org/alpha/tutorials/load_data/tf_records
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# from: README.md, Speech Commands Data Set v0.01
# (Fixed a few minor items --pco)
def which_set(filename, validation_percentage=10, testing_percentage=10):
    """Determines which data partition the file should belong to.

    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.

    Args:
      filename: File path of the data sample.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.

    Returns:
      String, one of 'training', 'validation', or 'testing'.
    """
    max_num_waves_per_class = 2 ** 27 - 1  # ~134M
    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name).encode('utf-8')
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(hash_name).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (max_num_waves_per_class + 1)) *
                       (100.0 / max_num_waves_per_class))
    if percentage_hash < validation_percentage:
        result = 'val'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'test'
    else:
        result = 'train'
    return result


# Start of my code
def get_all_filenames(config):
    """Return list of all training files except '_background_noise_"""
    path = Path(config['data_dir'])
    filenames = path.rglob('*.wav')
    filenames = map(lambda fn: str(fn), filenames)

    return list(filter(lambda x: 'background' not in x, filenames))


def build_class_dict(config):
    """Key = class (str), value = integer (0 to num_classes-1)."""
    path = Path(config['data_dir'])
    folders = path.glob('*')
    folders = filter(lambda f: f.is_dir(), folders)
    folders = map(lambda x: x.stem, folders)
    folders = list(filter(lambda x: 'background' not in x, folders))
    folders.sort()
    class_dict = {}
    for i, folder in enumerate(folders):
        class_dict[folder] = i

    return class_dict


def get_class_int(filename, class_dict):
    """Generate integer class label."""
    class_str = os.path.basename(os.path.dirname(filename))
    return class_dict[class_str]


class MyTFRWriter(object):
    def __init__(self, config, mode, shard_size=256):
        self.data_type = config['ds_type']
        self.mode = mode
        self.shard_size = shard_size
        path = Path(config['output_dir']) / config['ds_type']
        path.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.count = 0
        self.shard_num = 0
        self.writer = tf.io.TFRecordWriter(self._tfr_name())

    def write(self, example):
        self.writer.write(example)
        self.count += 1
        if self.count % self.shard_size == 0:
            self.writer.close()
            self.shard_num += 1
            self.writer = tf.io.TFRecordWriter(self._tfr_name())

    def close(self):
        self.writer.close()

    def _tfr_name(self):
        path = Path(self.path) / '{}_{:04d}.tfr'.format(self.mode, self.shard_num)
        return str(path)


def serialized_example(x, y):
    feature = {
        'x': _bytes_feature(tf.io.serialize_tensor(x)),
        'y': _int64_feature(y)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()


def to_log_mel_spectro(samples, m):
    samples = tf.reshape(samples, (-1,))
    stft = tf.signal.stft(samples, frame_length=512, frame_step=256)
    spectrogram = tf.abs(stft)
    return tf.math.log(spectrogram @ m + 1e-6)


def main(config):
    logger.info(f"Starting TFRecords creation with config: {config}")
    
    # Conversion des chemins en chemins absolus
    data_dir = Path(config['data_dir']).resolve()
    output_dir = Path(config['output_dir']).resolve()
    
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # Création du dossier de sortie
    output_dir.mkdir(parents=True, exist_ok=True)
    
    m = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=40,
        num_spectrogram_bins=257,
        sample_rate=16000,
        lower_edge_hertz=40,
        upper_edge_hertz=8000)
    
    class_dict = build_class_dict(config)
    logger.info(f"Found {len(class_dict)} classes: {list(class_dict.keys())}")
    
    filenames = get_all_filenames(config)
    logger.info(f"Found {len(filenames)} audio files")
    
    random.seed(0)
    random.shuffle(filenames)
    
    writers = {mode: MyTFRWriter(config, mode=mode)
               for mode in ['train', 'val', 'test']}
    
    for i, filename in enumerate(filenames):
        if i % 1000 == 0:
            logger.info(f"Processing file {i}/{len(filenames)}: {filename}")
        
        try:
            rawdata = tf.io.read_file(filename)
            kw_samples = tf.audio.decode_wav(rawdata).audio
            
            if kw_samples.shape[0] != 16000:
                logger.warning(f"Skipping {filename}: wrong number of samples ({kw_samples.shape[0]})")
                continue
            
            x = kw_samples  # default to samples
            if config['ds_type'] == 'log-mel':
                x = to_log_mel_spectro(kw_samples, m)
            elif config['ds_type'] == 'mfcc':
                x = to_log_mel_spectro(kw_samples, m)
                x = tf.signal.mfccs_from_log_mel_spectrograms(x)
            
            y = get_class_int(filename, class_dict)
            kw_split = which_set(filename)
            example = serialized_example(x, y)
            writers[kw_split].write(example)
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            continue
    
    for writer in writers.values():
        writer.close()
    
    logger.info("TFRecords creation completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build TFRecords dataset.')
    parser.add_argument(
        '--ds_type', type=str,
        help='Choose "log-mel", "mfcc", or "samples".',
        default='log-mel')
    parser.add_argument(
        '--data_dir', type=str,
        help='Path to training data (default = ~/Data/speech_commands_v0.01).',
        default=str(Path.home() / 'Data' / 'speech_commands_v0.01'))
    parser.add_argument(
        '--output_dir', type=str,
        help='Path to training data (default = ~/Data/speech_commands/tfrecords).',
        default=str(Path.home() / 'Data' / 'speech_commands' / 'tfrecords'))

    main(vars(parser.parse_args()))
