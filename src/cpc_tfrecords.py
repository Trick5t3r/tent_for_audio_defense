#! /usr/bin/env python
"""
Encode audio sample data using learned Contrastive Predictive Coding.

See this repo: https://github.com/pat-coady/contrast-pred-code

Input: `TFRecords/samples/*.tfr`, Output `TFRecords/cpc-enc/*.tfr`
"""
import dataset
from pathlib import Path
import numpy as np
import argparse

from genc import genc_model
from tfrecords import MyTFRWriter
from tfrecords import serialized_example


def build_genc():
    dim_z = 40  # dimension of latent embedding space
    genc = genc_model(dim_z=dim_z)
    zeros = np.zeros((64, 16000), dtype=np.float32)
    genc(zeros)  # send dummy batch through model to force Model.build()
    genc.load_weights(str(Path.cwd() / 'genc.h5'))

    return genc


def build_writer_config(config):
    return {
        'output_dir': config['output_dir'],
        'ds_type': 'cpc-enc',
    }


def build_dataset_reader_config(config):
    return {
        'data_dir': config['data_dir'],
        'ds_type': 'samples',
        'batch_sz': 1,
    }


def main(config):
    genc = build_genc()
    writer_config = build_writer_config(config)
    reader_config = build_dataset_reader_config(config)
    for mode in ['train', 'val', 'test']:
        writer = MyTFRWriter(writer_config, mode)
        ds = dataset.build_dataset(reader_config, mode)
        for batch in ds:
            x, y = batch
            x = genc(x)  # shape = (batch_sz, 63, dim_z)
            for idx in range(x.shape[0]):
                example = serialized_example(x[idx, ...], y[idx])
                writer.write(example)
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build TFRecords dataset.')
    parser.add_argument(
        '--data_dir', type=str,
        help='Path to training data (default = ~/Data/speech_commands_v0.01).',
        default=str(Path.home() / 'Data' / 'speech_commands' / 'tfrecords'))
    parser.add_argument(
        '--output_dir', type=str,
        help='Path to training data (default = ~/Data/speech_commands_v0.01).',
        default=str(Path.home() / 'Data' / 'speech_commands' / 'tfrecords'))

    main(vars(parser.parse_args()))
