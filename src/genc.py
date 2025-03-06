"""
Contrastive Predictive Encoding encoder model from:
https://github.com/pat-coady/contrast-pred-code

Not ideal, but manually copied model to this repo to make this project
self-contained.
"""
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Lambda


def genc_model(dim_z):
    """Build CPC encoder model (i.e. g_enc).

    Parameters
    ----------
    dim_z : int
        dimension of latent encoding

    Returns
    -------
    model : keras.Model
        Model that expects time sequence input of shape (N, T) and returns an encoded
        sequence of shape (N, T//256, dim_Z). N is the batch dimension, T is the
        sequence length.
    """
    model = keras.Sequential(name='genc')
    model.add(Lambda(lambda x: backend.expand_dims(x, axis=-1)))  # add dim for Conv1D
    model.add(Conv1D(filters=64, kernel_size=8, strides=8, padding='causal',
                     activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv1D(filters=64, kernel_size=4, strides=4, padding='causal',
                     activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv1D(filters=64, kernel_size=4, strides=4, padding='causal',
                     activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv1D(filters=dim_z, kernel_size=4, strides=2, padding='causal',
                     kernel_initializer='he_uniform'))

    return model
