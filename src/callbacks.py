"""
Custom Keras callbacks for logging, checkpointing, and lr schedule.
"""
import tensorflow as tf
from tensorflow import keras


class AzureLoggingCallback(tf.keras.callbacks.Callback):
    """Logging for Azure ML framework."""
    def __init__(self, run_logger):
        super(AzureLoggingCallback, self).__init__()
        self.run_logger = run_logger

    def on_epoch_end(self, epoch, logs=None):
        for k in logs:
            self.run_logger.log(k, logs[k])


def build_lr_scheduler(config):
    """Given config, return lr scheduler."""
    def scheduler(epoch):
        scale = 3
        if epoch < 10:
            return config['lr']
        elif epoch < 15:
            return config['lr'] / scale
        elif epoch < 20:
            return config['lr'] / scale ** 2
        elif epoch < 20:
            return config['lr'] / scale ** 3
        else:
            return config['lr'] / scale ** 3

    return scheduler


def build_callbacks(config, run_logger):
    """Build list of callbacks to monitor training."""
    callbacks = []
    cb = keras.callbacks.ModelCheckpoint(config['ckpt_path'] + '/cp-{epoch:04d}.ckpt',
                                         save_weights_only=True)
    callbacks.append(cb)

    cb = keras.callbacks.TensorBoard(config['tblog_path'],
                                     histogram_freq=1,
                                     update_freq=500)
    callbacks.append(cb)

    scheduler = build_lr_scheduler(config)
    cb = keras.callbacks.LearningRateScheduler(scheduler)
    callbacks.append(cb)

    if config['azure_ml']:
        cb = AzureLoggingCallback(run_logger)
        callbacks.append(cb)

    return callbacks
