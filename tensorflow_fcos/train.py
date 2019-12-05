from data.bdd_dataset.dataset import dataset_fn
from models.fcos import FCOS
import os
import tensorflow as tf

print('TensorFlow:', tf.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')


config = {
    'mode': 'train',
    'distribute_strategy': strategy,
    'image_height': 720,
    'image_width': 1280,
    'num_classes': 10,
    'dataset_fn': dataset_fn,
    'data_dir': './tfrecords',
    'batch_size': 4,
    'epochs': 250,
    'learning_rate': 1e-4,
    'model_dir': './model_files',
    'tensorboard_log_dir': './logs'
}
fcos_model = FCOS(config)
fcos_model.train()
