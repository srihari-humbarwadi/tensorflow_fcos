from data.bdd_dataset.dataset import dataset_fn
from models.fcos import FCOS
import os
import tensorflow as tf


print('TensorFlow:', tf.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'

tf.config.optimizer.set_jit(True)
strategy = tf.distribute.MirroredStrategy()
data_dir = os.environ['HOME'] + '/datasets/BDD100k'

config = {
    'mode': 'train',
    'distribute_strategy': strategy,
    'image_height': 720,
    'image_width': 1280,
    'num_classes': 10,
    'dataset_fn': dataset_fn,
    'data_dir': data_dir,
    'batch_size': 8,
    'epochs': 25,
    'learning_rate': 5e-4,
    'checkpoint_prefix': 'ckpt',
    'model_dir': './model_files',
    'tensorboard_log_dir': './logs',
    'log_after': 20,
    'restore_parameters': True
}
fcos_model = FCOS(config)
fcos_model.train()
