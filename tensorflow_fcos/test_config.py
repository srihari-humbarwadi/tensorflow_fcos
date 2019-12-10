from data.bdd_dataset.dataset import dataset_fn
from models.fcos import FCOS
import os
import tensorflow as tf
from tqdm import tqdm


print('TensorFlow:', tf.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
strategy = tf.distribute.MirroredStrategy()


if __name__ == '__main__':
    data_dir = os.environ['HOME'] + '/datasets/BDD100k'

    config = {
        'mode': 'train',
        'distribute_strategy': strategy,
        'image_height': 720,
        'image_width': 1280,
        'num_classes': 10,
        'dataset_fn': dataset_fn,
        'data_dir': data_dir,
        'batch_size': 4,
        'epochs': 50,
        'learning_rate': 1e-4,
        'model_dir': './model_files',
        'tensorboard_log_dir': './logs',
        'restore_parameters': False
    }

    fcos_model = FCOS(config)

    print('****Dataset Shape')
    for spec in tf.data.experimental.get_structure(fcos_model.train_dataset):
        print(spec)

    @tf.function
    def per_replica_run(images, targets):
        outputs = fcos_model.model(images, training=False)
        return outputs

    @tf.function
    def distributed_run(images, targets):
        return strategy.experimental_run_v2(per_replica_run,
                                            args=(images, targets))

    def run(i):
        for idx, (images, targets) in tqdm(enumerate(fcos_model.train_dataset),
                                           total=i):

            outputs = distributed_run(images, targets)
            if idx == i:
                return outputs

    print('****Dummy Output')
    outputs = run(2000)
    for tensor in outputs:
        print(tensor.shape)
