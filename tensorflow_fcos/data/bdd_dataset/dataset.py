import tensorflow as tf
from data.encode import load_data


def create_dataset(H, W, tf_records_pattern, batch_size):
    autotune = tf.data.experimental.AUTOTUNE
    train_files = tf.data.Dataset.list_files(tf_records_pattern)
    dataset = train_files.interleave(tf.data.TFRecordDataset,
                                     cycle_length=16,
                                     block_length=16,
                                     num_parallel_calls=autotune)
    dataset = dataset.shuffle(512)
    dataset = dataset.map(
        load_data(H, W), num_parallel_calls=autotune)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(autotune)
    return dataset


def dataset_fn(H, W, data_dir, batch_size):
    train_tf_records_pattern = data_dir + '/train*'
    val_tf_records_pattern = data_dir + '/val*'
    train_dataset = \
        create_dataset(H, W, train_tf_records_pattern, batch_size)
    val_dataset = \
        create_dataset(H, W, val_tf_records_pattern, batch_size)
    num_train_images = 70000
    num_val_images = 10000
    return train_dataset, val_dataset, num_train_images, num_val_images
