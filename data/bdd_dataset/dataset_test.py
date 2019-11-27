import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def draw_boxes_cv2(image, bbox_list):
    img = np.uint8(image).copy()
    bbox_list = np.int32(bbox_list)
    for box in bbox_list:
        img = cv2.rectangle(img, (box[0], box[1]),
                            (box[2], box[3]), [30, 15, 200], 2)
    return img


feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'xmins': tf.io.VarLenFeature(tf.float32),
    'ymins': tf.io.VarLenFeature(tf.float32),
    'xmaxs': tf.io.VarLenFeature(tf.float32),
    'ymaxs': tf.io.VarLenFeature(tf.float32),
    'labels': tf.io.VarLenFeature(tf.float32)
}


@tf.function
def parse_example(example_proto):
    parsed_example = tf.io.parse_single_example(
        example_proto, feature_description)
    image = tf.image.decode_jpeg(parsed_example['image'], channels=3)
    bboxes = tf.stack([
        tf.sparse.to_dense(parsed_example['xmins']) * W,
        tf.sparse.to_dense(parsed_example['ymins']) * H,
        tf.sparse.to_dense(parsed_example['xmaxs']) * W,
        tf.sparse.to_dense(parsed_example['ymaxs']) * H
    ], axis=-1)
    class_ids = tf.reshape(tf.sparse.to_dense(
        parsed_example['labels']), [-1, 1])
    return image, bboxes, class_ids


if __name__ == '__main__':
    H, W = 720, 1280
    autotune = tf.data.experimental.AUTOTUNE
    train_files = tf.data.Dataset.list_files('tfrecords/train*')
    train_dataset = train_files.interleave(tf.data.TFRecordDataset,
                                           cycle_length=4,
                                           block_length=16,
                                           num_parallel_calls=autotune)
    train_dataset = train_dataset.map(
        parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    for x in train_dataset.take(1):
        img = x[0].numpy()
        boxes = x[1].numpy()
        img_ = draw_boxes_cv2(img, boxes)
        plt.imsave('data/bdd_dataset/test.png', img_)
