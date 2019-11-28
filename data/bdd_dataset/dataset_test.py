import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tf_record_parser import parse_example


def draw_boxes_cv2(image, bbox_list):
    img = np.uint8(image).copy()
    bbox_list = np.int32(bbox_list)
    for box in bbox_list:
        img = cv2.rectangle(img, (box[0], box[1]),
                            (box[2], box[3]), [30, 15, 200], 2)
    return img


if __name__ == '__main__':
    H, W = 720, 1280
    autotune = tf.data.experimental.AUTOTUNE

    train_files = tf.data.Dataset.list_files('../../tfrecords/train*')
    train_dataset = train_files.interleave(tf.data.TFRecordDataset,
                                           cycle_length=4,
                                           block_length=16,
                                           num_parallel_calls=autotune)
    train_dataset = train_dataset.map(
        parse_example(H, W), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    for x in train_dataset.take(1):
        img = x[0].numpy()
        boxes = x[1].numpy()
        img_ = draw_boxes_cv2(img, boxes)
        plt.imsave('test.png', img_)
