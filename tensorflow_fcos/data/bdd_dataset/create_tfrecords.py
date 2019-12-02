import tensorflow as tf
from glob import glob
import numpy as np
from tqdm import tqdm_notebook
import json

train_image_paths = sorted(
    glob('bdd100k/images/100k/train/*'))
train_label_paths = sorted(
    glob('bdd100k/labels/100k/train/*'))
validation_image_paths = sorted(
    glob('bdd100k/images/100k/val/*'))
validation_label_paths = sorted(
    glob('bdd100k/labels/100k/val/*'))

print('Found training {} images'.format(len(train_image_paths)))
print('Found training {} labels'.format(len(train_label_paths)))
print('Found validation {} images'.format(len(validation_image_paths)))
print('Found validation {} labels'.format(len(validation_label_paths)))

class_map = {value: idx for idx, value in enumerate(['bus',
                                                     'traffic light',
                                                     'traffic sign',
                                                     'person',
                                                     'bike',
                                                     'truck',
                                                     'motor',
                                                     'car',
                                                     'train',
                                                     'rider'])}
for image, label in zip(train_image_paths, train_label_paths):
    assert image.split(
        '/')[-1].split('.')[0] == label.split('/')[-1].split('.')[0]
for image, label in zip(validation_image_paths, validation_label_paths):
    assert image.split(
        '/')[-1].split('.')[0] == label.split('/')[-1].split('.')[0]


def get_label(label_path, class_map, input_shape=512):
    with open(label_path, 'r') as f:
        temp = json.load(f)
    bbox = []
    class_ids = []
    for obj in temp['frames'][0]['objects']:
        if 'box2d' in obj:
            x1 = obj['box2d']['x1']
            y1 = obj['box2d']['y1']
            x2 = obj['box2d']['x2']
            y2 = obj['box2d']['y2']
            bbox.append(np.array([x1, y1, x2, y2]))
            class_ids.append(class_map[obj['category']])
    bbox = np.array(bbox, dtype=np.float32)
    class_ids = np.array(class_ids, dtype=np.float32)[..., None]
    return np.concatenate([bbox, class_ids], axis=-1)


train_labels = []
validation_labels = []

for path in tqdm_notebook(train_label_paths):
    train_labels.append(get_label(path, class_map))
for path in tqdm_notebook(validation_label_paths):
    validation_labels.append(get_label(path, class_map))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def intlist_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def floatlist_feature(value_list):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value_list))


def make_example(image_path, label):
    with tf.gfile.GFile(image_path, 'rb') as fid:
        raw_image = fid.read()
    dims = [720, 1280]
    bbox = label[:, :4]
    class_id = label[:, 4]
    feature = {
        'image': bytes_feature(raw_image),
        'xmins': floatlist_feature(bbox[:, 0] / dims[1]),
        'ymins': floatlist_feature(bbox[:, 1] / dims[0]),
        'xmaxs': floatlist_feature(bbox[:, 2] / dims[1]),
        'ymaxs': floatlist_feature(bbox[:, 3] / dims[0]),
        'labels': floatlist_feature(class_id)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecord(image_list, labels_list, path=''):
    with tf.io.TFRecordWriter(path) as writer:
        for image_path, label in tqdm_notebook(zip(image_list, labels_list),
                                               total=len(image_list)):
            example = make_example(image_path, label)
            writer.write(example.SerializeToString())


def create_dataset(images, labels, prefix='', folder='data', n_shards=8):
    n_samples = len(images)
    step_size = n_samples // n_shards + 1
    for i in range(0, n_samples, step_size):
        path = '{}/{}_000{}.tfrecords'.format(folder, prefix, i // step_size)
        write_tfrecord(images[i:i + step_size], labels[i:i + step_size], path)


if __name__ == '__main__':
    write_tfrecord(validation_image_paths[:100],
                   validation_labels[:100],
                   path='tfrecords/test.tfrecord')
