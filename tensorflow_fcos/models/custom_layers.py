import tensorflow as tf
from tensorflow.keras.initializers import Constant
from tensorflow.keras import backend as K
from data.encode import get_all_centers


class Scale(tf.keras.layers.Layer):
    def __init__(self, init_value=1.0, **kwargs):
        super(Scale, self).__init__(**kwargs)
        self.init_value = init_value

    def build(self, input_shape):
        self.scale = \
            self.add_weight(name='scale',
                            shape=[1],
                            dtype=K.floatx(),
                            trainable=True,
                            initializer=Constant(value=self.init_value))

    def call(self, x):
        '''
            From the FCOS paper, "since the regression targets are always
            positive we employ exp(x) to map any real number to (0, ∞) on
            the top of the regression branch"
        '''
        scaled_inputs = tf.multiply(self.scale, x)
        return tf.exp(scaled_inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(Scale, self).get_config()
        return config


class Decode(tf.keras.layers.Layer):
    def __init__(self,
                 H,
                 W,
                 num_classes,
                 **kwargs):
        super(Decode, self).__init__(**kwargs)
        self.H = H
        self.W = W

    def build(self):
        self.centers_list = tf.concat(get_all_centers(self.H, self.W),
                                      axis=0)

    def call(self,
             cls_target,
             ctr_target,
             reg_target,
             score_threshold,
             iou_threshold):
        cls_scores = tf.reduce_max(cls_target[0], axis=1)
        cls_ids = tf.argmax(cls_target[0], axis=1)
        score_map = cls_scores * ctr_target[0]

        valid_indices = tf.where(score_map > score_threshold)[:, 0]

        valid_scores = tf.gather(score_map, valid_indices)
        valid_cls_ids = tf.gather(cls_ids, valid_indices)
        valid_centers = tf.gather(self.centers, valid_indices)
        valid_ltrb = tf.gather(reg_target[0], valid_indices)

        decoded_boxes = tf.concat([
            valid_centers - valid_ltrb[:, :2],
            valid_centers + valid_ltrb[:, 2:]], axis=-1)
        nms_indices =  \
            tf.image.non_max_suppression(decoded_boxes,
                                         valid_scores,
                                         max_output_size=300,
                                         iou_threshold=iou_threshold)
        boxes = tf.gather(decoded_boxes, nms_indices)
        scores = tf.gather(valid_scores, nms_indices)
        ids = tf.gather(valid_cls_ids, nms_indices)
        return boxes[None, :], scores[None, :], ids[None, :]

    def get_config(self):
        config = super(Decode, self).get_config()
        config['H'] = self.H
        config['W'] = self.W
        config['score_threshold'] = self.score_threshold
        config['iou_threshold'] = self.iou_threshold
        return config
