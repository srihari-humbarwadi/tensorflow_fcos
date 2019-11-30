import tensorflow as tf
from tensorflow.keras.initializers import Constant
from tensorflow.keras import backend as K


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
        scaled_inputs = tf.multiply(self.scale, x)
        return tf.exp(scaled_inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(Scale, self).get_config()
        return config
