import tensorflow as tf


def conv_block(input_tensor=None,
               filters=None,
               kernel_size=None,
               padding='same',
               strides=1,
               w_init='he_normal',
               bn_act=True,
               name_prefix=None):
    _x = tf.keras.layers.Conv2D(filters=filters,
                                kernel_size=kernel_size,
                                padding=padding,
                                strides=strides,
                                kernel_initializer=w_init,
                                name='{}_conv_{}x{}'.format(name_prefix,
                                                            kernel_size,
                                                            kernel_size))(input_tensor)
    if bn_act:
        _x = tf.keras.layers.BatchNormalization(
            name='{}_bn'.format(name_prefix))(_x)
        _x = tf.keras.layers.ReLU(name='{}_relu'.format(name_prefix))(_x)
    return _x
