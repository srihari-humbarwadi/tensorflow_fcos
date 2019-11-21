from tensorflow.keras.layers import (Conv2D,
                                     BatchNormalization,
                                     ReLU)


def conv_block(input_tensor=None,
               filters=None,
               kernel_size=None,
               strides=1,
               padding='same',
               kernel_init='he_normal',
               bias_init='zeros',
               bn_act=True,
               name_prefix=None):

    _x = Conv2D(filters=filters, kernel_size=kernel_size,
                padding=padding, strides=strides,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                name='{}_conv_{}x{}'.format(name_prefix,
                                            kernel_size,
                                            kernel_size))(input_tensor)
    if bn_act:
        _x = BatchNormalization(
            name='{}_bn'.format(name_prefix))(_x)
        _x = ReLU(name='{}_relu'.format(name_prefix))(_x)
    return _x
