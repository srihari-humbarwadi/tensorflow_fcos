import tensorflow as tf
from ..layers.blocks import conv_block


class FPN:
    def __init__(self, H=800, W=1024):
        self.backbone = tf.keras.applications.ResNet50V2(
            input_shape=[H, W, 3], weights='imagenet', include_top=False)
        self.create_pyramid_features()

    def create_pyramid_features(self):
        '''
            From the FPN paper, "To start the iteration, we simply attach a
            1×1 convolutional layer on C5 to produce the coarsest resolution
            map. Finally, we append a 3×3 convolution on each merged map to
            generate the final feature map, which is to reduce the aliasing
            effect of upsampling. This final set of feature maps is called
            {P2, P3, P4, P5}, corresponding to {C2, C3, C4, C5} that are
            respectively of the same spatial sizes".
            From the FCOS paper, "P6 and P7 are produced by applying one
            convolutional layer with the stride being 2 on P5 and P6".
        '''
        C5 = self.backbone.get_layer('post_relu').output
        C4 = self.backbone.get_layer('conv4_block6_1_relu').output
        C3 = self.backbone.get_layer('conv3_block4_1_relu').output

        M5 = conv_block(C5, filters=256, kernel_size=1,
                        bn_act=False, name_prefix='C5')
        P5 = conv_block(M5, filters=256, kernel_size=3,
                        bn_act=False, name_prefix='P5')
        M5_upsampled = tf.keras.layers.UpSampling2D(size=(2, 2),
                                                    interpolation='nearest',
                                                    name='M5_upsampled')(M5)

        M4 = conv_block(C4, filters=256, kernel_size=1,
                        bn_act=False, name_prefix='C4')
        M4 = tf.keras.layers.Add(name='M4_M5_add')([M4, M5_upsampled])
        P4 = conv_block(M4, filters=256, kernel_size=3,
                        bn_act=False, name_prefix='P4')
        M4_upsampled = tf.keras.layers.UpSampling2D(size=(2, 2),
                                                    interpolation='nearest',
                                                    name='M4_upsampled')(M4)

        M3 = conv_block(C3, filters=256, kernel_size=1,
                        bn_act=False, name_prefix='C3')
        P3 = tf.keras.layers.Add(name='M3_M4_add')([M3, M4_upsampled])
        P3 = conv_block(P3, filters=256, kernel_size=3,
                        bn_act=False, name_prefix='P3')

        P6 = conv_block(P5, filters=256, kernel_size=3,
                        strides=2, bn_act=False, name_prefix='P6')
        P6_relu = tf.keras.layers.ReLU(name='P6_relu')(P6)
        P7 = conv_block(P6_relu, filters=256, kernel_size=3,
                        strides=2, bn_act=False, name_prefix='P7')

        self.pyramid_features = {
            'P3': P3,
            'P4': P4,
            'P5': P5,
            'P6': P6,
            'P7': P7,
        }
