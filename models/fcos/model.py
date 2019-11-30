import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal, Constant
from tensorflow.keras.layers import (Input,
                                     Concatenate,
                                     Reshape,
                                     ReLU,
                                     Add)
from ..blocks import conv_block, upsample_like
from ..custom_layers import Scale


class FCOS:
    def __init__(self, config):
        self._validate_config(config)
        for attr in config:
            setattr(self, attr, config[attr])
        self._build_fpn()
        self._build_model()
#         self._build_datasets()

    def _validate_config(self, config):
        attr_list = [
            'mode',
            'distribute_strategy',
            'image_height',
            'image_width',
            'num_classes',
            'data_dir',
            'dataset_fn',
            'batch_size',
            'epochs',
            'learning_rate',
            'model_dir',
            'tensorboard_log_dir'
        ]
        for attr in attr_list:
            assert attr in config, 'Missing {} in config'.format(attr)

    def _build_fpn(self):
        '''
            From the FPN paper, "To start the iteration, we simply attach a
            1×1 convolutional layer on C5 to produce the coarsest resolution
            map. Finally, we append a 3×3 convolution on each merged map to
            generate the final feature map, which is to reduce the aliasing
            effect of upsampling. This final set of feature maps is called
            {P2, P3, P4, P5}, corresponding to {C2, C3, C4, C5} that are
            respectively of the same spatial sizes".
            From the FCOS paper, "P6 and P7 are produced by applying one
            convolutional layer with the stride being 2 on P5 and P6,
            respectively".
        '''
        with self.distribute_strategy.scope():
            print('****Building FPN')
            self._backbone = tf.keras.applications.ResNet50V2(
                input_shape=[self.image_height, self.image_width, 3],
                weights='imagenet',
                include_top=False)
            C5 = self._backbone.get_layer('post_relu').output
            C4 = self._backbone.get_layer('conv4_block6_1_relu').output
            C3 = self._backbone.get_layer('conv3_block4_1_relu').output

            M5 = conv_block(C5, 256, 1, bn_act=False, name_prefix='C5')
            P5 = conv_block(M5, 256, 3, bn_act=False, name_prefix='P5')
            M5_upsampled = upsample_like(M5, C4, name='M5_upsampled')

            M4 = conv_block(C4, 256, 1, bn_act=False, name_prefix='C4')
            M4 = tf.keras.layers.Add(name='M4_M5_add')([M4, M5_upsampled])
            P4 = conv_block(M4, 256, 3, bn_act=False, name_prefix='P4')
            M4_upsampled = upsample_like(M4, C3, name='M4_upsampled')

            M3 = conv_block(C3, 256, 1, bn_act=False, name_prefix='C3')
            P3 = Add(name='M3_M4_add')([M3, M4_upsampled])
            P3 = conv_block(P3, 256, 3, bn_act=False, name_prefix='P3')

            P6 = conv_block(P5, 256, 3, 2, bn_act=False, name_prefix='P6')
            P6_relu = ReLU(name='P6_relu')(P6)
            P7 = conv_block(P6_relu, 256, 3, 2, bn_act=False, name_prefix='P7')

            self._pyramid_features = {
                'P3': P3,
                'P4': P4,
                'P5': P5,
                'P6': P6,
                'P7': P7
            }

    def _get_classification_head(self, p=0.01):
        kernel_init = RandomNormal(0.0, 0.01)
        bias_init = Constant(-np.log((1 - p) / p))

        input_layer = Input(shape=[None, None, 256])
        x = input_layer

        for i in range(4):
            x = conv_block(x, 256, 3, kernel_init=kernel_init,
                           name_prefix='c_head_{}'.format(i))
        classification_logits = conv_block(x, self.num_classes,
                                           3, kernel_init=kernel_init,
                                           bias_init=bias_init, bn_act=False,
                                           name_prefix='cls_logits')
        centerness_logits = conv_block(x, 1, 3,
                                       kernel_init=kernel_init, bn_act=False,
                                       name_prefix='ctr_logits')
        classification_logits = Reshape(
            target_shape=[-1, self.num_classes])(classification_logits)
        centerness_logits = Reshape(target_shape=[-1, 1])(centerness_logits)

        outputs = [classification_logits, centerness_logits]
        return tf.keras.Model(inputs=[input_layer],
                              outputs=[outputs],
                              name='classification_head')

    def _get_regression_head(self):
        '''
            From the FCOS paper, "since the regression targets are always
            positive we employ exp(x) to map any real number to (0, ∞) on
            the top of the regression branch"
        '''
        kernel_init = RandomNormal(0.0, 0.01)
        input_layer = Input(shape=[None, None, 256])
        x = input_layer

        for i in range(4):
            x = conv_block(x, 256, 3, kernel_init=kernel_init,
                           name_prefix='r_head_{}'.format(i))
        regression_logits = conv_block(x, 4, 3, kernel_init=kernel_init,
                                       bn_act=False, name_prefix='reg_logits')
        regression_logits = Reshape(target_shape=[-1, 4])(regression_logits)
        return tf.keras.Model(inputs=[input_layer],
                              outputs=[regression_logits],
                              name='regression_head')

    def _get_predictions_decoder(self):
        # TODO
        pass

    def _build_model(self):
        with self.distribute_strategy.scope():
            print('****Building FCOS')
            self._classification_head = self._get_classification_head()
            self._regression_head = self._get_regression_head()

            self._classification_logits = []
            self._centerness_logits = []
            self._regression_logits = []

            for i in range(3, 8):
                feature = self._pyramid_features['P{}'.format(i)]
                _cls_head_logits = self._classification_head(feature)
                _reg_head_logits = self._regression_head(feature)
                _reg_head_logits = \
                    Scale(init_value=1.0,
                          name='P{}_reg_outputs'.format(i))(_reg_head_logits)

                self._classification_logits.append(_cls_head_logits[0][0])
                self._centerness_logits.append(_cls_head_logits[0][1])
                self._regression_logits.append(_reg_head_logits)

            self._classification_logits = Concatenate(
                axis=1,
                name='classification_outputs')(self._classification_logits)
            self._centerness_logits = Concatenate(
                axis=1, name='centerness_outputs')(self._centerness_logits)
            self._regression_logits = Concatenate(
                axis=1, name='regression_outputs')(self._regression_logits)

            _image_input = self._backbone.input
            outputs = [self._classification_logits,
                       self._centerness_logits,
                       self._regression_logits]
            self.model = tf.keras.Model(
                inputs=[_image_input], outputs=outputs, name='FCOS')
            self.model.build([self.image_height, self.image_width, 3])

    def _build_datasets(self):
        print('****Building Datasets')
        with self.distribute_strategy.scope():
            self.train_dataset, self.val_dataset =  \
                self.dataset_fn(self.image_height,
                                self.image_width,
                                self.data_dir,
                                self.batch_size)

    def __call__(self):
        # TODO
        pass

    def _classification_loss(self, alpha=0.25, gamma=2):
        # TODO
        #   a) mask negative locations
        #   b) normalize loss value
        def focal_loss(y_true, y_pred):
            y_true = tf.one_hot(
                tf.cast(y_true, dtype=tf.int32), depth=self.num_classes + 1)
            y_true = y_true[:, :, 1:]
            y_pred_ = tf.sigmoid(y_pred)

            at = alpha * y_true + (1 - y_true) * (1 - alpha)
            pt = y_true * y_pred_ + (1 - y_true) * (1 - y_pred_)
            f_loss = at * \
                tf.pow(1 - pt, gamma) * \
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=y_true, logits=y_pred)
            return f_loss
        return focal_loss

    def _centerness_loss(self, labels, logits):
        # TODO
        #   a) mask negative locations
        #   b) normalize loss value
        bce_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits)
        return bce_loss

    def _regression_loss(self, labels, logits):
        # TODO
        #   a) IOU loss
        #   b) mask negative locations
        #   c) normalize loss value
        pass
