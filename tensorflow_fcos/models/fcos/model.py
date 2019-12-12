import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal, Constant
from tensorflow.keras.layers import (Input,
                                     Reshape,
                                     ReLU,
                                     Add)
from models.blocks import conv_block, upsample_like
from models.custom_layers import Scale
from models.fcos.losses import focal_loss, centerness_loss, iou_loss
from models.metrics import Metrics
from data.encode import all_centers
from pprint import pprint


class FCOS:
    def __init__(self, config):
        self._validate_config(config)
        for attr in config:
            setattr(self, attr, config[attr])
        self._build_fpn()
        self._build_model()
        self._build_datasets()
        self._build_optimizer()
        self._initialize_metrics()
        self._create_checkpoint_manager()
        self._create_summary_writer()

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
            'checkpoint_prefix',
            'model_dir',
            'tensorboard_log_dir',
            'restore_parameters'
        ]
        for attr in attr_list:
            assert attr in config, 'Missing {} in config'.format(attr)
        pprint('****Initializing FCOS with the following config')
        pprint(config)

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
            pprint('****Building FPN')
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
                           bn_act=False, name_prefix='c_head_{}'.format(i))
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
        kernel_init = RandomNormal(0.0, 0.01)
        input_layer = Input(shape=[None, None, 256])
        x = input_layer

        for i in range(4):
            x = conv_block(x, 256, 3, kernel_init=kernel_init,
                           bn_act=False, name_prefix='r_head_{}'.format(i))
        regression_logits = conv_block(x, 4, 3, kernel_init=kernel_init,
                                       bn_act=False, name_prefix='reg_logits')
        regression_logits = Reshape(target_shape=[-1, 4])(regression_logits)
        return tf.keras.Model(inputs=[input_layer],
                              outputs=[regression_logits],
                              name='regression_head')

    def _build_model(self):
        with self.distribute_strategy.scope():
            pprint('****Building FCOS')
            self._classification_head = self._get_classification_head()
            self._regression_head = self._get_regression_head()

            _classification_logits = []
            _centerness_logits = []
            _regression_logits = []

            for i in range(3, 8):
                feature = self._pyramid_features['P{}'.format(i)]
                _cls_head_logits = self._classification_head(feature)
                _reg_head_logits = self._regression_head(feature)
                _reg_head_logits = \
                    Scale(init_value=1.0,
                          name='P{}_reg_outputs'.format(i))(_reg_head_logits)

                _classification_logits.append(_cls_head_logits[0][0])
                _centerness_logits.append(_cls_head_logits[0][1])
                _regression_logits.append(_reg_head_logits)

            _image_input = self._backbone.input
            outputs = [_classification_logits,
                       _centerness_logits,
                       _regression_logits
                       ]
            self.model = tf.keras.Model(
                inputs=[_image_input], outputs=outputs, name='FCOS')

    def _build_datasets(self):
        pprint('****Building Datasets')
        with self.distribute_strategy.scope():
            self.train_dataset, self.val_dataset, \
                num_train_images, num_val_images =  \
                self.dataset_fn(self.image_height,
                                self.image_width,
                                self.data_dir,
                                self.batch_size)
            self.train_dataset = \
                self.distribute_strategy.experimental_distribute_dataset(
                    self.train_dataset)
            self.val_dataset = \
                self.distribute_strategy.experimental_distribute_dataset(
                    self.val_dataset)

            self.training_steps = num_train_images // self.batch_size
            self.val_steps = num_val_images // self.batch_size
        self._centers = tf.concat(all_centers[3:], axis=0)

    def _build_optimizer(self):
        pprint('****Setting Up Optimizer')
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate,
                                                  clipnorm=0.0001)

    def _initialize_metrics(self):
        pprint('****Initializing Metrics')
        self.metrics = [
            Metrics(name='cls_loss'),
            Metrics(name='ctr_loss'),
            Metrics(name='iou_loss'),
            Metrics(name='total_loss')
        ]

    def restore_checkpoint(self, checkpoint_path):
        self.checkpoint.restore(checkpoint_path)

    def _create_checkpoint_manager(self):
        pprint('****Creating Checkpoint Manager')
        with self.distribute_strategy.scope():
            self.checkpoint = tf.train.Checkpoint(model=self.model)
            if self.restore_parameters:
                pprint('****Restoring Parameters')
                self.model.build([self.image_height, self.image_width, 3])
                self.latest_checkpoint = tf.train.latest_checkpoint(
                    self.model_dir)
                self.restore_status = self.checkpoint.restore(
                    self.latest_checkpoint)
                pprint('****Restored Parameters')

    def _create_summary_writer(self):
        self.summary_writer = tf.summary.create_file_writer(
            logdir=self.tensorboard_log_dir)

    def _write_summaries(self, cls_loss, ctr_loss, reg_loss):
        pprint('****Writing Summaries')
        with self.summary_writer.as_default():
            tf.summary.scalar('cls_loss',
                              cls_loss, step=self.iterations)
            tf.summary.scalar('ctr_loss',
                              ctr_loss, step=self.iterations)
            tf.summary.scalar('iou_loss',
                              reg_loss, step=self.iterations)

    def _write_checkpoint(self):
        self.checkpoint.save(os.path.join(self.model_dir,
                                          self.checkpoint_prefix))

    def _reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()

    def _update_metrics(self, cls_loss, ctr_loss, reg_loss):
        self.metrics[0].update_state(cls_loss.numpy())
        self.metrics[1].update_state(ctr_loss.numpy())
        self.metrics[2].update_state(reg_loss.numpy())

    def _log_metrics(self):
        current_iteration = self.iterations % (
            self.epoch * self.training_steps)
        metrics_dict = {
            'epoch': str(self.epoch) + '/' + str(self.epochs),
            'batch': str(current_iteration) + '/' + str(self.training_steps),
            'iterations': str(self.iterations)
        }
        for metric in self.metrics:
            metrics_dict.update({metric.name: np.round(metric.result(), 3)})
        print(metrics_dict)

    def _compute_loss(self, targets, logits):
        cls_logits = tf.concat(logits[0], axis=1)
        ctr_logits = tf.concat(logits[1], axis=1)
        reg_logits = tf.concat(logits[2], axis=1)

        cls_labels = targets[0]
        ctr_labels = targets[1]
        reg_labels = targets[2]
        fg_mask = targets[3]
        normalizer_value = targets[4]

        cls_loss = focal_loss(cls_labels,
                              cls_logits,
                              normalizer_value,
                              self.num_classes)
        ctr_loss = centerness_loss(
            ctr_labels, ctr_logits, fg_mask, normalizer_value)
        reg_loss = iou_loss(reg_labels, reg_logits,
                            self._centers, fg_mask, normalizer_value)

        cls_loss = tf.nn.compute_average_loss(
            cls_loss, global_batch_size=self.batch_size)
        ctr_loss = tf.nn.compute_average_loss(
            ctr_loss, global_batch_size=self.batch_size)
        reg_loss = tf.nn.compute_average_loss(
            reg_loss, global_batch_size=self.batch_size)
        return cls_loss, ctr_loss, reg_loss

    def train(self):
        assert self.mode == 'train', 'Cannot train in inference mode'
        pprint('****Starting Training Loop')

        with self.distribute_strategy.scope():
            @tf.function
            def train_step(images, targets):
                with tf.GradientTape() as tape:
                    logits = self.model(images, training=True)
                    cls_loss, ctr_loss, reg_loss = self._compute_loss(
                        targets, logits)
                    loss = cls_loss + ctr_loss + reg_loss
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients,
                                                   self.model.trainable_variables))
                loss_ = tf.stack([cls_loss, ctr_loss, reg_loss], axis=0)
                return loss_

            @tf.function
            def distributed_train_step(images, targets):
                per_replica_losses = self.distribute_strategy.experimental_run_v2(train_step,
                                                                                  args=(images, targets))
                loss = self.distribute_strategy.reduce(
                    tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

                return loss

            def _train():
                self.epoch = 1
                self.iterations = 1
                for _ in range(1, self.epochs + 1):
                    for images, targets in self.train_dataset:
                        cls_loss, ctr_loss, reg_loss = distributed_train_step(
                            images, targets)
                        total_loss = cls_loss + ctr_loss + reg_loss
                        self._update_metrics(cls_loss,
                                             ctr_loss,
                                             reg_loss,
                                             total_loss)
                        self._log_metrics()
                        self.iterations += 1
                        if self.iterations % 25 == 0:
                            self._write_summaries(cls_loss,
                                                  ctr_loss,
                                                  reg_loss,
                                                  total_loss)
                    self._reset_metrics()
                    self._write_checkpoint()
                    self.epoch += 1
            _train()
