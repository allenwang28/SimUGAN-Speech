# -*- coding: utf-8 *-* 
"""Wav2Letter for speech recognition

The network is adapted from 
https://github.com/timediv/speechT/blob/master/speecht/speech_model.py
"""
import tensorflow as tf

from SimUGANSpeech.models.tf_decorate import define_scope
from SimUGANSpeech.models.tf_class import TensorflowModel

class Wav2Letter(TensorflowModel):

    def __init__(self,
                 input_shape,
                 output_shape,
                 learning_rate=1e-3,
                 learning_rate_decay_factor=0,
                 max_gradient_norm=5.0,
                 verbose=True):
        self._learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32, name='learning_rate')
        self._learning_rate_decay_op = self._learning_rate.assign(self._learning_rate * learning_rate_decay_factor)

        self._input_size = input_shape[2]
        self._num_classes = output_shape[2]
        tf.summary.scalar('learning_rate', self._learning_rate)
        self._max_gradient_norm = max_gradient_norm
        self.global_step = tf.Variable(0, trainable=False)
        super().__init__(input_shape, output_shape, verbose=verbose, sparse_output=True)


    @property
    def name(self):
        """Name of the model"""
        return "Wav2Letter"

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def predictions(self):
        """Construction of the graph"""
        with tf.variable_scope('Wav2Letter') as scope:
            # First layer scales up from input_size channels
            conv, channels = self._conv1d(self.input_tensor, 48, 2, self._input_size, 250, True)

            # 7 layers without striding of output size [batch_size, max_time / 2, 250]
            for _ in range(7):
                conv, channels = self._conv1d(conv, 7, 1, channels, channels, True)
            
            # 1 layer with high kernel width and output size [batch_size, max_time / 2, 2000]
            conv, channels = self._conv1d(conv, 32, 1, channels, channels * 8, True)

            # 1 fully connected layer of output size [batch_size, max_time / 2, 2000]
            conv, channels = self._conv1d(conv, 1, 1, channels, channels, True)

            # 1 fully connected layer of output size [batch_size, max_time / 2, num_classes]
            conv, channels = self._conv1d(conv, 1, 1, channels, self._num_classes, False)

            # transpose logits to size [max_time / 2, batch_size, num_classes]
            return tf.transpose(conv, (1,0,2))


    @define_scope
    def optimize(self):
        """Optimizer"""
        optimizer = tf.train.AdamOptimizer(self._learning_rate, epsilon=1e-3)
        gvs = optimizer.compute_gradients(self.loss)
        gradients, trainables = zip(*gvs)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, self._max_gradient_norm, name='clip_gradients')
        return optimizer.apply_gradients(zip(clipped_gradients, trainables),
                                         global_step=self.global_step, name='apply_gradients')

    @define_scope
    def loss(self):
        """Loss function"""
        cost = tf.nn.ctc_loss(self.output_tensor, self.predictions, self._batch_size // 2)
        return tf.reduce_mean(cost, name='loss')


    @define_scope
    def error(self):
        """Error function"""
        # TODO - make this different from the loss
        cost = tf.nn.ctc_loss(self.output_tensor, self.predictions, self._batch_size // 2)
        return tf.reduce_mean(cost, name='loss')


