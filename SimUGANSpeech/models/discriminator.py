# -*- coding: utf-8 *-* 
"""Implementation of SimGAN's Discriminator Network

Todo:
    - Everything 
"""
import tensorflow as tf

from SimUGANSpeech.models.tf_decorate import define_scope
from SimUGANSpeech.models.tf_class import TensorflowModel

class Discriminator(TensorflowModel):
    @property
    def name(self):
        """Name of the model"""
        return "Discriminator"

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def predictions(self):
        # conv1
        with tf.variable_scope('discrim_conv1') as scope:
            l1_filter = tf.get_variable('discrim_l1_filter', shape=(96, 3, 2))
            conv1 = tf.nn.conv2D(self.input_tensor, l1_filter, padding='SAME')
            #conv1 = tf.layers.batch_normalization(conv1, training=self._training)
            #conv1 = tf.contrib.layers.dropout(conv1, keep_prob=0.8, is_training=self._training)
        # conv2
        with tf.variable_scope('discrim_conv2') as scope:
            l2_filter = tf.get_variable('discrim_l2_filter', shape=(64, 3, 2))
            conv2 = tf.nn.conv2D(conv1, l2_filter, padding='SAME')
            #conv2 = tf.layers.batch_normalization(conv2, training=self._training)
            #conv2 = tf.contrib.layers.dropout(conv2, keep_prob=0.8, is_training=self._training)
        # max pooling
        with tf.variable_scope('discrim_max1') as scope:
            maxpool_filter = tf.get_variable('discrim_maxpool_filter', shape=(3, 1))
            max1 = tf.nn.max_pool2d(conv2, maxpool_filter)
        # conv3
        with tf.variable_scope('discrim_conv3') as scope:
            l3_filter = tf.get_variable('discrim_l3_filter', shape=(32, 3, 1))
            conv3 = tf.nn.conv2D(max1, l3_filter, padding='SAME')
            #conv3 = tf.layers.batch_normalization(conv3, training=self._training)
            #conv3 = tf.contrib.layers.dropout(conv3, keep_prob=0.8, is_training=self._training)
        # conv4
        with tf.variable_scope('discrim_conv4') as scope:
            l4_filter = tf.get_variable('discrim_l4_filter', shape=(32, 1, 1))
            conv4 = tf.nn.conv2D(conv3, l4_filter, padding='SAME')
            #conv4 = tf.layers.batch_normalization(conv4, training=self._training)
            #conv4 = tf.contrib.layers.dropout(conv4, keep_prob=0.8, is_training=self._training)
        # conv5
        with tf.variable_scope('discrim_conv5') as scope:
            l5_filter = tf.get_variable('discrim_l5_filter', shape=(2, 1, 1))
            conv5 = tf.nn.conv2D(conv4, l5_filter, padding='SAME')
            #conv5 = tf.layers.batch_normalization(conv5, training=self._training)
            #conv5 = tf.contrib.layers.dropout(conv5, keep_prob=0.8, is_training=self._training)
            self.lastLayer = conv5
        # softmax 
        with tf.variable_scope('discrim_softmax') as scope:
            self.results = tf.nn.softmax(conv5, self._num_output_features)
        return self

    @define_scope
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(0.001)
        return optimizer.minimize(self.loss)

    @define_scope
    def loss(self):
        # self._target_tensor: fake labels of discriminator output 
        self.refiner_d_loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(self.lastLayer, self.results), [1, 2], name='refiner_d_loss')
        
        # self.target_label: real label of real data
        self.synthetic_d_loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(self.real_data, self.real_label, "synthetic_d_loss")

      return tf.reduce_mean(self.refiner_d_loss + self.synthetic_d_loss, name="discrim_loss")

    @define_scope
    def error(self):
        #mistakes = tf.not_equal(
        #  tf.argmax(self.output_tensor, 1), tf.argmax(self.predictions, 1))
        #return tf.reduce_mean(tf.cast(mistakes, tf.float32))
        return 



