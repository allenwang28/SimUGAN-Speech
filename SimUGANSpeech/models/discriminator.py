# -*- coding: utf-8 *-* 
"""Implementation of SimGAN's Discriminator Network

Todo:
    - Everything 
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim

from SimUGANSpeech.models.tf_decorate import define_scope
from SimUGANSpeech.models.tf_class import TensorflowModel

class Discriminator(TensorflowModel):
    def __init__(self,
                 input_shape,
                 output_shape,
                 verbose=True):
        self.fake_logits = tf.placeholder(tf.int32, shape=output_shape)
        self.real_logits = tf.placeholder(tf.int32, shape=output_shape)
        super().__init__(input_shape, output_shape, verbose=verbose)

    @property
    def name(self):
        """Name of the model"""
        return "Discriminator"

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def predictions(self):
        # conv1
        with tf.variable_scope('discrim_conv1') as scope:
            conv1 = slim.conv2d(self.input_tensor, 96, 3, 2, scope="discrim_l1_filter", padding='SAME')

        # conv2
        with tf.variable_scope('discrim_conv2') as scope:
            conv2 = slim.conv2d(conv1, 64, 3, 2, scope="discrim_l2_filter", padding='SAME')

        # max pooling
        with tf.variable_scope('discrim_max1') as scope:
            max1 = slim.max_pool2d(conv2, 3, 1, scope='max_1', padding='SAME')
        # conv3
        with tf.variable_scope('discrim_conv3') as scope:
            conv3 = slim.conv2d(max1, 32, 3, 1, scope="discrim_l3_filter", padding='SAME')

        # conv4
        with tf.variable_scope('discrim_conv4') as scope:
            conv4 = slim.conv2d(conv3, 32, 1, 1, scope="discrim_l4_filter", padding='SAME')

        # conv5
        with tf.variable_scope('discrim_conv5') as scope:
            conv5 = slim.conv2d(conv4, 2, 1, 1, scope="discrim_l5_filter", padding='SAME')

        # softmax 
        with tf.variable_scope('discrim_softmax') as scope:
            results = tf.nn.softmax(conv5)
        return results, conv5

    @define_scope
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(0.001)
        return optimizer.minimize(self.loss)

    @define_scope
    def loss(self):
        # self._target_tensor: fake labels of discriminator output 
        real_label = tf.ones_like(self.real_logits)[:,0]
        fake_label = tf.zeros_like(self.fake_logits)[:,0]

        #print(self.real_logits.get_shape())
        #print(real_label.get_shape())
        #exit()

        refiner_d_loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.fake_logits, labels=fake_label), [1, 2], name='refiner_d_loss')
        
        # self.target_label: real label of real data
        synthetic_d_loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.real_logits,labels=real_label), [1, 2], name="synthetic_d_loss")

        return tf.reduce_mean(refiner_d_loss + synthetic_d_loss, name="discrim_loss")

    @define_scope
    def error(self):
        #mistakes = tf.not_equal(
        #  tf.argmax(self.output_tensor, 1), tf.argmax(self.predictions, 1))
        #return tf.reduce_mean(tf.cast(mistakes, tf.float32))
        return 



