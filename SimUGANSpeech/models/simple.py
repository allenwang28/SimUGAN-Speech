# -*- coding: utf-8 *-* 
"""Simple neural network 

Mainly written to show how to create a model using
the abstract TensorflowModel

"""

import tensorflow as tf

from SimUGANSpeech.models.tf_decorate import define_scope
from SimUGANSpeech.models.tf_class import TensorflowModel

class SimpleNN(TensorflowModel):
    @property
    def name(self):
        """Name of the model"""
        return "SimpleNN"

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def predictions(self):
        """Construction of the graph"""
        with tf.variable_scope('simple') as scope:
            x = self.input_tensor
            x = tf.contrib.slim.fully_connected(x, 200)
            x = tf.contrib.slim.fully_connected(x, 200)
            x = tf.contrib.slim.fully_connected(x, 10, tf.nn.softmax)
        return x

    @define_scope
    def optimize(self):
        """Optimizer"""
        optimizer = tf.train.RMSPropOptimizer(0.03)
        return optimizer.minimize(self.loss)

    @define_scope
    def loss(self):
        """Loss function"""
        logprob = tf.log(self.predictions + 1e-12)
        cross_entropy = -tf.reduce_sum(self.output_tensor * logprob)
        return cross_entropy

    @define_scope
    def error(self):
        """Error function"""
        mistakes = tf.not_equal(
            tf.argmax(self.output_tensor, 1), tf.argmax(self.predictions, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))
