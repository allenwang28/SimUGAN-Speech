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
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._max_input_length = input_shape[1]
        self._max_output_length = output_shape[1]
        self._batch_size = input_shape[0]
        self._training = True 

        self.fake_input = tf.placeholder(tf.float32, shape=input_shape)
        self.real_input = tf.placeholder(tf.float32, shape=input_shape)
        self.fake_logits = self.construct(self.fake_input, 'discrim', reuse=False)
        self.real_logits = self.construct(self.real_input, 'discrim', reuse=True)

        self.optimize
        self.loss
        #self.error

        tf.summary.scalar("{0}-loss".format(self.name), self.loss)
        #tf.summary.scalar("{0}-error".format(self.name), self.error)



    @property
    def name(self):
        """Name of the model"""
        return "Discriminator"

    #@define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def construct(self, layer, name, reuse=True):
        with tf.variable_scope(name, reuse=reuse) as sc:
              layer = slim.conv2d(layer, 96, 3, 2, scope="conv_1")
              layer = slim.conv2d(layer, 64, 3, 2, scope="conv_2")
              layer = slim.max_pool2d(layer, 3, 1, scope="max_1")
              layer = slim.conv2d(layer, 32, 3, 1, scope="conv_3")
              layer = slim.conv2d(layer, 32, 1, 1, scope="conv_4")
              logits = slim.conv2d(layer, 2, 1, 1, scope="conv_5")
              output = tf.nn.softmax(logits, name="softmax")
              self.discrim_vars = tf.contrib.framework.get_variables(sc)
        return logits

        """
        # conv1
        with tf.variable_scope('{0}_conv1'.format(name), reuse=reuse) as scope:
            conv1 = slim.conv2d(input_tensor, 96, 3, 2, scope="{0}_l1_filter".format(name), padding='SAME')

        # conv2
        with tf.variable_scope('{0}_conv2'.format(name), reuse=reuse) as scope:
            conv2 = slim.conv2d(conv1, 64, 3, 2, scope="{0}_l2_filter".format(name), padding='SAME')

        # max pooling
        with tf.variable_scope('{0}_max1'.format(name), reuse=reuse) as scope:
            max1 = slim.max_pool2d(conv2, 3, 1, scope='{0}_max_1'.format(name), padding='SAME')
        # conv3
        with tf.variable_scope('{0}_conv3'.format(name), reuse=reuse) as scope:
            conv3 = slim.conv2d(max1, 32, 3, 1, scope="{0}_l3_filter".format(name), padding='SAME')

        # conv4
        with tf.variable_scope('{0}_conv4'.format(name), reuse=reuse) as scope:
            conv4 = slim.conv2d(conv3, 32, 1, 1, scope="{0}_l4_filter".format(name), padding='SAME')

        # conv5
        with tf.variable_scope('{0}_conv5'.format(name), reuse=reuse) as scope:
            conv5 = slim.conv2d(conv4, 2, 1, 1, scope="{0}_l5_filter".format(name), padding='SAME')

        # softmax 
        with tf.variable_scope('{0}_softmax'.format(name)) as scope:
            results = tf.nn.softmax(conv5)
        return results 
        """

    @define_scope
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(0.01, name='discrim_optimizer')
        return optimizer.minimize(self.loss)

    @define_scope
    def loss(self):
        # generate labels for data
        real_label = tf.ones_like(self.real_logits, dtype=tf.int32)[:,:,:,0]
        fake_label = tf.zeros_like(self.fake_logits, dtype=tf.int32)[:,:,:,0]

        #print(self.real_logits.get_shape())
        #print(real_label.get_shape())
        #exit()

        refiner_d_loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.fake_logits,labels=fake_label), [1, 2])
        
        # self.target_label: real label of real data
        synthetic_d_loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.real_logits,labels=real_label), [1,2])

        return tf.reduce_mean(refiner_d_loss + synthetic_d_loss, name="discrim_loss")

    @define_scope
    def error(self):
        #mistakes = tf.not_equal(
        #  tf.argmax(self.output_tensor, 1), tf.argmax(self.predictions, 1))
        #return tf.reduce_mean(tf.cast(mistakes, tf.float32))
        return 



