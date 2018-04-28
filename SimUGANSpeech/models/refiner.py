# -*- coding: utf-8 *-* 
"""Implementation of SimGAN's Refiner Network

Todo:
    - Add support for more than just BasicRNNCell       
    - Add support for dropout parameters
    - Add support for hyperparameter options
    - Create a new class for parameters/parsing to take care of all of the above...
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim

from SimUGANSpeech.models.tf_decorate import define_scope
from SimUGANSpeech.models.tf_class import TensorflowModel

class Refiner(TensorflowModel):
    @property
    def name(self):
        """Name of the model"""
        return "Refiner"

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def predictions(self):
        # conv1
        with tf.variable_scope('refiner_conv1') as scope:
            l1_filter = tf.get_variable('refiner_l1_filter', shape=(64, 3, 1))
            conv1 = tf.nn.conv2D(self.input_tensor, l1_filter, padding='SAME')
            #conv1 = tf.layers.batch_normalization(conv1, training=self._training)
            #conv1 = tf.contrib.layers.dropout(conv1, keep_prob=0.8, is_training=self._training)
        # resnet
        with tf.variable_scope('refiner_resnet') as scope:
            num_outputs = 64
            l2_kernel = [3, 3]
            l2_stride = [1, 1]

            # resnet block 1
            layer = slim.conv2d(conv1, num_outputs, kernel_size, stride,
                padding=padding, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), 
                biases_initializer=tf.zeros_initializer(dtype=tf.float32), scope="conv1")
            layer = slim.conv2d(layer, num_outputs, kernel_size, stride,
                padding=padding, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(), 
                biases_initializer=tf.zeros_initializer(dtype=tf.float32), scope="conv2")
            resnet_output1 = tf.nn.relu(tf.add(conv1, layer))

            # resnet block 2
            layer = slim.conv2d(resnet_output1, num_outputs, kernel_size, stride,
                padding=padding, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), 
                biases_initializer=tf.zeros_initializer(dtype=tf.float32), scope="conv1")
            layer = slim.conv2d(layer, num_outputs, kernel_size, stride,
                padding=padding, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(), 
                biases_initializer=tf.zeros_initializer(dtype=tf.float32), scope="conv2")
            resnet_output2 = tf.nn.relu(tf.add(resnet_output1, layer))

            # resnet block 3
            layer = slim.conv2d(resnet_output2, num_outputs, kernel_size, stride,
                padding=padding, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), 
                biases_initializer=tf.zeros_initializer(dtype=tf.float32), scope="conv1")
            layer = slim.conv2d(layer, num_outputs, kernel_size, stride,
                padding=padding, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(), 
                biases_initializer=tf.zeros_initializer(dtype=tf.float32), scope="conv2")
            resnet_output3 = tf.nn.relu(tf.add(resnet_output2, layer))

            # resnet block 4
            layer = slim.conv2d(resnet_output3, num_outputs, kernel_size, stride,
                padding=padding, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), 
                biases_initializer=tf.zeros_initializer(dtype=tf.float32), scope="conv1")
            layer = slim.conv2d(layer, num_outputs, kernel_size, stride,
                padding=padding, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(), 
                biases_initializer=tf.zeros_initializer(dtype=tf.float32), cope="conv2")
            resnet_output = tf.nn.relu(tf.add(resnet_output3, layer))

        # conv2
        with tf.variable_scope('refiner_conv2') as scope:
            l3_filter = tf.get_variable('refiner_l3_filter', shape=(1, 1, 1))
            conv3 = tf.nn.conv2D(resnet_output, l3_filter, padding='SAME')
           #conv3 = tf.layers.batch_normalization(conv3, training=self._training)
           # conv3 = tf.contrib.layers.dropout(conv3, keep_prob=0.8, is_training=self._training)
        
        # activation function
        with tf.variable_scope('refiner_fc') as scope:
            output = tf.nn.tanh(conv3, self._num_output_features)
        return output, conv3

    @define_scope
    def optimize(self):
        logprob = tf.log(self.predictions + 1e-12)
        cross_entropy = -tf.reduce_sum(self.output_tensor * logprob)
        optimizer = tf.train.RMSPropOptimizer(0.03)
        return optimizer.minimize(cross_entropy)
   
    @define_scope
    def loss(self):
        # self._target_tensor: output of discriminator on the synthetic data; self._target_label: is the prediction label
        self._realism_loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(self.output_tensor, self.target_label), [1, 2], name='realism_loss')
        
        # self._output_tensor: output of refiner on the synthetic data; self.input_tensor: synthetic data
        self._regularization_loss = reg_scale * tf.reduce_sum(tf.abs(self.output_tensor - self.input_tensor), [1, 2, 3], name="regularization_loss")

        return tf.reduce_mean(realism_loss + self._regularization_loss)



    @define_scope
    def error(self):
        #mistakes = tf.not_equal(
        #    tf.argmax(self.output_tensor, 1), tf.argmax(self.predictions, 1))
        #return tf.reduce_mean(tf.cast(mistakes, tf.float32))
        return

