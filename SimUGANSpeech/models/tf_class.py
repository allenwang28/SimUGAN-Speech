# -*- coding: utf-8 *-* 
"""Abstract class for tensorflow models

Using this class as a base means you will only need to specify 
- the name
- construction of the graph
- optimizer
- loss 
- error

"""

import tensorflow as tf
from SimUGANSpeech.models.tf_decorate import define_scope

import time
import os

class TensorflowModel(object):

    def __init__(self,
                 input_shape,
                 output_shape,
                 sparse_output=False,
                 verbose=True):
        """Initialize the model

        Args:
            input_shape (array-like): Specifies the shape of the
                inputs to the tensorflow model.
            output_shape (array-like): Specifiess the shape of the
                outputs/labels of the tensorflow model
            verbose (:obj:`bool`, optional): Verbosity
                Defaults to True.

        """
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._max_input_length = input_shape[1]
        self._max_output_length = output_shape[1]
        self._batch_size = input_shape[0]
        self._training = True 

        self.input_tensor = tf.placeholder(tf.float32, shape=input_shape)
        if sparse_output:
            self.output_tensor = tf.sparse_placeholder(tf.int32)
        else:
            self.output_tensor = tf.placeholder(tf.float32, shape=output_shape)
        self.predictions
        self.optimize
        self.loss
        #self.error

        tf.summary.scalar("{0}-loss".format(self.name), self.loss)
        #tf.summary.scalar("{0}-error".format(self.name), self.error)


    @property
    def name(self):
        """Name of the model"""
        raise NotImplementedError('return the name of the model')


    @define_scope
    def predictions(self):
        """Construction of the graph"""
        raise NotImplementedError('construct the graph here')


    @define_scope
    def optimize(self):
        """Optimizer"""
        raise NotImplementedError('specify the optimizer here')


    @define_scope
    def loss(self):
        """Loss function"""
        raise NotImplementedError('specify loss function here')


    @define_scope
    def error(self):
        """Error function"""
        raise NotImplementedError('specify error')


    def _conv1d(self, input_tensor, filter_width, stride, in_channels, out_channels, relu):
        """Add a 1D Convolutional layer

        Args:
            input_tensor (tf object): The input to the convolutional layer
            filter_width (int): width of the filter 
            stride (int): Stride of the filter
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            relu (bool): whether or not to apply ReLU

        Returns:
            tf object: the output of the convolutional layer 

        """
        num_convolutional_layers = self.num_convolutional_layers
        self.num_convolutional_layers += 1

        layer_name = "{0}-conv{1}".format(self.name, num_convolutional_layers)
        with tf.variable_scope(layer_name) as layer:
            filters = tf.get_variable('filters', shape=[filter_width, in_channels, out_channels],
                                      dtype=tf.float32, initializer=tf.contrib.slim.xavier_initializer())
            bias = tf.Variable(tf.constant(0.0, shape=[out_channels]), name='bias')

            convolution_out = tf.nn.conv1d(input_tensor, filters, stride, 'SAME', use_cudnn_on_gpu=True, name='convolution')

            with tf.name_scope('summaries'):
                # add a depth of 1 (for grayscale) to result in a shape of [filter_width, in_channels, 1, out_channels]
                kernel_width_depth = tf.expand_dims(filters, 2)

                kernel_transposed = tf.transpose(kernel_width_depth, [3, 0, 1, 2])

                # Display random 3 filters from all output channels
                tf.summary.image("{0} filters".format(layer_name), kernel_transposed, max_outputs=3)
                tf.summary.histogram("{0} filters".format(layer_name), filters)
                tf.summary.image("{0} bias".format(layer_name), tf.reshape(bias, [1, 1, out_channels, 1]))
                tf.summary.histogram("{0} bias".format(layer_name), bias)

            convolution_out = tf.nn.bias_add(convolution_out, bias)

            if relu:
                activations = tf.nn.relu(convolution_out, name='activation')
                tf.summary.histogram("{0} activation".format(layer_name), activations)
                return activations, out_channels
            else:
                return convolution_out, out_channels

