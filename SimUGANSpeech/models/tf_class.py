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


    def var_trainable_op(self):
        return tf.trainable_variables()

