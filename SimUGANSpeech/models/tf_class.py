# -*- coding: utf-8 *-* 
"""Abstract class for a tensorflow model
"""

import abc
from abc import ABC, abstractmethod

import tensorflow as tf
from SimUGANSpeech.models.tf_decorate import define_scope

import time
import os

class TensorflowModel(ABC):
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 input_shape,
                 output_shape,
                 verbose=True):
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._max_input_length = input_shape[1]
        self._max_output_length = output_shape[1]
        self._batch_size = tf.placeholder(tf.int32, shape=(input_shape[0]))
        self._training = True 

        self.input_tensor = tf.placeholder(tf.float32, shape=input_shape)
        self.output_tensor = tf.placeholder(tf.float32, shape=output_shape)
        self.predictions
        self.optimize
        self.loss
        self.error

    @define_scope
    def predictions(self):
        raise NotImplementedError('construct the graph here')

    @define_scope
    def optimize(self):
        raise NotImplementedError('specify the optimizer here')

    @define_scope
    def loss(self):
        raise NotImplementedError('specify loss function here')

    @define_scope
    def error(self):
        raise NotImplementedError('specify error')
    
    def initial_op(self):
        return tf.global_variables_initializer()

    def var_trainable_op(self):
        return tf.trainable_variables()

