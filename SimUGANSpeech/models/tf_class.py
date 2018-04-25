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

    @property
    def save_dir(self):
        return os.path.join(SAVE_DIR_BASE, self.name)

    def __init__(self,
                 input_shape,
                 output_shape,
                 save_dir):
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._max_input_length = input_shape[1]
        self._max_output_length = output_shape[1]
        self._batch_size = tf.placeholder(tf.int32, shape=(input_shape[0]))
        self._save_dir = save_dir
        self._training = True 

        self._var_trainable_op = tf.trainable_variables()
        self._initial_op = tf.contrib.layers.xavier_initializer()

        self.input_tensor = tf.placeholder(tf.float32, shape=input_shape)
        self.output_tensor = tf.placeholder(tf.float32, shape=output_shape)
        self.predictions
        self.optimize
        self.loss
        self.error

    @define_scope()
    def predictions(self):
        raise NotImplementedError('construct the graph here')

    @define_scope()
    def optimize(self):
        raise NotImplementedError('specify the optimizer here')

    @define_scope()
    def loss(self):
        raise NotImplementedError('specify loss function here')

    @define_scope()
    def error(self):
        raise NotImplementedError('specify error')
