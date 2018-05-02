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
    def __init__(self,
                 input_shape,
                 output_shape,
                 fake_logits,
                 reg_scale=0.1,
                 learning_rate=0.001,
                 verbose=True):
        batch_size = input_shape[0]
        self.fake_logits = fake_logits
        self.input_tensor = tf.placeholder(tf.float32, shape=input_shape)
        self.output_tensor = tf.placeholder(tf.float32, shape=output_shape)

        self.reg_scale = reg_scale
        self.learning_rate = learning_rate

        super().__init__(input_shape, output_shape, verbose=verbose)

    @property
    def name(self):
        """Name of the model"""
        return "Refiner"

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def predictions(self):
        name=self.name

        with tf.variable_scope(name) as sc:
            kernel_size = [3, 3]
            stride = [1, 1]
            padding = 'SAME'

            layer = tf.layers.conv1d(inputs=self.input_tensor, filters=275, kernel_size=1, strides=1, padding=padding)

            #resnet block 1
            inputLayer = layer
            layer = tf.layers.conv1d(inputs=inputLayer, filters=275, kernel_size=1, strides=1, activation=tf.nn.relu, padding=padding)
            layer = tf.layers.conv1d(inputs=layer, filters=275, kernel_size=1, strides=1, activation=None, padding=padding)
            layer = tf.nn.relu(tf.add(inputLayer, layer))

            #resnet block 2
            inputLayer = layer
            layer = tf.layers.conv1d(inputs=inputLayer, filters=275, kernel_size=1, strides=1, activation=tf.nn.relu, padding=padding)
            layer = tf.layers.conv1d(inputs=layer, filters=275, kernel_size=1, strides=1, activation=None, padding=padding)
            layer = tf.nn.relu(tf.add(inputLayer, layer))

            #resnet block 3
            inputLayer = layer
            layer = tf.layers.conv1d(inputs=inputLayer, filters=275, kernel_size=1, strides=1, activation=tf.nn.relu, padding=padding)
            layer = tf.layers.conv1d(inputs=layer, filters=275, kernel_size=1, strides=1, activation=None, padding=padding)
            layer = tf.nn.relu(tf.add(inputLayer, layer))

            #resnet block 4
            inputLayer = layer
            layer = tf.layers.conv1d(inputs=inputLayer, filters=275, kernel_size=1, strides=1, activation=tf.nn.relu, padding=padding)
            layer = tf.layers.conv1d(inputs=layer, filters=275, kernel_size=1, strides=1, activation=None, padding=padding)
            layer = tf.nn.relu(tf.add(inputLayer, layer))

            layer = tf.layers.conv1d(inputs=layer, filters=275, kernel_size=1, strides=1, activation=None, padding=padding)

            output = tf.nn.tanh(layer, name="tanh")
            self.refiner_vars = tf.contrib.framework.get_variables(sc)
        return output 

        """
        with tf.variable_scope(name) as sc:
            num_outputs = 64
            kernel_size = [3, 3]
            stride = [1, 1]
            padding = 'SAME'

            layer = slim.conv2d(self.input_tensor, 64, 3, 1, scope="refiner_conv_1")

            #resnet block 1
            inputLayer = layer
            layer = slim.conv2d(layer, num_outputs, kernel_size, stride, padding=padding, activation_fn=tf.nn.relu, scope="conv2a")
            layer = slim.conv2d(layer, num_outputs, kernel_size, stride, padding=padding, activation_fn=None, scope="conv2b")
            layer = tf.nn.relu(tf.add(inputLayer, layer))

            #resnet block 2
            inputLayer = layer
            layer = slim.conv2d(layer, num_outputs, kernel_size, stride, padding=padding, activation_fn=tf.nn.relu, scope="conv3a")
            layer = slim.conv2d(layer, num_outputs, kernel_size, stride, padding=padding, activation_fn=None, scope="conv3b")
            layer = tf.nn.relu(tf.add(inputLayer, layer))

            #resnet block 3
            inputLayer = layer
            layer = slim.conv2d(layer, num_outputs, kernel_size, stride, padding=padding, activation_fn=tf.nn.relu, scope="conv4a")
            layer = slim.conv2d(layer, num_outputs, kernel_size, stride, padding=padding, activation_fn=None, scope="conv4b")
            layer = tf.nn.relu(tf.add(inputLayer, layer))

            #resnet block 4
            inputLayer = layer
            layer = slim.conv2d(layer, num_outputs, kernel_size, stride, padding=padding, activation_fn=tf.nn.relu, scope="conv5a")
            layer = slim.conv2d(layer, num_outputs, kernel_size, stride, padding=padding, activation_fn=None, scope="conv5b")
            layer = tf.nn.relu(tf.add(inputLayer, layer))

            layer = slim.conv2d(layer, 1, 1, 1, activation_fn=None, scope="conv_6")
            output = tf.nn.tanh(layer, name="tanh")
            self.refiner_vars = tf.contrib.framework.get_variables(sc)
        return output 
        """


    @define_scope
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate, name='refiner_optimizer')
        return optimizer.minimize(self.loss)

   
    @define_scope
    def loss(self):
        # self._target_tensor: output of discriminator on the synthetic data; self._target_label: is the prediction label
        target_label = tf.ones_like(self.fake_logits, dtype=tf.int32)[:,:,0]


        #print(self.fake_logits.get_shape())
        #print(target_label.get_shape())
       #exit()


        realism_loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_label, logits=self.fake_logits), name='realism_loss')
        # self._output_tensor: output of refiner on the synthetic data; self.input_tensor: synthetic data
        regularization_loss = self.reg_scale * tf.reduce_sum(tf.abs(self.output_tensor- self.input_tensor), name="regularization_loss")
        return tf.reduce_mean(realism_loss + regularization_loss)


    @define_scope
    def error(self):
        #mistakes = tf.not_equal(
        #    tf.argmax(self.output_tensor, 1), tf.argmax(self.predictions, 1))
        #return tf.reduce_mean(tf.cast(mistakes, tf.float32))
        return

