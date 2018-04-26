# -*- coding: utf-8 *-* 
"""MnistSession class 

A class that used to train for the Mnist task 

Using the TensorflowSession as an abstract class should make
implementation more straightforward

Todo:
    * write the infer function

"""
from SimUGANSpeech.tf_session.tf_session import TensorflowSession

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from SimUGANSpeech.models.simple import SimpleNN
from SimUGANSpeech.definitions import TENSORFLOW_DIR, DATA_DIR, TF_LOGS_DIR

class MnistSession(TensorflowSession):
    @property 
    def name(self):
        """Name of the session"""
        return "MNIST"

    @property 
    def logs_path(self):
        """Path to save Tensorboard logs"""
        return os.path.join(TF_LOGS_DIR, 'simple')


    @property
    def session_save_dir(self):
        """Path to save session checkpoints"""
        return os.path.join(TENSORFLOW_DIR, 'mnist_session')


    def initialize(self):
        """Mnist specific initializations"""
        mnist_save_path = os.path.join(DATA_DIR, 'mnist_ex')
        self.mnist = input_data.read_data_sets(mnist_save_path, one_hot = True)

        input_shape = (None, 784)
        output_shape = (None, 10)
        self.clf = SimpleNN(input_shape, output_shape, verbose=True)
        self.summary_op = tf.summary.merge_all()


    def train(self,
              num_epochs,
              backup_rate=100,
              display_rate=10,
              batch_size=100):
        """Run the training loop for MNIST

        Args:
            num_epochs (int): The number of epochs to run
            backup_rate (:obj:`int`, optional): How many 
                epochs should be run before backing up
            display_rate (:obj:`int`, optional): How many
                epochs should be run before displaying information
            batch_size (:obj:`int`, optional): Size of a 
                minibatch

        """
        for epoch in range(num_epochs):
            # Create a checkpoint
            if epoch % backup_rate == 0:
                backup_number = epoch / backup_rate
                if self._verbose:
                    print ("Saving checkpoint {0}".format(backup_number))
                    self.save_checkpoint(backup_number)

            # Display loss information
            if self._verbose: 
                if epoch % display_rate == 0:
                    print ("Test error: {:6.2f}%".format(100 * self.test()))
                    
            num_batches = int(self.mnist.train.num_examples/batch_size)

            for i in range(num_batches):
                # Load batch
                batch_xs, batch_ys = self.mnist.train.next_batch(batch_size)
                
                # Create map for batch to graph
                feed_dict = { self.clf.input_tensor : batch_xs,
                              self.clf.output_tensor : batch_ys }

                # Run optimizer, get cost, and summarize
                _, l, summary = self.sess.run([self.clf.optimize, self.clf.loss, self.summary_op], feed_dict=feed_dict)
                self.summary_writer.add_summary(summary, epoch * num_batches + i)

        self.save_checkpoint()

    def test(self):
        """Return the error for the current model."""
        images, labels = self.mnist.test.images, self.mnist.test.labels
        error = self.sess.run(self.clf.error, {self.clf.input_tensor : images, self.clf.output_tensor: labels})
        return error
             
    def infer(self):
        """Infer"""
        # TODO
        pass
