# -*- coding: utf-8 *-* 
"""SimGAN Session class 

A class that used to train for the SimGAN spectogram task 

Using the TensorflowSession as an abstract class should make
implementation more straightforward

Todo:
    * write the infer function

"""
from SimUGANSpeech.tf_session import TensorflowSession

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from SimUGANSpeech.data import LibriSpeechBatchGenerator
from SimUGANSpeech.data import SyntheticSpeechBatchGenerator

from SimUGANSpeech.models import Refiner
from SimUGANSpeech.models import Discriminator

from SimUGANSpeech.definitions import TENSORFLOW_DIR, DATA_DIR, TF_LOGS_DIR

class SimGANSession(TensorflowSession):
    @property 
    def name(self):
        """Name of the session"""
        return "SimGANSpeech"

    @property 
    def logs_path(self):
        """Path to save Tensorboard logs"""
        return os.path.join(TF_LOGS_DIR, 'simgan')


    @property
    def session_save_dir(self):
        """Path to save session checkpoints"""
        return os.path.join(TENSORFLOW_DIR, 'simgan_session')


    def initialize(self):
        """Initializations"""
        training_folder_names = [ 'dev-clean' ]
        testing_folder_names = []

        features = [ 'spectrogram' ] 
        feature_sizes = [ 1200]
        batch_size = 10
        verbose = True
        chunk_pct = 0.2
        num_epochs = 100
        validation_pct = 0.8

        # specify classifier parameters
        
        # discriminator takes spectrograms as input
        d_input_shape = (batch_size, feature_sizes[0], 200, 1)
         
        # discriminator's output is a 0 or a 1
        d_output_shape = (batch_size, 1)

        # refiner takes spectrograms as input
        r_input_shape = (batch_size, feature_sizes[0], 200, 1)

        # refiner outputs spectrograms
        r_output_shape = (batch_size, feature_sizes[0], 200, 1)

        # construct classifier
        self.refiner_clf = Refiner(r_input_shape, r_output_shape, verbose=True)
        self.discrim_clf = Discriminator(d_input_shape, d_output_shape, verbose=True)

        self.librispeech = LibriSpeechBatchGenerator(training_folder_names,
                                                     testing_folder_names,
                                                     features,
                                                     feature_sizes=feature_sizes,
                                                     batch_size=batch_size,
                                                     chunk_pct=chunk_pct,
                                                     validation_pct=validation_pct,
                                                     verbose=verbose)     

        self.syntheticspeech = SyntheticSpeechBatchGenerator(training_folder_names,
                                                             testing_folder_names,
                                                             features,
                                                             feature_sizes=feature_sizes,
                                                             batch_size=batch_size,
                                                             chunk_pct=chunk_pct,
                                                             validation_pct=validation_pct,
                                                             verbose=verbose)  

        self.summary_op = tf.summary.merge_all()


    def train(self,
              num_epochs,
              backup_rate=100,
              display_rate=10):
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
        num_synthetic_iterations = 10
        num_real_iterations = 10

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
                    #print ("Test error: {:6.2f}%".format(100 * self.test()))
                    # TODO 
                    continue
                    

            for j in range(num_synthetic_iterations):
                # Sample a mini-batch of synthetic images x_i
                synthetic_spectrograms = self.syntheticspeech.get_training_batch()[0]

                feed_dict = { self.refiner_clf.input_tensor : synthetic_spectrograms }
                refined_spectrograms = self.sess.run(self.refiner_clf.predictions, feed_dict=feed_dict)

                # Update theta by taking a SGD step on mini-batch loss L_r
                feed_dict = { self.discrim_clf.input_tensor : refined_spectrograms }
                discrim_logits = self.sess.run(self.discrim_clf.predictions, feed_dict=feed_dict)

                feed_dict = { self.refiner_clf.output_tensor : discrim_logits }

                _, l, summary = sess.run([self.refiner_clf.optimize, self.refiner_clf.loss, self.summary_op],
                                         feed_dict=feed_dict)

            for j in range(num_real_iterations):
                # Sample a mini-batch of synthetic images x_i, and real images y_j
                synthetic_spectrograms = self.syntheticspeech.get_training_batch()[0]
                real_spectrograms = self.librispeech.get_training_batch()[0]

                # Compute refined with current theta
                feed_dict = { self.refiner_clf.input_tensor : synthetic_spectrograms }
                refined_spectrograms = self.sess.run(self.refiner_clf.predictions, feed_dict=feed_dict)

                # Update phi by taking a SGD step on mini-batch loss L_d
                feed_dict = { self.discrim_clf.fake_input : refined_spectrograms,
                              self.discrim_clf.real_input : real_spectrograms }

                _, l, summary = sess.run([self.discrim_clf.optimize, self.discrim_clf.loss, self.summary_op],
                                         feed_dict=feed_dict)

            # Summarize
            self.summary_writer.add_summary(summary, epoch * num_batches + i)
        self.save_checkpoint()

    def test(self):
        """Return the error for the current model."""
        mfcc, transcriptions = self.librispeech.get_validation_data()
        transcriptions = one_hot_transcriptions(transcriptions)
        error = self.sess.run(self.clf.error, {self.clf.input_tensor : mfcc, self.clf.output_tensor: transcriptions})
        return error
             
    def infer(self):
        """Infer"""
        # TODO
        pass
