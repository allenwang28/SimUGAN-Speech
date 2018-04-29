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

from SimUGANSpeech.models import SimpleNN
from SimUGANSpeech.definitions import TENSORFLOW_DIR, DATA_DIR, TF_LOGS_DIR

class MnistSession(TensorflowSession):
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

        feature_sizes = [ 1200, 100 ]
        batch_size = 10
        verbose = True
        chunk_pct = 0.2
        num_epochs = 100

        # specify classifier parameters
        input_shape = (batch_size, 200, feature_sizes[0], 1)
        output_shape = (batch_size, feature_sizes[1], 26)


        # construct classifier
        self.discrim_clf = Discriminator(input_shape, output_shape, verbose=True)
        self.refiner_clf = Refiner(input_shape, output_shape, verbose=True)

        self.librispeech = LibriSpeechBatchGenerator(training_folder_names,
                                                     testing_folder_names,
                                                     feature_sizes=feature_sizes,
                                                     batch_size=batch_size,
                                                     chunk_pct=chunk_pct,
                                                     validation_pct=validation_pct,
                                                     verbose=verbose)        

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
                    
            num_batches = self.librispeech.num_batches

            for i in range(num_batches):
                # Load batch
                batch_mfcc, batch_transcriptions = self.librispeech.get_training_batch()
                batch_transcriptions = one_hot_transcriptions(batch_transcriptions)
                
                # Create map for batch to graph
                feed_dict = { self.refiner_clf.input_tensor : batch_mfcc,
                              self.refiner_clf.output_tensor : batch_transcriptions }

                # train REFINER and discri networks
                for k in xrange(2):
                    _, l, summary = self.sess.run([self.refiner_clf.optimize, self.summary_op], feed_dict=feed_dict)

                feed_dict = { self.discrim_clf.input_tensor : self.refiner_clf.results, 
                              self.discrim_clf.output_tensor : batch_transcriptions }

                for k in xrange(1):
                    _, l, summary = sess.run([self.discrim_clf.optimize, self.summary_op], feed_dict=feed_dict)

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
