# -*- coding: utf-8 *-* 
"""wav2lettersession 

A class used to train for speech recognition using LibriSpeech
and Wav2Letter

"""

from SimUGANSpeech.tf_session import TensorflowSession

import os
import tensorflow as tf
import numpy as np

from SimUGANSpeech.data import LibriSpeechBatchGenerator
from SimUGANSpeech.models import Wav2Letter
from SimUGANSpeech.definitions import TENSORFLOW_DIR, DATA_DIR, TF_LOGS_DIR
from SimUGANSpeech.util.data_util import tf_transcriptions, get_sequence_lengths

vocabulary_size = 29

class Wav2LetterSession(TensorflowSession):
    @property
    def name(self):
        """Name of the session"""
        return "ASR"


    @property
    def logs_path(self):
        """Path to save Tensorboard logs"""
        return os.path.join(TF_LOGS_DIR, 'Wav2LetterSession')


    @property
    def session_save_dir(self):
        """Path to save session checkpoints"""
        return os.path.join(TENSORFLOW_DIR, 'Wav2LetterSession')


    def initialize(self):
        """Initializations"""
        training_folder_names = [ 'dev-clean' ]
        testing_folder_names = []
        features = [ 'mfcc', 'transcription' ]
        feature_sizes = [ 1200, 10 ]
        batch_size = 10
        verbose = True
        chunk_pct = None
        validation_pct = 0.3
        num_mfcc_features = 40

        self.batch_size = batch_size
        self.max_transcription_time = feature_sizes[1]

        input_shape = (batch_size, feature_sizes[0], num_mfcc_features)
        output_shape = (batch_size, feature_sizes[1], vocabulary_size)

        self.librispeech = LibriSpeechBatchGenerator(training_folder_names,
                                                     testing_folder_names,
                                                     features,
                                                     feature_sizes=feature_sizes,
                                                     batch_size=batch_size,
                                                     chunk_pct=chunk_pct,
                                                     validation_pct=validation_pct,
                                                     verbose=verbose)

        self.clf = Wav2Letter(input_shape, output_shape, verbose=True)
        self.summary_op = tf.summary.merge_all()


    def train(self,
              num_epochs,
              backup_rate=100,
              display_rate=10):
        """Run the training loop for ASR 

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
                backup_number = int(epoch / backup_rate)
                if self._verbose:
                    print ("Saving checkpoint {0}".format(backup_number))
                    self.save_checkpoint(backup_number)

            # Display loss information
            if self._verbose: 
                if epoch % display_rate == 0:
                    # print ("Test error: {:6.2f}%".format(100 * self.test()))
                    print ("epoch {0}".format(epoch))
                    
            num_batches = self.librispeech.num_batches

            for i in range(num_batches):
                # Load batch
                batch = self.librispeech.get_training_batch()
                batch_mfcc, batch_transcriptions = batch[0], batch[1]


                sequence_lengths = get_sequence_lengths(batch_transcriptions)
                batch_transcriptions = tf_transcriptions(batch_transcriptions, self.max_transcription_time)
                
                # Create map for batch to graph
                feed_dict = { self.clf.input_tensor : batch_mfcc,
                              self.clf.output_tensor : batch_transcriptions,
                              self.clf.sequence_lengths_tensor : sequence_lengths
                            }

                # Run optimizer, get cost, and summarize
                _, l, summary = self.sess.run([self.clf.optimize, self.clf.loss, self.summary_op], feed_dict=feed_dict)
                print (l)
                self.summary_writer.add_summary(summary, epoch * num_batches + i)

        self.save_checkpoint()

    def test(self):
        """Return the error for the current model."""
        batch = self.librispeech.get_validation_data()
        mfcc, transcriptions = batch[0], batch[1]
        transcriptions = tf_transcriptions(transcriptions, self.max_transcription_time)
        error = self.sess.run(self.clf.error, {self.clf.input_tensor : mfcc, self.clf.output_tensor: transcriptions})
        return error


    def infer(self):
        """Infer"""
        # TODO
        pass
