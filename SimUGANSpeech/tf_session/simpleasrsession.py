# -*- coding: utf-8 *-* 
"""ASRSession class 

A class that used to train for simple speech recognition
using LibriSpeech

"""

from SimUGANSpeech.tf_session import TensorflowSession

import os
import tensorflow as tf

from SimUGANSpeech.data import LibriSpeechBatchGenerator
from SimUGANSpeech.models import SimpleNN
from SimUGANSpeech.definitions import TENSORFLOW_DIR, DATA_DIR, TF_LOGS_DIR
from SimUGANSpeech.util.data_util import text_to_indices

def one_hot_transcriptions(transcriptions):
    """One hot encode transcriptions"""
    t_idx = [text_to_indices(transcription) for transcription in transcriptions]
    return tf.one_hot(t_idx, 26, dtype=tf.uint8)


class SimpleASRSession(TensorflowSession):
    @property
    def name(self):
        """Name of the session"""
        return "ASR"


    @property
    def logs_path(self):
        """Path to save Tensorboard logs"""
        return os.path.join(TF_LOGS_DIR, 'simpleasr')


    @property
    def session_save_dir(self):
        """Path to save session checkpoints"""
        return os.path.join(TENSORFLOW_DIR, 'simpleasrsession')


    def initialize(self):
        """Initializations"""
        training_folder_names = [ 'dev-clean' ]
        testing_folder_names = []
        features = [ 'mfcc', 'transcription' ]
        feature_sizes = [ 1200, 100 ]
        batch_size = 10
        verbose = True
        chunk_pct = 0.2
        num_epochs = 100
        validation_pct = 0.3
        num_mfcc_features = 40

        input_shape = (batch_size, num_mfcc_features, feature_sizes[0], 1)
        output_shape = (batch_Size, feature_sizes[1], 26)

        self.librispeech = LibriSpeechBatchGenerator(training_folder_names,
                                                     testing_folder_names,
                                                     feature_sizes=feature_sizes,
                                                     batch_size=batch_size,
                                                     chunk_pct=chunk_pct,
                                                     validation_pct=validation_pct,
                                                     verbose=verbose)

        self.clf = SimpleASR(input_shape, output_shape, verbose=True)
        self.summary_op = tf.summary.merge_all()


    def train(self,
              num_epochs,
              backup_rate=100,
              display_rate=10,
              batch_size=100):
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
                feed_dict = { self.clf.input_tensor : batch_mfcc,
                              self.clf.output_tensor : batch_transcriptions }

                # Run optimizer, get cost, and summarize
                _, l, summary = self.sess.run([self.clf.optimize, self.clf.loss, self.summary_op], feed_dict=feed_dict)
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
