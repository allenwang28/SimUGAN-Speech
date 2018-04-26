# -*- coding: utf-8 *-* 

import tensorflow as tf
import os

from SimUGANSpeech.definitions import TENSORFLOW_DIR, TF_LOGS_DIR

class TensorflowSession(object):
    def __init__(self,
                 restore=True,
                 verbose=True):
        self.initialize()

        # Create all folders if needed
        for path in [self.session_save_dir, self.logs_path]:
            if not os.path.exists(path):
                os.mkdir(path)

        self.summary_writer = tf.summary.FileWriter(self.logs_path)

        self.sess = tf.Session()
        self.sess.run(self.initial_op())
        self.saver = tf.train.Saver()

        self._verbose = verbose

        # Restore session if possible
        if restore:
            if self._verbose:
                print ("Restoring session")
                latest = tf.train.latest_checkpoint(self.session_save_dir)
                if latest:
                    self.saver.restore(self.sess, latest)
                else:
                    if self._verbose:
                        print ("Checkpoint not found. Re-initializing.")
        else:
            if self._verbose:
                print ("Not restoring session. Re-initializing")


    @property
    def name(self):
        """Name of the session"""
        raise NotImplementedError("name the session here")


    @property
    def logs_path(self):
        """Path to save Tensorboard logs"""
        raise NotImplementedError("specify the location to save tensorboard logs")


    @property
    def session_save_dir(self):
        """Path to save session checkpoints"""
        raise NotImplementedError("specify the location to save session")


    def initialize(self):
        """Initialize the session"""
        raise NotImplementedError("implement the initializer here")


    def save_checkpoint(self, backup_number=None):
        """Save a checkpoint"""
        if not backup_number:
            path = os.path.join(self.session_save_dir, '{0}.cpkt'.format(self.name))
        else:
            path = os.path.join(self.session_save_dir, '{0}-backup{1}.cpkt'.format(self.name, backup_number))
        return self.saver.save(self.sess, path)


    def initial_op(self):
        """Initial ops for the session"""
        return tf.global_variables_initializer()


    def train(self, num_epochs):
        """Training loop for the session"""
        raise NotImplementedError("specify the training loop")
    

    def test(self):
        """Testing for the session"""
        raise NotImplementedError("specify the testing loop")


    def infer(self):
        """Inference for the session"""
        raise NotImplementedError("specify the training loop")
