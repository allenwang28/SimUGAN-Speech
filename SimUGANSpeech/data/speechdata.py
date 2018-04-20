# -*- coding: utf-8 *-* 
"""Speech Data Generator

This module is used to provide a batch generator for 
training our models.

Since we preprocess all of our data the same way, the
SpeechBatchGenerator only has to have access to the folders

Todo:
    * Pad/truncate the data in the batch generator

"""

import os
import numpy as np
import sys

import pickle
import copy

from SimUGANSpeech.util.data_util import randomly_sample_stack

DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_TIME_STEPS = 50
DEFAULT_MAX_OUTPUT_LENGTH = 50

DEFAULT_CHUNK_PROCESS_PERCENTAGE = 0.3

ACCEPTED_LABELS =   ['transcription_chars',
                     'voice_id']

FEATURES = [ 
             'spectrogram',
             'transcription',
             'id',
           ]


class SpeechBatchGenerator(object):
    def __init__(self,
                 folder_dir,
                 folder_names,
                 features,
                 feature_sizes,
                 batch_size=DEFAULT_BATCH_SIZE,
                 chunk_pct=DEFAULT_CHUNK_PROCESS_PERCENTAGE,
                 verbose=True):
        """SpeechBatchGenerator class initializer
        
        Args:
            folder_dir (str): The path to the data folder
            folder_paths (list of str): List of the folder names (or datasets)
                (e.g., dev-clean, dev-test, etc.)
            features (list of str): List of desired features.
                See constant defined FEATURES for list of valid features.
            feature_sizes (list of int): List of maximum length of features.
                Has to be the same shape as features. The features will be
                truncated or padded to match the specified shape.
                If no maximum/truncation desired, just provide None
            batch_size (:obj:`int`, optional): The desired batch size.
                Defaults to 10
            chunk_pct (:obj:`float`, optional): The percentage of chunks to
                load into memory at a time. 
                The lower the value, the less likely to reach a memory error.
                The higher the value, the more efficient the batch generator
                Defaults to 0.3
            verbose (:obj:`bool`, optional): Whether or not to print statements.
                Defaults to True.
        
        """
        features = [f.lower() for f in features]
        for feature in features:
            if feature not in FEATURES: 
                raise ValueError('Invalid feature')
        self._features = features
        self._feature_sizes = feature_sizes
        self._batch_size = batch_size
        self._chunk_pct = chunk_pct

        if len(feature_sizes) != len(features):
            raise ValueError('Length of feature_sizes should match length of features')
        self._verbose = verbose

        spectro_paths = []
        transcription_paths = []
        id_paths = []
        self._num_chunks = 0
        self._total_samples = 0
        self._max_spectro_feature_length = 0

        # Load the master files
        for fname in folder_names:
            fpath = os.path.join(folder_dir, fname)
            master_path = os.path.join(fpath, 'master.pkl')
            try:
                master = pickle.load(open(master_path, 'rb'))
            except:
                raise RuntimeError("""
                    There was a problem with loading the master file, {0}.\n
                    Make sure the data is preprocessed. Check in /scripts
                """.format(master_path)) 
            spectro_paths += master['spectrogram_paths']
            transcription_paths += master['transcription_paths']
            id_paths += master['id_paths']
            self._num_chunks += master['num_chunks']
            self._total_samples += master['num_samples']
            self._max_spectro_feature_length = max(self._max_spectro_feature_length,
                                                   master['max_spectro_feature_length'])
       
        file_lists = {
                        'spectrogram':   spectro_paths, 
                        'transcription': transcription_paths,
                        'id':            id_paths
                     }

        keep_lists = []
        for feature in features:
            keep_lists.append(file_lists[feature])
        self._all_paths = list(zip(*keep_lists))


    def batch_generator(self):
        """Generator that randomly yields features

        Batch generator that yields features specified during initialization.
        See Notes for more details about implementation.

        Notes:
            All of our data is split up into chunks, meaning we have to 
            do a lot of processing to randomly sample.
            
            For a single epoch, the gist of it is this:
            1. Randomly load N chunks, call these C 
            2. Randomly sample from C
            3. When all samples are exhausted (C is empty, or has 
               less samples than batch size), go back to 1

        Yields:
            list of tuples

        """
        self.num_epochs = 0
        N = int(np.ceil(self._chunk_pct * self._num_chunks))

        data = []
        while True:
            self.num_epochs += 1
            remaining_chunks = copy.deepcopy(self._all_paths)
            while remaining_chunks:
                # Load up N chunks
                file_queue = randomly_sample_stack(remaining_chunks, N)

                feature_buffer = []
                for feature_file_tuple in file_queue:
                    feature_file_data = []
                    for feature_file in feature_file_tuple:
                        feature_file_data.append(pickle.load(open(feature_file, 'rb')))
                    feature_buffer.append(feature_file_data)

                data += list(zip(*feature_buffer))
                # TODO - pad/truncate the data
                while len(data) > self._batch_size:
                    # Note: If batch size > remaining elements, we just load the next chunk
                    batch = randomly_sample_stack(data, self._batch_size)
                    yield batch

