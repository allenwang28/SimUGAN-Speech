# -*- coding: utf-8 *-* 
"""Speech Data Generator

This module is used to provide a batch generator for 
training our models.

Since we preprocess all of our data the same way, the
SpeechBatchGenerator only has to have access to the folders

"""

import os
import numpy as np
import sys

import time

import pickle
import copy

from SimUGANSpeech.util.data_util import randomly_sample_stack, pad_or_truncate
from SimUGANSpeech.util.audio_util import get_spectrograms

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
        self._batch_size = batch_size
        self._chunk_pct = chunk_pct

        if len(feature_sizes) != len(features):
            raise ValueError('Length of feature_sizes should match length of features')
        self._verbose = verbose

        spectro_paths = []
        transcription_paths = []
        id_paths = []
        file_paths = []
        self._num_chunks = 0
        self._total_samples = 0
        self._max_spectro_feature_length = 0
        self._max_text_length = 0

        if self._verbose:
            print ("Loading the master file...")
            s = time.time()


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
            file_paths += master['file_paths']
            self._num_chunks += master['num_chunks']
            self._total_samples += master['num_samples']
            self._max_spectro_feature_length = max(self._max_spectro_feature_length,
                                                   master['max_spectro_feature_length'])
            self._max_text_length = max(self._max_text_length, master['max_text_length'])

        # If the specified spectrogram feature size not provided, set it to the max
        for i, feature in enumerate(features):
            if feature == "spectrogram":
                if not feature_sizes[i]:
                    feature_sizes[i] = self._max_spectro_feature_length
            elif feature == "transcription":
                if not feature_sizes[i]:
                    feature_sizes[i] = self._max_text_length

        self._feature_sizes = feature_sizes

        if self._verbose:
            print ("Finished loading the master file in {0} seconds.".format(time.time() - s))

        file_lists = {
                        'spectrogram':   spectro_paths, 
                        'transcription': transcription_paths,
                        'id':            id_paths
                     }

        keep_lists = []
        for feature in features:
            keep_lists.append(file_lists[feature])
        self._all_paths = list(zip(*keep_lists))

        self._flacs = []
        for f in file_paths: 
            self._flacs += list(np.load(f))

    def batch_generator(self):
        N = int(np.ceil(self._chunk_pct * len(self._flacs)))
        remaining_files = copy.deepcopy(self._flacs)
        while True:
            paths_batch = randomly_sample_stack(remaining_files, N)
            spectrograms = pad_or_truncate(get_spectrograms(paths_batch), self._max_spectro_feature_length)
            while len(spectrograms) > self._batch_size:
                batch = randomly_sample_stack(spectrograms, self._batch_size)
                yield list(map(list, zip(*batch)))



    def batch_generator1(self):
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
            list of lists: Shape = (num_features, batch_size)

        """
        self.num_epochs = -1
        N = int(np.ceil(self._chunk_pct * self._num_chunks))

        if self._verbose:
            print ("Loading {0} files at once.".format(N))

        data = []
        while True:
            self.num_epochs += 1
            remaining_chunks = copy.deepcopy(self._all_paths)
            while remaining_chunks:
                # Load up N chunks
                file_queue = randomly_sample_stack(remaining_chunks, N)

                if self._verbose:
                    s = time.time()
                    print ("Loading chunks...")

                for feature_file_tuple in file_queue:
                    feature_file_data = []
                    for feature_size, feature_file in zip(self._feature_sizes, feature_file_tuple):
                        #fd = pickle.load(open(feature_file, 'rb'))
                        fd = np.load(open(feature_file, 'rb'))
                        fd = pad_or_truncate(fd, feature_size)
                        feature_file_data.append(fd)
                    data += list(zip(*feature_file_data))

                if self._verbose:
                    print ("Finished loading chunks in {0}".format(time.time() - s))

                while len(data) > self._batch_size:
                    # Note: If batch size > remaining elements, we just load the next chunk
                    batch = randomly_sample_stack(data, self._batch_size)
                    yield list(map(list, zip(*batch)))

