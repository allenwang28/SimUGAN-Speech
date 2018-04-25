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
from SimUGANSpeech.preprocessing.audio import AudioParams

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
                 audio_params=None,
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
            audio_params (:obj:`AudioParameters`, optional): Parameters for audio
                See /preprocessing/audio.py for more information.
                Defaults to None
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
        self._feature_sizes = feature_sizes

        if len(feature_sizes) != len(features):
            raise ValueError('Length of feature_sizes should match length of features')
        self._verbose = verbose

        audio_files = []
        ids = []
        transcriptions = []
        self._audio_params = audio_params

        if self._verbose:
            print ("Loading the master file...")
            t = time.time()
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
            audio_files += master['paths']
            transcriptions += master['transcriptions']
            ids += master['ids']
        
        if self._verbose:
            print ("Finished loading the master file in {0}".format(time.time() - t))

        assert (len(audio_files) == len(ids) == len(transcriptions))
        self._chunk_size = int(np.ceil(self._chunk_pct * len(audio_files)))
        self._all_files = list(zip(audio_files, ids, transcriptions))

        self.epoch = -1


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
            list of lists: Shape = (num_features, batch_size)

        """
        data = []
        while True:
            self.epoch += 1
            remaining_files = copy.deepcopy(self._all_files)
            while remaining_files:
                chunk = randomly_sample_stack(remaining_files, self._chunk_size)
                af_chunk, id_chunk, t_chunk = zip(*chunk)

                for feature, feature_size in zip(self._features, self._feature_sizes):
                    fdata = []
                    if feature == 'spectrogram':
                        spectrograms = get_spectrograms(af_chunk, verbose=self._verbose, params=self._audio_params)
                        fdata.append(pad_or_truncate(spectrograms, feature_size))
                    elif feature == "id":
                        fdata.append(id_chunk)
                    elif feature == "transcription":
                        fdata.append(pad_or_truncate(t_chunk, feature_size))
                data += list(zip(*(fdata)))

                while len(data) > self._batch_size:
                    batch = randomly_sample_stack(data, self._batch_size)
                    yield list(map(list, zip(*batch)))

