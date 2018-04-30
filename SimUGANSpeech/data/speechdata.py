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

from SimUGANSpeech.util.data_util import randomly_sample_stack, truncate, pad_or_truncate, randomly_split
from SimUGANSpeech.util.audio_util import get_audio_features 
from SimUGANSpeech.preprocessing.audio import AudioParams

DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_TIME_STEPS = 50
DEFAULT_MAX_OUTPUT_LENGTH = 50

DEFAULT_CHUNK_PROCESS_PERCENTAGE = 0.3
DEFAULT_VALIDATION_PCT=0.3

ACCEPTED_LABELS =   ['transcription_chars',
                     'voice_id']

FEATURES = [ 
             'spectrogram',
             'mfcc',
             'transcription',
             'id',
           ]


def load_speech_data(folder_dir, folder_names):
    """Load speech data from master file

    Using the data processed from the speech generation scripts,
    extract the audio file paths, transcriptions, and ids.

    Args:
        folder_dir (str): The directory of the data
        folder_names (str): The individual folder names 

    Returns:
        list of tuples: Each entry is a tuple, each are: 
            (audio_file, transcription, id)

    """
    audio_files = []
    ids = []
    transcriptions = []
    if not folder_names:
        return []
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

    assert (len(audio_files) == len(ids) == len(transcriptions))
    speech_data = list(zip(audio_files, ids, transcriptions))  
    return speech_data


class SpeechBatchGenerator(object):
    def __init__(self,
                 folder_dir,
                 training_folder_names,
                 testing_folder_names,
                 features,
                 feature_sizes,
                 validation_pct=DEFAULT_VALIDATION_PCT,
                 audio_params=None,
                 batch_size=DEFAULT_BATCH_SIZE,
                 chunk_pct=DEFAULT_CHUNK_PROCESS_PERCENTAGE,
                 verbose=True):
        """SpeechBatchGenerator class initializer
        
        Args:
            folder_dir (str): The path to the data folder
            training_folder_names (list of str): List of the folder names (or datasets)
                (e.g., dev-clean, dev-test, etc.) to be used for training
            training_folder_names (list of str): List of the folder names (or datasets)
                (e.g., dev-clean, dev-test, etc.) to be used for testing 
            features (list of str): List of desired features.
                See constant defined FEATURES for list of valid features.
            feature_sizes (list of int): List of maximum length of features.
                Has to be the same shape as features. The features will be
                truncated or padded to match the specified shape.
            validation_pct (:obj:`float`, optional): The percent of training
                data to be held back for validation.
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
                If None provided, all data is loaded at once with no batch generator.
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
        

        training_data = load_speech_data(folder_dir, training_folder_names)
        self._test_data = load_speech_data(folder_dir, testing_folder_names)

        self._validation_data, self._training_data = randomly_split(training_data, validation_pct)
        self.num_training_samples = len(self._training_data)
        self.num_batches = int(np.ceil(self.num_training_samples / self._batch_size))

        if chunk_pct:
            self._training_batch_generator = self.chunk_data_batch_generator(self._training_data)
        else:
            self._training_batch_generator = self.data_batch_generator(self._training_data)

        self._audio_params = audio_params


    def _get_speechdata_features(self, speechdata):
        """Extract the features from speechdata

        From the speech data (list of tuples), extract
        user-specified features

        Args:
            speechdata (list of tuples): The list of 
                (audio file, transcription, id) tuples

        Returns:
            list of tuples: The features 

        """
        audio_files, ids, transcriptions = zip(*speechdata)
        feature_data = []
        for feature, feature_size in zip(self._features, self._feature_sizes):
            if feature == 'spectrogram' or feature == 'mfcc':
                features = get_audio_features(audio_files, 
                                              feature,
                                              verbose=self._verbose, 
                                              maximum_size=feature_size,
                                              params=self._audio_params)
                feature_data.append(features)
            elif feature == "id":
                feature_data.append(ids)
            elif feature == "transcription":
                feature_data.append(truncate(transcriptions, feature_size))
        return list(zip(*(feature_data)))


    def get_training_batch(self):
        """Get a batch of training data"""
        return next(self._training_batch_generator)


    def get_test_data(self):
        """Get all testing data"""
        return list(map(list, zip(*self._get_speechdata_features(self._test_data))))


    def get_validation_data(self):
        """Get all validation data"""
        return list(map(list, zip(*self._get_speechdata_features(self._validation_data))))


    def data_batch_generator(self, speech_data):
        """Generator that randomly yields features 

        This batch generator differs in that all features are loaded 
        immediately, not in chunks. This should be used for smaller features
        (like transcriptions, ids, mfccs) and not larger features
        (like spectrograms). 

        Args:
            speech_data (list of str): The speech data
        
        Yields:
            list of lists: Shape = (num_features, batch_size)

        """
        data = self._get_speechdata_features(speech_data)
        while True:
            indices = list(range(len(data)))

            while indices:
                indices_batch = randomly_sample_stack(indices, self._batch_size)
                batch = [data[i] for i in indices_batch]
                yield list(map(list, zip(*batch)))


    def chunk_data_batch_generator(self, speech_data):
        """Generator that randomly yields features in chunks

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
        
        Args:
            speech_data (list of str): The speech data

        Yields:
            list of lists: Shape = (num_features, batch_size)

        """
        data = []
        chunk_size = int(np.ceil(self._chunk_pct * len(speech_data)))
        while True:
            remaining_files = copy.deepcopy(speech_data)
            while remaining_files:
                chunk = randomly_sample_stack(remaining_files, chunk_size)
                feature_data_chunk = self._get_speechdata_features(chunk)
                data += feature_data_chunk

                while len(data) > self._batch_size:
                    batch = randomly_sample_stack(data, self._batch_size)
                    yield list(map(list, zip(*batch)))

