# -*- coding: utf-8 *-* 
"""LibriSpeech Folder Parsing Module
"""

import os
import numpy as np
import re


import sys

import tensorflow as tf

import SimUGANSpeech.data.audio as audio

FEATURES = [ 
             'spectrogram',
             'transcription',
             'id',
           ]

POSSIBLE_FOLDERS = [
                     'dev-clean',
                     'dev-other',
                     'test-clean',
                     'test-other',
                     'test-clean-100',
                     'test-clean-360',
                     'train-other-500',
                   ]

DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_TIME_STEPS = 50
DEFAULT_MAX_OUTPUT_LENGTH = 50

NUM_CHAR_FEATURES = 28 # 26 letters + 1 space + 1 EOF (represented as 0s)

ACCEPTED_LABELS =   ['transcription_chars',
                     'voice_id']


class LibriSpeechBatchGenerator:
    def __init__(self,
                 folder_dir,
                 folder_names,
                 features,
                 feature_sizes,
                 feature_params=None,
                 batch_size=DEFAULT_BATCH_SIZE,
                 randomize=True,
                 verbose=True):
        """LibriSpeechBatchGenerator class initializer
        
        Args:
            folder_dir (str): The path to LibriSpeech 
            folder_paths (list of str): List of the folder names (or datasets)
                (e.g., dev-clean, dev-test, etc.)
            features (list of str): List of desired features.
                See constant defined FEATURES for list of valid features.
            feature_sizes (list of int): List of maximum length of features.
                Has to be the same shape as features. The features will be
                truncated or padded to match the specified shape.
                If no maximum/truncation desired, just provide None
            feature_params (:obj:`list of objects`, optional): A list of 
                feature parameters. If None provided, all features will
                use default parameters.
                Otherwise, feature_params should have the same shape as features.
                An entry of None will use default parameters.
                Defaults to None
            batch_size (:obj:`int`, optional): The desired batch size.
                Defaults to 10
            randomize (:obj:`bool`', optional): Whether or not to randomize data
                in batch generation. Defaults to True.
            verbose (:obj:`bool`, optional): Whether or not to print statements.
                Defaults to True.
        
        """
        features = [f.lower() for f in features]
        for feature in features:
            if feature not in FEATURES: 
                raise ValueError('Invalid feature')
        self._features = features
        self._feature_sizes = feature_sizes

        if len(feature_sizes) != len(features):
            raise ValueError('Length of feature_sizes should match length of features')

        self._folder_dir = folder_dir
        self._folder_names = folder_names
        self._folder_paths = [os.path.join(folder_dir, fname) for fname in folder_names]
        self._v = verbose


    def batch_generator(self):
        pass













class LibriSpeechData_:
    def __init__(self, 
                 feature,
                 num_features,
                 label,
                 batch_size, 
                 max_time_steps,
                 max_output_length,
                 folder_paths):
        """LibriSpeechData class initializer

        Args:
            feature (str): The name of the feature you want.
            num_features (int): The number of features you want.
                For instance, if using mfcc, then you may only want 13 cepstral coefficients.
            label (str): The name of the label you want.
            batch_size (int): The size of a batch to be generated. Must be > 0
            max_time_steps (int): The maximum amount of time steps/windows
                accepted for features. Features will be truncated or
                zero-padded.
            max_output_length (int): The maximum length of output sequences.
                If label is voice_id, provide None.
            folder_paths (list of str): All folder paths to use.

        Raises:
            ValueError : If an invalid value for feature is provided
            ValueError : If an invalid value for label is provided

        """
        feature = feature.lower()
        label = label.lower()

        if feature not in ACCEPTED_FEATURES:
            raise ValueError('Invalid feature')
        if label not in ACCEPTED_LABELS:
            raise ValueError('Invalid label')

        self._feature = feature
        self._label = label
        self._folder_paths = folder_paths

        _maybe_download_and_extract(folder_paths)
        self._maybe_preprocess()

        self.batch_size = batch_size
        self.num_features = num_features
        self.max_input_length = max_time_steps
        self.num_output_features = NUM_CHAR_FEATURES
        self.max_output_length = max_output_length

        # 1 input channel
        self.input_shape = (batch_size, num_features, max_time_steps, 1) 
        self.output_shape = (batch_size, NUM_CHAR_FEATURES, max_output_length)

    def _maybe_preprocess(self):
        """Preprocess raw LibriSpeech data and save if not already done
        """
        for folder_path in self._folder_paths:
            bn = os.path.basename(folder_path)
            file_names = ["{0}-mfcc".format(bn),
                          "{0}-power_banks".format(bn), 
                          "{0}-".format(bn), 
                          "{0}-fb".format(bn)]

            self._processed_data_paths = [os.path.join(folder_path, "{0}.npy".format(file_name)) for file_name in file_names]

            if any(not os.path.exists(file_path) for file_path in self._processed_data_paths):
                data = _get_data_from_path(folder_path)
                _save_data(data, folder_path, file_names)

    def _prepare_for_tf(self, inputs, outputs):
        """Prepare inputs and outputs for tensorflow

        Convert transcribed characters to a one hot encoded format

        Args:
            inputs (list of numpy arrays): features
            outputs (list of strings): Transcriptions
        
        Returns:
            np.array : tensorized inputs
            np.array : tensorized outputs

        """
        inputs = np.array(inputs)
        if self._label == 'transcription_chars':
            # One hot encode the outputs 
            outputs = data_util.str_to_one_hot(outputs, self.max_output_length)
        outputs = np.array(outputs)
        return inputs, outputs

    def batch_generator(self, tf=False, randomize=True):
        """Create a batch generator

        Args:
            tf (:obj:`bool`, optional): Whether to yield formatted for tensorflow
            randomize (:obj:`bool`, optional): Whether to randomize 

        Returns:
            generator : batch generator of mfcc features and labels
        """
        if self._feature == "mfcc":
            self._features = np.load(self._processed_data_paths[0])[:, 1:self._num_features]
        elif self._feature == "power_bank":
            self._features = np.load(self._processed_data_paths[1])[:, :self._num_features]
        else:
            raise ValueError('Invalid feature')
        
        

    def _batch_generator_dep(self, tf=False, randomize=True):
        """Create a batch generator

        Deprecated version - we preprocess the data and load it now,
        which allows it to be randomized better.

        Args:
            tf (:obj:`bool`, optional): Whether to yield formatted for tensorflow
            randomize (:obj:`bool`, optional): Whether to randomize 

        Returns:
            generator : batch generator of mfcc features and labels
        """
        for folder_path in self._folder_paths:
            voice_txt_dict = _voice_txt_dict_from_path(folder_path)

            inputs = []
            outputs = []

            for voice_id in voice_txt_dict:
                txt_files = voice_txt_dict[voice_id]
                for txt_file in txt_files:
                    transcriptions, flac_files = _transcriptions_and_flac(txt_file)
                    for transcription, flac_file in zip(transcriptions, flac_files):
                        # Process feature
                        if self._feature == 'mfcc':
                            feature = get_mfcc_from_file(flac_file)[:, 1:self.num_features]
                        elif self._feature == 'power_bank':
                            feature = get_filterbank_from_file(flac_file)[:, :self.num_features]
                        else:
                            raise ValueError('Invalid feature')

                        if len(feature) < self.max_input_length:
                            # zero-pad 
                            pad_length = self.max_input_length - len(feature)
                            feature = np.pad(feature, ((0, pad_length), (0,0)), 'constant')
                        elif len(feature) > self.max_input_length:
                            feature = feature[:self.max_input_length]
                        inputs.append(feature)

                        # Process output
                        if self._label == 'transcription_chars':
                            # Lower case the transcription and keep only letters and spaces
                            transcription = transcription.lower()
                            transcription = re.sub(r'[^a-z ]+', '', transcription)
                            transcription_tokens = list(transcription)
                            if len(transcription_tokens) < self.max_output_length:
                                pad_length = self.max_output_length - len(transcription_tokens)
                                transcription_tokens += ['0'] * pad_length
                            elif len(transcription_tokens) > self.max_output_length:
                                transcription_tokens = transcription_tokens[:self.max_output_length]
                            outputs.append("".join(transcription_tokens))
                        elif self._label == 'voice_id':
                            outputs.append(voice_id)
                        else:
                            raise ValueError('Invalid label')
                   
                        if len(inputs) >= self.batch_size:
                            if tf:
                                inputs, outputs = self._prepare_for_tf(inputs, outputs)
                            yield inputs, outputs

                            inputs = []
                            outputs = []




































def _get_data_from_path(folder_path):
    """Gets spectrograms, transcriptions, and ids from a folder path

    Given the path to a LibriSpeech directory, get all STFT spectrograms from
    all .flac files, their associated speaker (voice_id), and the transcription

    Args:
        folder_path (str): The path to a LibriSpeech folder

    Returns:
        list : The spectrograms 
        list : The transcriptions from .trans.txt files
        list : The voice ids

    """
    voice_txt_dict = voice_txt_dict_from_path(folder_path)

    spectrograms = []
    transcriptions = []
    ids = []

    for voice_id in voice_txt_dict:
        txt_files = voice_txt_dict[voice_id]
        for txt_file in txt_files:
            t, flac_files = transcriptions_and_flac(txt_file)

            transcriptions += t

            for flac_file in flac_files:
                # TODO - add parameters for these
                spectrograms.append(_get_spectrogram_from_file(flac_file))
                power_banks.append(get_filterbank_from_file(flac_file))
                ids.append(voice_id)
    return spectrograms, transcriptions, ids

def _save_data(data, saved_file_paths):
    """Given data, save to numpy arrays

    Given a list of features, transcriptions, and ids, save each to numpy files.

    Args:
        data (tuple): A tuple of 3 lists - spectrograms, transcriptions, ids.
        saved_file_paths (list of strings): Location to save files to 
        
    Returns:

        list of strings: Locations of all the files saved
    """
    spectrograms, transcriptions, ids = data
    saved_file_paths = [os.path.join(folder_path, "{0}.npy".format(file_name)) for file_name in file_names]

    # First 0-pad the features.
    for i, features in enumerate([mfccs, power_banks]):
        max_feature_length = max(feature.shape[0] for feature in features)
        padded_features = []
        for feature in features:
            pad_length = max_feature_length - feature.shape[0]
            padded = np.pad(feature, ((0, pad_length), (0,0)), 'constant')
            padded_features.append(padded)

        padded_features = np.dstack(padded_features)
       
        np.save(saved_file_paths[i], padded_features)

    transcriptions = np.array(transcriptions)
    ids = np.array(ids)

    np.save(saved_file_paths[2], transcriptions)
    np.save(saved_file_paths[3], ids)

    return saved_file_paths

"""
if __name__ == "__main__":
    folder_path = "../data/LibriSpeech"
    libri = LibriSpeechData('mfcc', 12, 'transcription_chars', 10, 150, 100, ['../../data/LibriSpeech'])
 
    batch = libri.batch_generator(tf=True)
    feature, transcription = next(batch)

    print (feature, transcription)
    print (data_util.one_hot_to_str(transcription))
"""
