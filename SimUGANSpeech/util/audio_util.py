# -*- coding: utf-8 *-* 
"""Helper functions for audio operations

This module is used to provide helper functions for any audio related
operations. Included functions:
    - get_spectrograms

Notes:
    - This module is different from the audio module in data, as this
      module would work specifically on audio processing operations on 
      files.

Todo:

"""
import numpy as np
import os

import SimUGANSpeech.preprocessing.audio as audio
from SimUGANSpeech.util.data_util import pad_or_truncate
import time

from deco import concurrent, synchronized
import pickle

#@concurrent 
def get_audio_feature_from_path_cc(path, feature, params):
    """Utility function for concurrent spectrogram from path"""
    if feature == 'mfcc':
        return audio.get_mfcc_from_file(path,
                                        max_time_in_s=params.max_time_in_s,
                                        pre_emphasis=params.pre_emphasis,
                                        frame_size_in_ms=params.frame_size_in_ms,
                                        frame_stride_in_ms=params.frame_stride_in_ms,
                                        window_function=params.window_function,
                                        NFFT=params.nfft,
                                        num_filters=params.num_filters,
                                        cep_lifter=params.cep_lifter,
                                        apply_mean_normalize=params.mean_normalize)
    elif feature == 'spectrogram':
        return audio.get_spectrogram_from_path(path,
                                               highcut=params.highcut,
                                               lowcut=params.lowcut,
                                               log=params.spectro_log,
                                               thresh=params.spectro_thresh,
                                               NFFT=params.nfft,
                                               frame_size_in_ms=params.frame_size_in_ms,
                                               frame_stride_in_ms=params.frame_stride_in_ms,
                                               max_time_in_s=params.max_time_in_s,
                                               real=params.real)
    else:
        raise ValueError("Unsupported feature {0} provided".format(feature))          


#@synchronized
def get_audio_features_from_path_cc(audio_files, feature, params):
    """Utility function for synchronizing concurrent spectrogram from paths"""
    # Use a dictionary - concurrent processing does not preserve order
    results = {}
    for path in audio_files:
        results[path] = get_audio_feature_from_path_cc(path, feature, params)
    return [results[path] for path in audio_files] 


def get_audio_features(audio_files, feature, params=None, maximum_size=None, save_path=None, verbose=True):
    """Get all spectrograms from audio files.

    Given a list of audio files, return a list of padded/truncated 2D numpy
    arrays with shape: (num_windows, window_size)

    Args:
        audio_files (list of str): all audio files
        feature (str): 'mfcc' or 'spectrogram'
        params (:obj:`SpectroGramParams`, optional): Used to specify
            parameters for spectrogram generation.
            If None is provided, then the default parameters are used.
            Defaults to None.
        maximum_size (:obj:`int`, optional): Used for zero-padding and/or
            truncation. If None provided, then the maximum will be calculated
            and everything will be zero-padded.
            Defaults to None.
        save_path (:obj:`str`, optional): Path to save the .npy file.
            If None, then the file isn't saved.
            Defaults to None.
        verbose (:obj:`bool`, optional): Whether or not to print progress statements.
            Defaults to True.
    """
    if not params:
        params = audio.AudioParams()
    else:
        params = params

    try: 
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
            return data
    except:
        if verbose:
            print ("Generating {0}s from audio files.".format(feature))
            start = time.time()

        results = get_audio_features_from_path_cc(audio_files, feature, params)

        if verbose:
            end = time.time()
            print ("Completed generating {0}s in {1} seconds".format(feature, end - start))
            print ("Now padding/truncating the {0}s".format(feature))
            start = time.time()
        
        if not maximum_size:
            maximum_size = max(result.shape[0] for result in results)

        results = pad_or_truncate(results, maximum_size)

        if verbose:
            end = time.time()
            print ("Completed padding/truncated each {0} in {1} seconds".format(feature, end - start))

        return results

