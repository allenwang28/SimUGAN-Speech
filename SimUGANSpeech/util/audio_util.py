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

import SimUGANSpeech.data.audio as audio
import time

from deco import concurrent, synchronized
import pickle

@concurrent 
def get_spectrogram_from_path_cc(path, params):
    """Utility function for concurrent spectrogram from path"""
    return audio.get_spectrogram_from_path(path,
                                           highcut=params.highcut,
                                           lowcut=params.lowcut,
                                           log=params.spectro_log,
                                           thresh=params.spectro_thresh,
                                           frame_size_in_ms=params.frame_size_in_ms,
                                           frame_stride_in_ms=params.frame_stride_in_ms,
                                           real=params.real)

@synchronized
def get_spectrograms_from_path_cc(audio_files, params):
    """Utility function for synchronizing concurrent spectrogram from paths"""
    spectrograms = []
    for path in audio_files:
        spectrograms.append(get_spectrogram_from_path_cc(path, params))
    return spectrograms


@concurrent
def pad_spectrogram(spectro, maximum_size):
    """Utility function for concurrent padding of a spectrogram"""
    size = spectro.shape[0]
    if size > maximum_size:
        return spectro[:maximum_size, :]
    else:
        pad_length = maximum_size - size
        return np.pad(spectro, ((0, pad_length), (0,0)), 'constant')


@synchronized
def pad_spectrograms(spectrograms, maximum_size):
    """Utility function for synchronizing concurrent padding of spectrograms"""
    fixed_spectrograms = []
    for spectro in spectrograms:
        fixed_spectrograms.append(pad_spectrogram(spectro, maximum_size))
    return fixed_spectrograms


# TODO - try looking at numpy again later.
# Using pickle because we get a MemoryError 
def get_spectrograms(audio_files, params=None, maximum_size=None, save_path=None, verbose=True):
    """Get all spectrograms from audio files.

    Given a list of audio files, return a list of padded/truncated 2D numpy
    arrays with shape: (num_windows, window_size)

    Args:
        audio_files (list of str): all audio files
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
        params = audio.SpectrogramParams()
    else:
        params = params

    try: 
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
            return data
    except:
        if verbose:
            print ("Generating spectrograms from audio files.")
            start = time.time()

        spectrograms = get_spectrograms_from_path_cc(audio_files, params)

        if verbose:
            end = time.time()
            print ("Completed generating spectrograms in {0} seconds".format(end - start))
            print ("Now padding/truncating each spectrogram")
            start = time.time()
        
        if not maximum_size:
            maximum_size = max(spectro.shape[0] for spectro in spectrograms) 

        spectrograms = pad_spectrograms(spectrograms, maximum_size)

        if verbose:
            end = time.time()
            print ("Completed padding/truncated each spectrogram in {0} seconds".format(end - start))

        if save_path:
            if verbose:
                print ("Saving spectrograms to {0}".format(save_path))
            pickle.dump(spectrograms, open(save_path, 'wb'))
        return spectrograms


