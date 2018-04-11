# -*- coding: utf-8 *-* 
"""LibriSpeech Folder Parsing Module

This module is used for processing the files provided by LibriSpeech
to be used in our models.

The LibriSpeech folder structure (from LibriSpeech/README.txt):
<corpus root>
    |
    .- README.TXT
    |
    .- READERS.TXT
    |
    .- CHAPTERS.TXT
    |
    .- BOOKS.TXT
    |
    .- train-clean-100/
                   |
                   .- 19/
                       |
                       .- 198/
                       |    |
                       |    .- 19-198.trans.txt
                       |    |    
                       |    .- 19-198-0001.flac
                       |    |
                       |    .- 14-208-0002.flac
                       |    |
                       |    ...
                       |
                       .- 227/
                            | ...
, where 19 is the ID of the reader, and 198 and 227 are the IDs of the chapters
read by this speaker. The *.trans.txt files contain the transcripts for each
of the utterances, derived from the respective chapter and the FLAC files contain
the audio itself.

Notes:
    So far, we have only been working from the dev-clean folder.
    It should work for all LibriSpeech folders, assuming their folder structures
    are consistent.

Todo:
    * Options for specifying which cepstral coefficients
    * Test for other folders
    * Add other ways to specify sequences (right now, only characters are supported)
    * Find a better way to prepare data for tensorflow
    * Find a way to shuffle the data from the batch generator
    * Probably move downloading functionality into its own file once we support
      more than just LibriSpeech
"""

import os
import numpy as np
import re
import urllib
import tarfile

import pickle

import sys

# import tensorflow as tf

import src.data.audio as audio
import src.tools.audio_util as audio_util


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

LIBRISPEECH_URL_BASE = "http://www.openslr.org/resources/12/{0}"

def _voice_txt_dict_from_path(folder_path):
    """Creates a dictionary between voice ids and txt file paths

    Walks through the provided LibriSpeech folder directory and 
    creates a dictionary, with voice_id as a key and all associated text files

    Args:
        folder_path (str): The path to the LibriSpeech folder
    
    Returns:
        dict : Keys are the voice_ids, values are lists of the paths to trans.txt files.

    """
    voice_txt_dict = {}

    for dir_name, sub_directories, file_names in os.walk(folder_path):
        if len(sub_directories) == 0: # this is a root directory
            file_names = list(map(lambda x: os.path.join(dir_name, x), file_names))
            txt_file = list(filter(lambda x: x.endswith('txt'), file_names))[0]

            voice_id = os.path.basename(dir_name)

            if voice_id not in voice_txt_dict:
                voice_txt_dict[voice_id] = [txt_file]
            else:
                voice_txt_dict[voice_id].append(txt_file)

    return voice_txt_dict

def _transcriptions_and_flac(txt_file_path):
    """Gets the transcriptions and .flac files from the path to a trans.txt file.

    Given the path to a trans.txt file, this function will return a list of 
    transcriptions and a list of the .flac files. Each flac file entry index corresponds 
    to the transcription entry index.

    Args:
        txt_file_path (str): The path to a trans.txt file

    Returns:
        list : A list of transcriptions
        list : A list of paths to .flac files
    """
    transcriptions = []
    flac_files = []

    parent_dir_path = os.path.dirname(txt_file_path)

    with open(txt_file_path, 'rb') as f:
        lines = [x.decode('utf8').strip() for x in f.readlines()]
    
    for line in lines:
        splitted = re.split(' ', line)
        file_name = splitted[0]
        transcriptions.append(" ".join(splitted[1:]))
        flac_files.append(os.path.join(parent_dir_path, "{0}.flac".format(file_name)))
 
    return transcriptions, flac_files 


def flacpath_transcription_id(folder_path, save=True):
    """Get a all .flac files, transcriptions, and ids in a path

    Args:
        folder_path (str): Path to the folder
        save (:obj:`bool`, optional): Whether or not
            to save the data into a .pkl file

    Returns:
        dict: Contains flac files, transcriptions, and ids

    """
    # Use pickle so we don't have to run this every time
    pkl_path = os.path.join(folder_path, 'unified.pkl') 

    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            return data
    except:
        # pickle file load unsuccessful
        voice_txt_dict = _voice_txt_dict_from_path(folder_path)
        
        flac_files = []
        transcriptions = []
        ids = []

        for voice_id in voice_txt_dict:
            txt_files = voice_txt_dict[voice_id]
            for txt_file in txt_files:
                t, flacs = _transcriptions_and_flac(txt_file)
                flac_files += flacs
                transcriptions += t

                for flac_file in flac_files:
                    ids.append(voice_id)
        res = {'paths': flac_files, 'transcriptions': transcriptions, 'ids': ids}

        if save:
            pickle.dump(res, open(pkl_path, 'wb'))

        return res


def _print_download_progress(count, block_size, total_size):
    """Print the download progress.

    Used as a callback in _maybe_download_and_extract.

    """
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)

    sys.stdout.write(msg)
    sys.stdout.flush()


def _maybe_download_and_extract(folder_dir, folder_names, verbose):
    """Download and extract data if it doesn't exist.
    
    Args:
        folder_dir (str): The path to LibriSpeech 
        folder_paths (list of str): List of the folder names
            (e.g., dev-clean, dev-test, etc.)
        verbose (bool): Whether or not to progress

    """
    for fname in folder_names:
        if not os.path.exists(os.path.join(folder_dir, fname)):
            tar_file_name = fname + ".tar.gz"
            tar_file_path = os.path.join(folder_dir, tar_file_name)
            url = LIBRISPEECH_URL_BASE.format(tar_file_name)
            
            if verbose:
                print ("{0} not found. Downloading {1}".format(fname, tar_file_name))

            if verbose:
                file_path, _ = urllib.request.urlretrieve(url=url,
                                                          filename=tar_file_path,
                                                          reporthook=_print_download_progress)
                print ()
                print ("Download complete. Extracting {0}".format(tar_file_path))
            else:
                file_path, _ = urllib.request.urlretrieve(url=url,
                                                          filename=tar_file_path)
            tarfile.open(name=tar_file_path, mode="r:gz").extractall(folder_dir)


class LibriSpeechBatchGenerator:
    def __init__(self,
                 folder_dir,
                 folder_names,
                 features,
                 feature_sizes,
                 feature_params=None,
                 batch_size=DEFAULT_BATCH_SIZE,
                 save=None,
                 randomize=True,
                 verbose=True):
        """LibriSpeechBatchGenerator class initializer
        
        Args:
            folder_dir (str): The path to LibriSpeech 
            folder_paths (list of str): List of the folder names
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
            save (:obj:`str`, optional): A path to a directory to save 
                things we process.
                If provided, we save: 
                    - unified flac files, transcriptions, ids 
                    - all listed features, padded
                If None, the batch generator will have to process these every time.
                Defaults to None.
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

        _maybe_download_and_extract(folder_dir, folder_names, verbose) 


        # TODO - maybe check if the files already exist so we can avoid loading things
        # we don't need
        if save:
            save_unified = True

            # TODO - try looking at numpy again later.
            # Using pickle because we get a MemoryError 
            spectro_name = "spectro"
            for fname in folder_names:
                spectro_name += "_" + fname
            # TODO - clean these next 4 lines up
            for feature, feature_size in zip(features, feature_sizes):
                if feature == "spectrogram":
                    if feature_size:
                        spectro_name += feature_size
            spectro_path = os.path.join(save, spectro_name + ".pkl")

        flac_paths = []
        transcriptions = []
        ids = []

        for folder_path in self._folder_paths:
            data = flacpath_transcription_id(folder_path, save=save_unified)
            flac_paths += data['paths']
            transcriptions += data['transcriptions']
            ids += data['ids']
        
        if not feature_params:
            feature_params = [None] * len(features)

        self._data = []
        for feature, feature_size, params in zip(features, feature_sizes, feature_params):
            if feature == "spectrogram":
                self._data.append(audio_util.get_spectrograms(flac_paths,
                                                              params=params,
                                                              maximum_size=feature_size,
                                                              save_path=spectro_path))
            else:
                raise NotImplementedError("Right now we only support spectrograms")


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
