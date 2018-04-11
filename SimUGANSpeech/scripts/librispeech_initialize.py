# -*- coding: utf-8 *-* 
"""Script used to download and preprocess LibriSpeech data. 

This module is used for downloading and processing the files 
provided by LibriSpeech

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

import numpy as np
import os

import urllib
import pickle
import tarfile
import sys

import SimUGANSpeech.data.audio as audio
import SimUGANSpeech.util.audio_util as audio_util
from SimUGANSpeech.util.data_util import chunkify

from SimUGANSpeech.definitions import LIBRISPEECH_DIR

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


def flacpath_transcription_id(folder_path):
    """Get a all .flac files, transcriptions, and ids in a path

    Args:
        folder_path (str): Path to the folder

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
                ids += [voice_id] * len(flacs)

        res = {'paths': flac_files, 'transcriptions': transcriptions, 'ids': ids}

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
        else:
            if verbose:
                print ("{0} found. Skipping...".format(fname))



def librispeech_initialize(folder_dir,
                           folder_names,
                           num_chunks,
                           verbose=True):
    """Download LibriSpeech data, preprocess and save 

    For each LibriSpeech dataset, extract:
    - transcriptions
    - ids 
    - spectrograms
    into multiple chunks.

    Assuming we're processing dev-clean at /path/to/dev-clean, 
    for batch i, these are saved as:

    /path/to/dev-clean/transcriptions-i.pkl
    /path/to/dev-clean/ids-i.pkl
    /path/to/dev-clean/spectrograms.pkl

    Args:
        folder_dir (str): The path to LibriSpeech 
        folder_paths (list of str): List of the folder names
            (e.g., dev-clean, dev-test, etc.)
        num_chunks (int): The number of chunks to split
            the data into per folder name. This is required
            since there is a lot of data.
        verbose (:obj:`bool`, optional): Whether or not to
            print statements. Defaults to True.

    """
    _maybe_download_and_extract(folder_dir, folder_names, verbose)

    folder_paths = [os.path.join(folder_dir, fname) for fname in folder_names]
    for fpath in folder_paths:
        data = flacpath_transcription_id(fpath)

        transcription_chunks = chunkify(data['transcriptions'], num_chunks)
        id_chunks = chunkify(data['ids'], num_chunks)
        path_chunks = chunkify(data['paths'], num_chunks)
        
        for i in range(num_chunks):
            transcription_path = os.path.join(fpath, "transcription-{0}.pkl".format(i))
            id_path = os.path.join(fpath, "id-{0}.pkl".format(i))
            spectrograms_path = os.path.join(fpath, "spectrograms-{0}.pkl".format(i))

            pickle.dump(transcription_chunks[i], open(transcription_path, 'wb'))
            pickle.dump(id_chunks[i], open(id_path, 'wb'))

            # This function will automatically save for us
            audio_util.get_spectrograms(path_chunks[i],
                                        params=None,
                                        maximum_size=None,
                                        save_path=spectrograms_path,
                                        verbose=verbose)


if __name__ == '__main__':
    # Parameters
    folder_path = LIBRISPEECH_DIR
    folder_names = [ 'dev-clean' ]
    features = [ 'spectrogram' ]

    librispeech_initialize(folder_path, folder_names, 10, verbose=True)

