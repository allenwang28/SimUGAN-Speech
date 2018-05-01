# -*- coding: utf-8 *-* 
"""Script used for generating synthetic samples

Notes:
    We use the SAPI python interface from 
    https://github.com/DeepHorizons/tts

    This means that generation is only supported on Windows machines.

    A lot of the issues are due to existing Python TTS systems not 

TODO:

"""
import os
#import tts.sapi # this import is only used when generating the samples
import pickle
import time
import sys
import shutil
import numpy as np

import re

import argparse

from SimUGANSpeech.definitions import SYNTHETIC_DIR
from SimUGANSpeech.definitions import LIBRISPEECH_DIR 
from SimUGANSpeech.data.librispeechdata import POSSIBLE_FOLDERS
from SimUGANSpeech.util.data_util import chunkify
import SimUGANSpeech.util.audio_util as audio_util

from deco import concurrent, synchronized

def get_librispeech_texts(folder_paths):
    """Get all transcriptions from any saved LibriSpeech data

    Scans LIBRISPEECH_DIR for any folders and gets the list of
    all transcriptions.

    Notes:
        This function assumes librispeech_initialize.py has been
        run, with all LibriSpeech information processed

    Args:   
        folder_paths (list of str): List of the folder names
            (e.g., dev-clean, dev-test, etc.)

    Returns:
        List of strings: all transcriptions

    """
    transcriptions = []
    for fpath in [os.path.join(LIBRISPEECH_DIR, f) for f in folder_paths]:
        if not os.path.exists(fpath):
            raise ValueError("{0} doesn't exist".format(fpath))

        master_path = os.path.join(fpath, 'master.pkl')
        try:
            master = pickle.load(open(master_path, 'rb'))
        except:
            raise RuntimeError("""
                There was a problem with loading the master file, {0}.\n
                Make sure librispeech_initialize.py is run in /scripts
            """.format(master_path)) 

        transcriptions = master['transcriptions']

    return transcriptions

def generate_speech(voice, text, save_path):
    """Generate and save a synthetic tts sample 

    Notes:
        Uses a SAPI wrapper (meaning that this is only available on windows)

    Args:
        text (str): The text to be spoken
        save_path (str): Path to save the sample to

    """
    voice.create_recording(save_path, text)


def generate_speech_from_texts(save_dir, texts, percentage=1.0, verbose=True): 
    """Generate and save speech samples from a list of text

    Since we use SAPI, we have to specify the voice as well.
    We will generate a sample from every available voice.

    Notes:
        If len(texts) = N and there are two possible voices,
        then the samples created will be:

        save_dir/0-v0.flac
        save_dir/0-v1.flac
        save_dir/1-v0.flac
        save_dir/1-v1.flac
        ...

        Further, a file_map object will be saved that maps
        each .flac file to their respective transcription. 
        This is saved to 

        save_dir/file_map.pkl
    
    Args:
        save_dir (str): Path to save samples to
        texts (list of str): The list of texts to generate
            speech samples.
        percentage (:obj:`float`, optional): The percent of 
            samples to process. Defaults to 1.0
        verbose (:obj:`bool`, optional): verbosity. Defaults to True

    """
    master = {}

    master['paths'] = []
    master['transcriptions'] = []
    master['ids'] = []

    voice = tts.sapi.Sapi()

    num_to_process = int(np.ceil(percentage * len(texts)))
    texts = texts[:num_to_process]

    if not os.path.exists(save_dir):
        if verbose:
            print ("{0} doesn't exist. Creating folder now...".format(save_dir))
        os.makedirs(save_dir)

    if verbose:
        print ("Provided {0} samples".format(len(texts)))
        start = time.time()

    num_voices = len(voice.get_voice_names())
    total_generated = num_voices * len(texts)
    num_samples = len(texts)

    for j, vname in enumerate(voice.get_voice_names()):
        voice.set_voice(vname)
        for i, text in enumerate(texts):
            if verbose:
                pct_complete = (j * num_samples + i) / total_generated
                msg = "\r- Generation progress: {0:.1%}".format(pct_complete)
                sys.stdout.write(msg)
                sys.stdout.flush()

            save_name = '{0}-v{1}.flac'.format(i, j)
            sample_save_path = os.path.join(save_dir, save_name)
            if not os.path.exists(sample_save_path):
                generate_speech(voice, text, sample_save_path)
            master['paths'].append(sample_save_path)
            master['ids'].append(j)
            master['transcriptions'].append(text)

    master_path = os.path.join(save_dir, 'master.pkl')
    if verbose:
        end = time.time()
        print ("")
        print ("Finished generating all of the samples in {0} seconds.".format(end - start))
        print ("Now saving the master path to {0}".format(master_path))

    pickle.dump(master, open(master_path, 'wb'))


def remap_paths(save_dir, verbose):
    if verbose:
        print ("Remapping master file")
    
    master_path = os.path.join(save_dir, 'master.pkl')
    master = pickle.load(open(master_path, 'rb'))
    master['paths'] = [os.path.join(save_dir, os.path.basename(p)) for p in master['paths']]
    pickle.dump(master, open(master_path, 'wb'))



if __name__ == "__main__":
    # Arg parse...
    parser = argparse.ArgumentParser(description="Script used to generate and preprocess synthesized data.")

    # Options to generate samples and generate spectrograms
    parser.add_argument("--clean", action='store_true',
                        help="Delete all existing data")
    parser.add_argument("--percentage", default=1.0, type=float,
                        help="Between 0.0 and 1.0. Use this to only process a subset of the data.")
    parser.add_argument("--remap", action='store_true',
                        help="Remap the master file")

    # Options for which folders to use
    for folder in POSSIBLE_FOLDERS: 
        parser.add_argument("--{0}".format(folder), action='store_true',
                            help="Include {0} LibriSpeech".format(folder))

    parser.add_argument("--all", action='store_true',
                        help="Include all possible LibriSpeech folders")

    # Verbosity
    parser.add_argument("--verbose", action='store_true',
                        help="Verbosity")

    # Process Parameters
    args = parser.parse_args()

    if args.percentage < 0.0 or args.percentage > 1.0:
        raise ValueError("Percentage must be between 0 and 1")

    folder_dir = LIBRISPEECH_DIR
    folder_names = []

    # Get all paths to LibriSpeech data
    if args.all:
        folder_names = POSSIBLE_FOLDERS
    else:
        for folder in POSSIBLE_FOLDERS:
            # Optional arguments mess up the folder names.
            folder_key = folder.replace("-", "_") 
            if vars(args)[folder_key]:
                folder_names.append(folder)

    if args.verbose:
        print ("Processing: ")
        for fname in folder_names:
            print ("- {0}".format(fname))

    save_dir = SYNTHETIC_DIR

    if args.clean:
        # Delete all of the saves that we have already
        if args.verbose:
            print ("Cleaning {0}".format(save_dir))

        if os.path.exists(save_dir):
            shutil.rmtree(save_dir) 
        print ("Completed cleaning process.")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.remap:
        remap_paths(save_dir, args.verbose)
    else:
        transcriptions = get_librispeech_texts(folder_names)
        generate_speech_from_texts(save_dir, transcriptions, percentage=args.percentage, verbose=args.verbose)
