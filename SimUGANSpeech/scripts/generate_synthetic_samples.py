# -*- coding: utf-8 *-* 
"""Script used for generating synthetic samples

Notes:
    gTTS can only save into mp3 files. Soundfile (which
    our audio module uses to open files) only accepts
    .wav or .flac files.

    Therefore we need to convert these .mp3 to
    .flac files, which we'll (hackily) do using ffmpeg

    Install here:
    https://github.com/adaptlearning/adapt_authoring/wiki/Installing-FFmpeg

Issues:
    It seems that gTTS only provides one voice. This can be
    a problem if we decide to try other voices later.

"""
import os
from gtts import gTTS
import pickle
import time
import sys
import shutil

import argparse

from subprocess import call

from SimUGANSpeech.definitions import SYNTHETIC_DIR
from SimUGANSpeech.definitions import LIBRISPEECH_DIR 
from SimUGANSpeech.data.librispeechdata import POSSIBLE_FOLDERS
from SimUGANSpeech.util.data_util import chunkify
import SimUGANSpeech.util.audio_util as audio_util

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

        transcription_paths = master['transcription_paths']
        for tp in transcription_paths:
            transcriptions += list(pickle.load(open(tp, 'rb')))

    return transcriptions


def generate_speech(text, save_path, slow):
    """Generate and save a synthetic tts sample 

    Notes:
        Uses google's tts API (gTTS)

        Can explore other options, but gTTS seems convenient and simple


    Args:
        text (str): The text to be spoken
        save_path (str): Path to save the sample to
        slow (bool): Whether or not to have the voice speak slowly

    """
    tts = gTTS(text=text, lang='en', slow=slow)
    mp3_save = os.path.join(save_path, ".mp3")
    tts.save(mp3_save)
    call(["ffmpeg", "-i", mp3_save, save_path])
    os.remove(mp3_save)


def generate_speech_from_texts(save_dir, texts, slow, verbose=True): 
    """Generate and save speech samples from a list of text

    Notes:
        If len(texts) = N, then the samples created will be:

        save_dir/0.flac
        save_dir/1.flac
        ...
        sav_dir/N-1.flac

        Further, a file_map object will be saved that maps
        each .flac file to their respective transcription. 
        This is saved to 

        save_dir/file_map.pkl
    
    Args:
        save_dir (str): Path to save samples to
        texts (list of str): The list of texts to generate
            speech samples.
        slow (bool): Whether or not to have the voice speak slowly
        verbose (:obj:`bool`, optional): verbosity. Defaults to True

    """
    file_map = {}

    if not os.path.exists(save_dir):
        if verbose:
            print ("{0} doesn't exist. Creating folder now...".format(save_dir))
        os.makedirs(save_dir)

    if verbose:
        print ("Provided {0} samples".format(len(texts)))
        start = time.time()

    for i, text in enumerate(texts):
        if verbose:
            pct_complete = i / len(texts)
            msg = "\r- Generation progress: {0:.1%}".format(pct_complete)
            sys.stdout.write(msg)
            sys.stdout.flush()
        sample_save_path = os.path.join(save_dir, '{0}.flac'.format(i))
        generate_speech(text, sample_save_path, slow)
        file_map[sample_save_path] = text 

    file_map_path = os.path.join(save_dir, 'file_map.pkl')

    if verbose:
        end = time.time()
        print ("")
        print ("Finished generating all of the samples in {0} seconds.".format(end - start))
        print ("Now saving the file map to {0}".format(file_map_path))

    pickle.dump(file_map, open(file_map_path, 'wb'))


def process_synthetic_data(save_dir, num_chunks, verbose=True):
    """Take all generated synthesized data and process them

    For all synthesized data, create chunks for:
    - transcriptions
    - spectrograms

    For any given chunk, i, the information will be saved to:
    {save_dir}/transcription-i.pkl
    {save_dir}/spectrograms-i.pkl

    Args:
        save_dir (str): Path to the generated synthesized data
        num_chunks (int): The number of chunks to save data to
        verbose (:obj:`bool`, optional): Verbosity. Defaults to True 

    """
    file_map_path = os.path.join(save_dir, 'file_map.pkl')
    file_map = pickle.load(open(file_map_path, 'rb'))

    master_file_path = os.path.join(save_dir, 'master.pkl')

    all_files = list(file_map.keys())

    file_chunks = chunkify(all_files, num_chunks)
    master = {}
    master['num_chunks'] = num_chunks
    master['num_samples'] = len(all_files)
    master['transcription_paths'] = []
    master['spectrogram_paths']  = []
    msfl = 0

    for i in range(num_chunks):
        if verbose:
            print ("--------------------------")
            print ("Processing chunk {0} of {1}".format(i+1, num_chunks))
        file_chunk = file_chunks[i]
        transcription_path = os.path.join(save_dir, 'transcription-{0}.pkl'.format(i))
        spectrogram_path = os.path.join(save_dir, 'spectrograms-{0}.pkl'.format(i))

        master['transcription_paths'].append(transcription_path)
        master['spectrogram_paths'].append(spectrogram_path)

        pickle.dump([file_map[f] for f in file_chunk], open(transcription_path, 'wb'))

        s = audio_util.get_spectrograms(file_chunk,
                                        params=None,
                                        maximum_size=None,
                                        save_path=spectrogram_path,
                                        verbose=verbose)
        msfl = max(msfl, s[0].shape[0])
    if verbose:
        print ("Saving master file to {0}".format(master_file_path))
    master['max_spectro_feature_length'] = msfl
    pickle.dump(master, open(master_file_path, 'wb'))




if __name__ == "__main__":
    # Arg parse...
    parser = argparse.ArgumentParser(description="Script used to generate and preprocess synthesized data.")

    # Options to download and generate spectrograms
    parser.add_argument("--generate", action='store_true',
                        help="Generate speech samples")
    parser.add_argument("--process", action='store_true',
                        help="Process the speech samples")
    parser.add_argument("--clean", action='store_true',
                        help="Delete all existing processed data (but don't delete LibriSpeech folder)")
    parser.add_argument("--num_chunks", default=5, type=int,
                        help="Number of chunks to save to. Should be >1 to avoid MemoryError.")
    parser.add_argument("--slow", action='store_true',
                        help="Whether or not the speaker should speak slowly.")

    # Options for which folders to use
    for folder in POSSIBLE_FOLDERS: 
        parser.add_argument("--{0}".format(folder), action='store_true',
                            help="Include {0}".format(folder))

    parser.add_argument("--all", action='store_true',
                        help="Include all possible folders")

    # Verbosity
    parser.add_argument("--verbose", action='store_true',
                        help="Verbosity")

    # Process Parameters
    args = parser.parse_args()

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

        shutil.rmtree(save_dir) 
        print ("Completed cleaning process.")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.generate:
        transcriptions = get_librispeech_texts(folder_names)
        generate_speech_from_texts(save_dir, transcriptions, args.slow, verbose=args.verbose)
    if args.process:
        process_synthetic_data(save_dir, args.num_chunks, verbose=args.verbose)

