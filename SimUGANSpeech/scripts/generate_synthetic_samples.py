# -*- coding: utf-8 *-* 
"""Script used for generating synthetic samples

Issues:
    It seems that gTTS only provides one voice. This can be
    a problem if we decide to try other voices later.

"""
import os
from gtts import gTTS
import pickle
import time
import sys

from SimUGANSpeech.definitions import SYNTHETIC_DIR
from SimUGANSpeech.definitions import LIBRISPEECH_DIR 

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
    tts.save(save_path)


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




if __name__ == "__main__":
    # Arg parse...

    # Compile a list of texts
    folder_paths = ['dev-clean']
    transcriptions = get_librispeech_texts(folder_paths)

    generate_speech_from_texts(SYNTHETIC_DIR, transcriptions, False)



