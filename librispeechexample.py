import numpy as np
import os

from src.data.librispeechdata import LibriSpeechBatchGenerator
import src.data.audio as audio

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LIBRISPEECH_DIR = os.path.join(DATA_DIR, 'LibriSpeech')
SAVE_DIR = os.path.join(BASE_DIR, 'saves')


# Parameters
folder_path = LIBRISPEECH_DIR
folder_names = [ 'dev-clean' ]

features = [ 'spectrogram' ]


audio_max_length_in_s = 5
audio_max_length = int(audio_max_length_in_s / audio.DEFAULT_FRAME_SIZE_MS)
feature_sizes = [ audio_max_length ]

save = SAVE_DIR
verbose = True

if __name__ == '__main__':

    lsbg = LibriSpeechBatchGenerator(folder_path,
                                     folder_names,
                                     features=features,
                                     feature_sizes=feature_sizes,
                                     save=save,
                                     verbose=verbose)







