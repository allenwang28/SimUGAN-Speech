import numpy as np
import os

from src.data.librispeechdata import LibriSpeechBatchGenerator

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LIBRISPEECH_DIR = os.path.join(DATA_DIR, 'LibriSpeech')
SAVE_DIR = os.path.join(BASE_DIR, 'saves')


# Parameters
folder_path = LIBRISPEECH_DIR
folder_names = [ 'dev-clean' ]
features = [ 'spectrogram' ]
feature_sizes = [ None ]

save = SAVE_DIR
verbose = True


lsbg = LibriSpeechBatchGenerator(folder_path,
                                 folder_names,
                                 features=features,
                                 feature_sizes=feature_sizes,
                                 save=save,
                                 verbose=verbose)







