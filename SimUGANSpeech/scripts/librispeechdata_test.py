import os

from SimUGANSpeech.data.librispeechdata import LibriSpeechBatchGenerator
from SimUGANSpeech.definitions import LIBRISPEECH_DIR

if __name__ == "__main__":

    # Parameters
    folder_dir = LIBRISPEECH_DIR
    folder_names = [ 
                     'dev-clean',
                   ]

    features = [
                 'spectrogram',
               ]

    feature_sizes = [
                      3762, # Pad or truncate to only 2000
                    ]

    batch_size = 5
    verbose = True

    num_iterations = 2

    lsg = LibriSpeechBatchGenerator(folder_dir,
                                    folder_names,
                                    features,
                                    feature_sizes,
                                    batch_size=batch_size,
                                    verbose=verbose)
    print (lsg._all_paths)
