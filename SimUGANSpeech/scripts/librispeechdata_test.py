import os
import matplotlib.pyplot as plt
import numpy as np

from SimUGANSpeech.data.librispeechdata import LibriSpeechBatchGenerator
from SimUGANSpeech.preprocessing.audio import AudioParams

if __name__ == "__main__":
    # Parameters
    folder_names = [ 
                     'dev-clean',
                   ]

    features = [
                 'mfcc',
                 'transcription'
               ]

    feature_sizes = [
                      200, 
                      100,
                    ]

    batch_size = 2
    verbose = True

    chunk_pct=None
    num_iterations = 3

    audio_params = AudioParams()
    audio_params.max_time_in_s = 3

    lsg = LibriSpeechBatchGenerator(folder_names,
                                    [],
                                    features,
                                    feature_sizes,
                                    batch_size=batch_size,
                                    chunk_pct=chunk_pct,
                                    verbose=verbose)

    for i in range(num_iterations):
        batch = lsg.get_training_batch()
        assert (len(batch) == len(features))
        assert (len(batch[0]) == batch_size)

        spectrograms = batch[0]
        transcriptions = batch[1]
        print (transcriptions)

        first_spectrogram = spectrograms[0]

        print (first_spectrogram.shape)

        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,4))
        cax = ax.matshow(np.transpose(first_spectrogram), interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
        fig.colorbar(cax)
        plt.title('Spectrogram')
        plt.show()

