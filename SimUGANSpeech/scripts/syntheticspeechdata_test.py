import os
import matplotlib.pyplot as plt
import numpy as np

from SimUGANSpeech.data.syntheticdata import SyntheticSpeechBatchGenerator

if __name__ == "__main__":
    # Parameters
    features = [
                 'spectrogram',
               ]

    feature_sizes = [
                      100, 
                    ]

    batch_size = 1
    verbose = True

    chunk_pct=0.3
    num_iterations = 3

    sbg = SyntheticSpeechBatchGenerator(features,
                                        feature_sizes,
                                        batch_size=batch_size,
                                        chunk_pct=chunk_pct,
                                        verbose=verbose)

    for i in range(num_iterations):
        batch = sbg.get_training_batch()
        assert (len(batch) == len(features))
        assert (len(batch[0]) == batch_size)

        spectrograms = batch[0]
        first_spectrogram = spectrograms[0]

        print (first_spectrogram.shape)

        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,4))
        cax = ax.matshow(np.transpose(first_spectrogram), interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
        fig.colorbar(cax)
        plt.title('Spectrogram')
        plt.show()

