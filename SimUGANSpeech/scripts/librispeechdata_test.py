import os
import matplotlib.pyplot as plt
import numpy as np

from SimUGANSpeech.data.librispeechdata import LibriSpeechBatchGenerator

if __name__ == "__main__":
    # Parameters
    folder_names = [ 
                     'dev-clean',
                   ]

    features = [
                 'spectrogram',
               ]

    feature_sizes = [
                      None, # Pad or truncate to only 2000
                    ]

    batch_size = 1
    verbose = True

    chunk_pct=0.3
    num_iterations = 10

    lsg = LibriSpeechBatchGenerator(folder_names,
                                    features,
                                    feature_sizes,
                                    batch_size=batch_size,
                                    chunk_pct=chunk_pct,
                                    verbose=verbose)

    bg = lsg.batch_generator()

    for i in range(num_iterations):
        batch = next(bg)
        assert (len(batch) == batch_size)

        first_sample = batch[0]
        first_spectrogram = first_sample[0]

        print (first_spectrogram.shape)

        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,4))
        cax = ax.matshow(np.transpose(first_spectrogram), interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
        fig.colorbar(cax)
        plt.title('Spectrogram')
        plt.show()

