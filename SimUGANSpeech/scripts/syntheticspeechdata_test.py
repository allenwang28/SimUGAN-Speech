import os
import matplotlib.pyplot as plt

from SimUGANSpeech.data.syntheticdata import SyntheticSpeechBatchGenerator

if __name__ == "__main__":
    # Parameters
    features = [
                 'spectrogram',
               ]

    feature_sizes = [
                      None, 
                    ]

    batch_size = 1
    verbose = True

    chunk_pct=0.3
    num_iterations = 10

    sbg = SyntheticSpeechBatchGenerator(features,
                                        feature_sizes,
                                        batch_size=batch_size,
                                        chunk_pct=chunk_pct,
                                        verbose=verbose)

    bg = sbg.batch_generator()

    for i in range(num_iterations):
        batch = next(bg)
        assert (len(batch) == batch_size)

        first_sample = batch[0]
        first_spectrogram = first_sample[0]

        print (first_spectrogram.shape)

        plt.imshow(first_spectrogram)
        plt.show()


