import os

from SimUGANSpeech.data.librispeechdata import LibriSpeechBatchGenerator

if __name__ == "__main__":
    # Parameters
    folder_names = [ 
                     'dev-other',
                   ]

    features = [
                 'spectrogram',
               ]

    feature_sizes = [
                      3762, # Pad or truncate to only 2000
                    ]

    batch_size = 1
    verbose = True

    chunk_pct=0.3
    num_iterations = 1

    lsg = LibriSpeechBatchGenerator(folder_names,
                                    features,
                                    feature_sizes,
                                    batch_size=batch_size,
                                    chunk_pct=chunk_pct,
                                    verbose=verbose)

    bg = lsg.batch_generator()

    for i in range(num_iterations):
        spectrograms = next(bg)
        print (spectrograms[0][0][0].shape)

