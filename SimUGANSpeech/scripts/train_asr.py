import numpy as np

from SimUGANSpeech.data import SyntheticSpeechBatchGenerator, LibriSpeechBatchGenerator
from SimUGANSpeech.models.deepspeech2 import 


if __name__ == "__main__":
    features = [
                'spectrogram',
                'transcription',
               ]
    
    feature_sizes = [
                      None,
                      None,
                    ]
    
    batch_size = 10
    verbose = True
    num_epochs = 1

    input_shape = (batch_size, )

    lbg = LibriSpeechBatchGenerator(features,
                                    feature_sizes,
                                    batch_size=batch_size,
                                    verbose=verbose)

    bg = lbg.batch_generator()

    ds2 = DeepSpeech2()

    while (lbg.num_epochs < num_epochs):
        spectrograms, transcriptions = next(bg)

