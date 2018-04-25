import numpy as np
import os
import tensorflow as tf

from SimUGANSpeech.data import SyntheticSpeechBatchGenerator, LibriSpeechBatchGenerator
from SimUGANSpeech.definitions import TENSORFLOW_DIR
from SimUGANSpeech.models.deepspeech2 import DeepSpeech2
from SimUGANSpeech.util.data_util import text_to_indices


if __name__ == "__main__":
    folder_names = [
                     'dev-clean',
                   ]

    features = [
                'spectrogram',
                'transcription',
               ]
    
    feature_sizes = [
                      1200,
                      100,
                    ]
    
    batch_size = 10
    verbose = True
    num_epochs = 1
    chunk_pct = 0.2

    input_shape = (batch_size, feature_sizes[0], 200)
    output_shape = (batch_size, feature_sizes[1], 26)

    lbg = LibriSpeechBatchGenerator(folder_names,
                                    features,
                                    feature_sizes,
                                    batch_size=batch_size,
                                    chunk_pct=chunk_pct,
                                    verbose=verbose)

    bg = lbg.batch_generator()

    
    model_save_path = os.path.join(TENSORFLOW_DIR, 'deepspeech2')
    ds2 = DeepSpeech2(input_shape, output_shape, model_save_path)

    ds2.build_graph()

    with tf.Session(graph=ds2.graph) as sess:
        while (lbg.epoch < num_epochs):
            spectrograms, transcriptions = next(bg)
            t_idx = [text_to_indices(trans) for trans in transcriptions]
            transcriptions_one_hot = tf.one_hot(t_idx, 26, dtype=tf.uint8)
 
            feed_dict = {ds2.input_tensor : spectrograms,
                         ds2.output_tensor : transcriptions_one_hot}
            
            _, l, pred, y = sess.run([ds2.optimizer, ds2.loss, ds2.predictions, ds2.output_tensor])
            print (ds2.loss)
