import numpy as np
import tensorflow 
import os
import sys

from tensorflow.examples.tutorials.mnist import input_data

from SimUGANSpeech.models.simple import SimpleNN
from SimUGANSpeech.definitions import TENSORFLOW_DIR

NUM_ITER = 10
BATCH_SIZE = 100

if __name__ == "__main__":
    mnist_save_path = os.path.join(TENSORFLOW_DIR, 'mnist_ex')
    nn_save_path = os.path.join(mnist_save_path, 'model')
    mnist = input_data.read_data_sets(mnist_save_path, one_hot = True)

    input_shape = (None, 784)
    output_shape = (None, 10)

    clf = SimpleNN(input_shape, output_shape, nn_save_path)

    sess = tensorflow.Session()
    sess.run(tensorflow.global_variables_initializer())

    for i in range(NUM_ITER):
        images, labels = mnist.test.images, mnist.test.labels
        error = sess.run(clf.error, {clf.input_tensor: images, clf.output_tensor: labels})

        print ('Test error: {:6.2f}%'.format(100 * error))

        for _ in range(100):
            batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
            
            feed_dict = { clf.input_tensor : batch_xs,
                          clf.output_tensor : batch_ys }

            sess.run(clf.optimize, feed_dict=feed_dict)
