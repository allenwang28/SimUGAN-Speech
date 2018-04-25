import numpy as np
import tensorflow 
import os
import sys

from tensorflow.examples.tutorials.mnist import input_data

from SimUGANSpeech.models.simple import SimpleNN
from SimUGANSpeech.definitions import TENSORFLOW_DIR
from SimUGANSpeech.definitions import DATA_DIR

NUM_ITER = 10
BATCH_SIZE = 100
RESTORE = True

if __name__ == "__main__":
    mnist_save_path = os.path.join(DATA_DIR, 'mnist_ex')
    model_save_dir = os.path.join(TENSORFLOW_DIR, 'simple')
    model_save_path = os.path.join(model_save_dir, 'model.cpkt')
    mnist = input_data.read_data_sets(mnist_save_path, one_hot = True)

    input_shape = (None, 784)
    output_shape = (None, 10)

    clf = SimpleNN(input_shape, output_shape, verbose=True)

    sess = tensorflow.Session()
    sess.run(clf.initial_op())

    saver = tensorflow.train.Saver()

    if RESTORE:
        print ("Restoring session")
        latest = tensorflow.train.latest_checkpoint(model_save_dir)
        if latest:
            saver.restore(sess, latest)
        else:
            print ("Checkpoint not found. Continuing...")
    else:
        print ("Not restoring session")


    for i in range(NUM_ITER):
        if i % 5 == 0:
            model_backup_save_path = os.path.join(model_save_dir, 'backup-{0}.cpkt'.format(i / 5))

        images, labels = mnist.test.images, mnist.test.labels
        error = sess.run(clf.error, {clf.input_tensor: images, clf.output_tensor: labels})

        print ('Test error: {:6.2f}%'.format(100 * error))

        for _ in range(100):
            batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
            
            feed_dict = { clf.input_tensor : batch_xs,
                          clf.output_tensor : batch_ys }

            sess.run(clf.optimize, feed_dict=feed_dict)
    
        saver.save(sess, model_backup_save_path)

    save_path = saver.save(sess, model_save_path)
    print ("Saved session to {0}".format(save_path))
