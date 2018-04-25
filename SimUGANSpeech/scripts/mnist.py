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
    # file paths
    mnist_save_path = os.path.join(DATA_DIR, 'mnist_ex')
    model_save_dir = os.path.join(TENSORFLOW_DIR, 'simple')
    model_save_path = os.path.join(model_save_dir, 'model.cpkt')
    mnist = input_data.read_data_sets(mnist_save_path, one_hot = True)

    # specify classifier parameters
    input_shape = (None, 784)
    output_shape = (None, 10)

    # construct classifier
    clf = SimpleNN(input_shape, output_shape, verbose=True)


    # start tensorflow session
    sess = tensorflow.Session()
    sess.run(clf.initial_op())

    # saver used for checkpoints
    saver = tensorflow.train.Saver()


    # If we want to restore, restore if possible
    if RESTORE:
        print ("Restoring session")
        latest = tensorflow.train.latest_checkpoint(model_save_dir)
        if latest:
            saver.restore(sess, latest)
        else:
            print ("Checkpoint not found. Continuing...")
    else:
        print ("Not restoring session")


    # Start training
    for i in range(NUM_ITER):
        # Back up every 5 
        if i % 5 == 0:
            model_backup_save_path = os.path.join(model_save_dir, 'backup-{0}.cpkt'.format(i / 5))
            saver.save(sess, model_backup_save_path)

        # Test/Evaluate
        images, labels = mnist.test.images, mnist.test.labels
        error = sess.run(clf.error, {clf.input_tensor: images, clf.output_tensor: labels})
        print ('Test error: {:6.2f}%'.format(100 * error))

        # Train
        for _ in range(100):
            # Load batch
            batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
            
            # Create map for batch to graph
            feed_dict = { clf.input_tensor : batch_xs,
                          clf.output_tensor : batch_ys }

            # Run a step of optimizer
            sess.run(clf.optimize, feed_dict=feed_dict)

    save_path = saver.save(sess, model_save_path)
    print ("Saved session to {0}".format(save_path))
