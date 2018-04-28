import numpy as np
import os
import sys

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

from SimUGANSpeech.models.discriminator import Discriminator
from SimUGANSpeech.models.refiner import Refiner
from SimUGANSpeech.definitions import TENSORFLOW_DIR, DATA_DIR, TF_LOGS_DIR

NUM_EPOCHS = 1000
BATCH_SIZE = 100
RESTORE = False 
DISPLAY_RATE = 10
BACKUP_RATE = 100

if __name__ == "__main__":
    # file paths
    mnist_save_path = os.path.join(DATA_DIR, 'mnist_ex')
    model_save_dir = os.path.join(TENSORFLOW_DIR, 'simple')
    model_save_path = os.path.join(model_save_dir, 'model.cpkt')
    logs_path = os.path.join(TF_LOGS_DIR, 'simple')
    mnist = input_data.read_data_sets(mnist_save_path, one_hot = True)

    # specify classifier parameters
    input_shape = (None, 784)
    output_shape = (None, 10)

    # construct classifier
    discrim_clf = Discriminator(input_shape, output_shape, verbose=True)
    refiner_clf = Refiner(input_shape, output_shape, verbose=True)


    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()
    
    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path)

    # start tf session
    sess = tf.Session()
    sess.run(clf.initial_op())

    # saver used for checkpoints
    saver = tf.train.Saver()

    # If we want to restore, restore if possible
    if RESTORE:
        print ("Restoring session")
        latest = tf.train.latest_checkpoint(model_save_dir)
        if latest:
            saver.restore(sess, latest)
        else:
            print ("Checkpoint not found. Continuing...")
    else:
        print ("Not restoring session")


    # Start training
    for epoch in range(NUM_EPOCHS):
        if epoch % BACKUP_RATE == 0:
            backup_num = epoch / BACKUP_RATE
            model_backup_save_path = os.path.join(model_save_dir, 'backup-{0}.cpkt'.format(backup_num))
            saver.save(sess, model_backup_save_path)

        if epoch % DISPLAY_RATE == 0:
            # Test/Evaluate
            images, labels = mnist.test.images, mnist.test.labels
            error = sess.run(clf.error, {clf.input_tensor: images, clf.output_tensor: labels})
            print ('Test error: {:6.2f}%'.format(100 * error))

        num_batches = int(mnist.train.num_examples/BATCH_SIZE)
        # Train
        for i in range(num_batches):
            # Load batch
            batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
            
            # Create map for batch to graph
            feed_dict = { clf.input_tensor : batch_xs,
                          clf.output_tensor : batch_ys }

            # Run optimizer, get cost, and summarize
            _, l, summary = sess.run([clf.optimize, clf.loss, merged_summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary, epoch * num_batches + i)

    save_path = saver.save(sess, model_save_path)
    print ("Saved session to {0}".format(save_path))
