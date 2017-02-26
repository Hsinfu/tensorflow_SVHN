import numpy as np
import tensorflow as tf
from scipy.io import loadmat

from svhn import DataSet
from model import *


# Load dataset
train_mat = loadmat('../../data/train_32x32.mat')
train_mean = np.load('../../data/train_32x32_mean.npy')
train_dataset = DataSet(train_mat, train_mean)

# Training setting
train_batch_size = 100
training_epochs = 50
display_epoch = 1


# Launch the graph
# allow_growth to set the memory growth while use
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:

    # Allocate/Init the variable in GPU
    sess.run(init)

    # Training epoch cycle
    for epoch in range(training_epochs):
        avg_acc, avg_loss = [0., 0.]
        total_batch = int(train_dataset.num_examples / train_batch_size)

        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = train_dataset.next_batch(train_batch_size)

            # Run optimization op (backprop) and loss op (to get loss value)
            _, l, a = sess.run(
                [optimizer, loss, accuracy],
                feed_dict={x: batch_x, y: batch_y})

            # Compute average loss
            avg_loss += l / total_batch
            avg_acc += a / total_batch

        # Display logs
        if epoch % display_epoch == 0:
            print("Training epoch: %04d loss=%.9f acc=%.9f" % ((epoch+1), avg_loss, avg_acc))

    print("Optimization Finished!")
    saver.save(sess, './model.ckpt')
