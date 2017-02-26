import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from ..svhn import DataSet
from model import *


# Load dataset
test_mat = loadmat('../../data/test_32x32.mat')
train_mean = np.load('../../data/train_32x32_mean.npy')
test_dataset = DataSet(test_mat, train_mean)

# Testing setting
test_batch_size = 10

# Launch the graph
# allow_growth to set the memory growth while use
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:

    # Allocate/Init the variable in GPU and load model parameters
    sess.run(init)
    saver.restore(sess, './model.ckpt')

    avg_acc, avg_loss, answers = [0., 0., []]
    total_batch = int(test_dataset.num_examples / test_batch_size)

    for i in range(total_batch):
        batch_x, batch_y = test_dataset.next_batch(test_batch_size)

        # Session run on GPU/CPU to get the ans, loss, acc
        ans, lo, acc = sess.run(
            [pred_class, loss, accuracy],
            feed_dict={x: batch_x, y: batch_y, keep_prob1: 1, keep_prob2: 1})

        # Store the ans and compute average loss and acc
        answers.extend(ans)
        avg_loss += lo / total_batch
        avg_acc += acc / total_batch

    print("Testing Num: %d Loss: %.9f Accuracy: %.9f" % (len(answers), avg_loss, avg_acc))

    # Write to labels.txt
    answers = [10 if ans == 0 else ans for ans in answers]
    with open('labels.txt', 'w') as fptr:
        for ans in answers:
            fptr.write('%d\n' % ans)
