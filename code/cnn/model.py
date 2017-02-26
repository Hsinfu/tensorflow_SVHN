import tensorflow as tf

# Network Parameters
learning_rate = 0.0002
n_input = 3072
n_classes = 10

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
# dropout (keep probability)
keep_prob1 = tf.placeholder(tf.float32)
keep_prob2 = tf.placeholder(tf.float32)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# Create model
def conv_net(x, weights, biases, keep_prob1, keep_prob2):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 32, 32, 3])

    # Convolution Layers
    conv1_1 = conv2d(x, weights['wc1_1'], biases['bc1_1'])
    conv1_2 = conv2d(conv1_1, weights['wc1_2'], biases['bc1_2'])
    conv1_3 = conv2d(conv1_2, weights['wc1_3'], biases['bc1_3']),
    pool1 = maxpool2d(conv1_3)

    conv2_1 = conv2d(pool1, weights['wc2_1'], biases['bc2_1'])
    conv2_2 = conv2d(conv2_1, weights['wc2_2'], biases['bc2_2'])
    conv2_3 = conv2d(conv2_2, weights['wc2_3'], biases['bc2_3'])
    pool2 = maxpool2d(conv2_3)

    conv3_1 = conv2d(pool2, weights['wc3_1'], biases['bc3_1'])
    conv3_2 = conv2d(conv3_1, weights['wc3_2'], biases['bc3_2'])
    conv3_3 = conv2d(conv3_2, weights['wc3_3'], biases['bc3_3'])
    pool3 = maxpool2d(conv3_3)

    # Fully connected layer
    # Reshape conv output to fit fully connected layer input
    fc1 = tf.reshape(pool3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob1)

    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob2)

    # Output, class prediction
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out


# Store layers weight & bias
weights = {
    'wc1_1': tf.Variable(tf.truncated_normal([3, 3, 3, 32], stddev=0.1)),
    'wc1_2': tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.1)),
    'wc1_3': tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.1)),
    'wc2_1': tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1)),
    'wc2_2': tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1)),
    'wc2_3': tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1)),
    'wc3_1': tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1)),
    'wc3_2': tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1)),
    'wc3_3': tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1)),
    'wd1': tf.Variable(tf.truncated_normal([4 * 4 * 128, 512], stddev=0.1)),
    'wd2': tf.Variable(tf.truncated_normal([512, 128], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([128, n_classes], stddev=0.1))
}

biases = {
    'bc1_1': tf.Variable(tf.constant(0.1, shape=[32])),
    'bc1_2': tf.Variable(tf.constant(0.1, shape=[32])),
    'bc1_3': tf.Variable(tf.constant(0.1, shape=[32])),
    'bc2_1': tf.Variable(tf.constant(0.1, shape=[64])),
    'bc2_2': tf.Variable(tf.constant(0.1, shape=[64])),
    'bc2_3': tf.Variable(tf.constant(0.1, shape=[64])),
    'bc3_1': tf.Variable(tf.constant(0.1, shape=[128])),
    'bc3_2': tf.Variable(tf.constant(0.1, shape=[128])),
    'bc3_3': tf.Variable(tf.constant(0.1, shape=[128])),
    'bd1': tf.Variable(tf.constant(0.1, shape=[512])),
    'bd2': tf.Variable(tf.constant(0.1, shape=[128])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob1, keep_prob2)
pred_class = tf.argmax(pred, 1)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Add ops to save and restore all the variables.
saver = tf.train.Saver(sharded=False)
