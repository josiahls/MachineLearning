"""
Convolution is just creating a feature map of a region.

It usually goes: conv -> hidden layer -> pool -> repeat -> output

convolution just runs a windows on an image based on certain parameters
this is where features get highlighted

pooling takes the result and -> might find the max value for each window

conv + pool = HL

full connected layer: neurons fully connected to each other




"""


import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# one_hot means that one is on and the rest are off
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 10 classes, 0-9
'''
0=0
1=1
2=2
Well one_hot goes and makes
0 = [1,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0]
3 = [0,0,0,1,0,0,0,0]
...
'''

n_classes = 10
# so this creates batches of 100 images
batch_size = 128

# height x width
# we flatten the picture out into
# 784 pixels long
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(X,W):
    return tf.nn.conv2d(X,W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(X):
    #                        size of window   movement of window
    return tf.nn.max_pool(X, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')


def convolutional_neural_network_model(X):
    # creates an arrayof weights in one giant tensor
    # biases are somthing added in after the weights
    # so
    # input_data * weights + biases
    weights = {'W_conv1': tf.Variable(tf.random_normal([5,5,1,32])),
                      'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
                      'W_fc': tf.Variable(tf.random_normal([7*7*64,1024])),
                      'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'B_conv1': tf.Variable(tf.random_normal([32])),
                      'B_conv2': tf.Variable(tf.random_normal([64])),
                      'B_fc': tf.Variable(tf.random_normal([1024])),
                      'out': tf.Variable(tf.random_normal([n_classes]))}

    X = tf.reshape(X, shape=[-1,28,28,1])

    conv1 = tf.nn.relu(conv2d(X, weights['W_conv1']) + biases['B_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['B_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1,7*7*64])
    fc = tf.nn.relu(tf.matmul(fc,weights['W_fc'])+biases['B_fc'])

    fc = tf.nn.dropout(fc,keep_rate)

    output = tf.matmul(fc,weights['out'])+biases['out']

    return output


def train_neural_network(X):
    prediction = convolutional_neural_network_model(X)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    # we want to minimize the cost
    # has a param is the learning rate that is 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # cycles of feed forward and + backprop
    num_epochs = 3
    with tf.Session() as sess:
        # the init all var initializes as the variables
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train_sarsa.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train_sarsa.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, ' completed out of ', num_epochs, ' Loss: ', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({X: mnist.test.images, y: mnist.test.labels}))


train_neural_network(x)
