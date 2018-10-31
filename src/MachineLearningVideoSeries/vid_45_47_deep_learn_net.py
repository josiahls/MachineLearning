"""
Note each pixel will be a feature
Remmber earlier when we had features x1,x2,x3

input > weight > hidden layer 1 (activation function) > weights > hidden layer 2
(activation function) > weights > output layer

compare the output with the intended output > cost or loss function (cross entropy) (how far off we are)
optimization function (optimizer) > minimize cost (AdamOptimizer...SGD... AdaGrad) (goes back through the network to
manipulate the weights) ie backpropagation

backpropagation

feedforward + backprop = epoch
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

# number of hidden layers
n_nodeshl1 = 500
n_nodeshl2 = 500
n_nodeshl3 = 500

n_classes = 10
# so this creates batches of 100 images
batch_size = 100

# height x width
# we flatten the picture out into
# 784 pixels long
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


def neural_network_model(data):
    # creates an arrayof weights in one giant tensor
    # biases are somthing added in after the weights
    # so
    # input_data * weights + biases
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodeshl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodeshl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodeshl1, n_nodeshl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodeshl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodeshl2, n_nodeshl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodeshl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodeshl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    # relu is the activation function
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    # relu is the activation function
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    # relu is the activation function
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

    return output


def train_neural_network(X):
    prediction = neural_network_model(X)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    # we want to minimize the cost
    # has a param is the learning rate that is 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # cycles of feed forward and + backprop
    num_epochs = 10
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

        correct = tf.equal(tf.argmax(prediction, 1), tf.arg_max(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({X: mnist.test.images, y: mnist.test.labels}))


train_neural_network(x)
