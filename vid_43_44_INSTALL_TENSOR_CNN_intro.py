"""
CNN THEORY:

Everyhting startsfromthe concept of a neuron.
It is conposed of dendrits which go to the neuron,
and the axon which outputs to another denarites.
the connection between these is the synapse

Neural network doesnt actually use any of these term tehe
So if the neoron gets a set of inputs, each one has its definition of
what makes it fire.

SO you might have something like:
x1 -> weight1 ->
x2 -> weight2 -> sum(x1,x2,x3) -> f(step function that has a limit to make it fire (1)) -> 0 or 1 and goesto the next neuron
x3 -> weight3 ->

Usually people dont use a step function. They might use a sigmouyed function which is more s shaped
which is an activation function

Y = f(xbar * wbar)

A deep neural network might have:

input   hidden layers
x1      -> n1 ->
        ->    -> n4     -> n6
x2      -> n2 ->        ->
        ->    -> n5     -> n7
x3      -> n3 ->

This is a deep neural network has more than 1
hidden layer.

So why did this take so long to use?????

The optimization problem is harder. You need larger amounts of data.
The typical optimization is by modifying the weights.
Which the number of weights is massively processing intensive.

Note:
    You use ImageNet for images
    Wikipedia for text
    or chat logs
    Tatoba
    CommonCrawl - basically every website that has been parsed

The main point of failure is the data sets.

Note Machine learning reddit can be helpful.

INSTALL TENSOR FLOW

*Note that following:
https://www.tensorflow.org/install/install_sources
Shows you how to install and build tensor flow to compile with your
specifc architecture. Just following the instructions below
installs rebuilt tensflow packages that may not be optimized for
your system.

1. sudo apt-get install python3-pip python3-dev
2. a. pip3 install tensorflow     # Python 3.n; CPU support (no GPU support)
2. b. pip3 install tensorflow-gpu # Python 3.n; GPU support
3. (optional) sudo pip3 install --upgrade tfBinaryURL   # Python 3.n
4. (optional) sudo pip3 uninstall tensorflow # for Python 3.n
5. (optional) sudo apt-get install htop

GOTO:
https://www.tensorflow.org/install/install_linux
for installation help
"""

import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)

# note mul is depreciated
result = tf.multiply(x1, x2)

# note this just creates the tensor
print(result)

# this is to actually run the compile tensor code
sess = tf.Session()
print(sess.run(result))
# closes the object o free ram
sess.close()

# or you can do:
# which will automatically close
with tf.Session() as sess:
    output = sess.run(result)
    print(output)

# The following line will not work
# print(output)


