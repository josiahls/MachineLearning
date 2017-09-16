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







"""