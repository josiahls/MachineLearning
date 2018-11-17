import tensorflow as tf


class TensorFlowDeepForwardFeedNetwork(object):

    def __init__(self, struct: list):
        self.struct = struct

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(struct[0], activation=tf.nn.relu),
            *tuple([tf.keras.layers.Dense(size, activation=tf.nn.relu) for size in struct[1:-1]]),
            tf.keras.layers.Dense(struct[-1], activation=tf.nn.softmax)
        ])
        self.model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.loss = None
        self.accuracy = None

    def train(self, X, Y, params):
        """ Init hyper params """
        epochs = params.pop('epochs', 5)

        """ Train Model """
        history = self.model.fit(X, Y, epochs=epochs)
        self.loss = history.history['loss']
        self.accuracy = history.history['acc']

    def use(self, X):
        return self.model.predict(X)