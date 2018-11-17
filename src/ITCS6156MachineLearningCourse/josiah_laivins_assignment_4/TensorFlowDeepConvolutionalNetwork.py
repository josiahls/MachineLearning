import tensorflow as tf
import numpy as np

class TensorFlowDeepConvolutionalNetwork(object):

    def __init__(self, struct: list):
        self.struct = struct

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Reshape(target_shape=(np.sqrt(self.struct[0]), np.sqrt(self.struct[0]), 1),
                                    input_shape=(self.struct[0],)),

            tf.keras.layers.Conv2D(self.struct[1], kernel_size=3, activation='relu', input_shape=(np.sqrt(self.struct[0]),
                                                                                      np.sqrt(self.struct[0]), 1)),

            *tuple([tf.keras.layers.Conv2D(size, kernel_size=3, activation='relu') for size in self.struct[2:-1]]),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.struct[-1], activation='relu'),
            tf.keras.layers.Dense(self.struct[-1], activation='relu'),
            tf.keras.layers.Dense(self.struct[-1], activation='softmax'),
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
