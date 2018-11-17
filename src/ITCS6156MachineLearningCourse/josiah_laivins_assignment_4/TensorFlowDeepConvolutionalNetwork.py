import tensorflow as tf


class TensorFlowDeepConvolutionalNetwork(object):

    def __init__(self, struct: list):
        self.struct = struct

        # Do convolutions in layers
        conv_layer_pairs = []
        for size in struct[1:-1]:
            conv_layer_pairs.append(tf.keras.layers.Conv1D(size, activation=tf.nn.relu, kernel_size=(3)))
            # conv_layer_pairs.append(tf.keras.layers.MaxPool1D(pool_size=(2, 2), strides=(2, 2)))

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Reshape(target_shape=(28, 28, 1), input_shape=(784,)),
            tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax'),
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
