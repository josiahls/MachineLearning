import tensorflow as tf
import copy
import numpy as np

class TensorFlowDeepResidualNetwork(object):
    """
    Referenced: https://blog.waya.ai/deep-residual-learning-9610bb62c355 for res net implementation

    """

    def __init__(self, struct: list):
        self.struct = struct

        self.model = self.get_residual_model()
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

    def get_residual_model(self):

        cardinality = 32

        def residual_network(x):
            """
            ResNeXt by default. For ResNet set `cardinality` = 1 above.

            """

            def add_common_layers(y):
                y = tf.keras.layers.BatchNormalization()(y)
                y = tf.keras.layers.LeakyReLU()(y)

                return y

            def grouped_convolution(y, nb_channels, _strides):
                # when `cardinality` == 1 this is just a standard convolution
                if cardinality == 1:
                    return tf.keras.layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)

                assert not nb_channels % cardinality
                _d = nb_channels // cardinality

                # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
                # and convolutions are separately performed within each group
                groups = []
                for j in range(cardinality):
                    group = tf.keras.layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
                    groups.append(
                        tf.keras.layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))

                # the grouped convolutional layer concatenates them as the outputs of the layer
                y = tf.keras.layers.concatenate(groups)

                return y

            def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
                """
                Our network consists of a stack of residual blocks. These blocks have the same topology,
                and are subject to two simple rules:
                - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
                - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
                """
                shortcut = y

                # we modify the residual building block as a bottleneck design to make the network more economical
                y = tf.keras.layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
                y = add_common_layers(y)

                # ResNeXt (identical to ResNet when `cardinality` == 1)
                y = grouped_convolution(y, nb_channels_in, _strides=_strides)
                y = add_common_layers(y)

                y = tf.keras.layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
                # batch normalization is employed after aggregating the transformations and before adding to the shortcut
                y = tf.keras.layers.BatchNormalization()(y)

                # identity shortcuts used directly when the input and output are of the same dimensions
                if _project_shortcut or _strides != (1, 1):
                    # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
                    # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
                    shortcut = tf.keras.layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides,
                                                      padding='same')(
                        shortcut)
                    shortcut = tf.keras.layers.BatchNormalization()(shortcut)

                y = tf.keras.layers.add([shortcut, y])

                # relu is performed right after each batch normalization,
                # expect for the output of the block where relu is performed after the adding to the shortcut
                y = tf.keras.layers.LeakyReLU()(y)

                return y

            for size in self.struct[1:-1]:

                init_size = copy.deepcopy(size)
                # conv1
                x = tf.keras.layers.Conv2D(init_size, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
                x = add_common_layers(x)

                init_size = init_size*2
                # conv2
                x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
                for i in range(3):
                    project_shortcut = True if i == 0 else False
                    x = residual_block(x, init_size, init_size*2, _project_shortcut=project_shortcut)

                init_size = init_size*2
                # conv3
                for i in range(4):
                    # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
                    strides = (2, 2) if i == 0 else (1, 1)
                    x = residual_block(x, init_size, init_size*2, _strides=strides)

                init_size = init_size*2
                # conv4
                for i in range(6):
                    strides = (2, 2) if i == 0 else (1, 1)
                    x = residual_block(x, init_size, init_size*2, _strides=strides)

                # conv5
                for i in range(3):
                    strides = (2, 2) if i == 0 else (1, 1)
                    x = residual_block(x, init_size, init_size*2, _strides=strides)

            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(self.struct[-1])(x)

            return x

        image_tensor = tf.keras.layers.Input(shape=(self.struct[0],))
        reshape_tensor = tf.keras.layers.Reshape(target_shape=(np.sqrt(self.struct[0]), np.sqrt(self.struct[0]), 1),
                                                 input_shape=(self.struct[0],))(image_tensor)
        network_output = residual_network(reshape_tensor)

        model = tf.keras.models.Model(inputs=[image_tensor], outputs=[network_output])
        return model
