from tensorflow import keras
import tensorflow as tf

from indoxGen.synthCore.GAN.config import TabularGANConfig


class Generator(keras.Model):
    """
    Generator class for the GAN model, which takes in random noise and generates synthetic tabular data.

    Attributes:
    -----------
    config : TabularGANConfig
        Configuration object containing the parameters for the generator architecture.
    model : keras.Model
        The actual Keras model built using the specified layers in the configuration.
    """

    def __init__(self, config: TabularGANConfig):
        """
        Initializes the generator model based on the configuration provided.

        Parameters:
        -----------
        config : TabularGANConfig
            Configuration object containing the parameters for the generator architecture.
        """
        super(Generator, self).__init__()
        self.config = config
        self.model = self.build_model()

    def build_model(self) -> keras.Model:
        """
        Builds the generator model based on the configuration.

        Returns:
        --------
        keras.Model:
            A Keras model representing the generator architecture.
        """
        # Input layer
        inputs = keras.Input(shape=(self.config.input_dim,))

        # First dense layer
        x = keras.layers.Dense(self.config.generator_layers[0],
                               kernel_initializer='he_normal')(inputs)
        x = keras.layers.LeakyReLU(negative_slope=0.2)(x)
        x = keras.layers.BatchNormalization()(x)

        # Additional hidden layers
        for units in self.config.generator_layers[1:]:
            x = keras.layers.Dense(units, kernel_initializer='he_normal')(x)
            x = keras.layers.LeakyReLU(negative_slope=0.2)(x)
            x = keras.layers.BatchNormalization()(x)

        # Output layer with 'tanh' activation
        outputs = keras.layers.Dense(self.config.output_dim, activation='tanh')(x)

        return keras.Model(inputs=inputs, outputs=outputs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass through the generator.

        Parameters:
        -----------
        inputs : tf.Tensor
            A batch of input noise vectors to generate synthetic data from.

        Returns:
        --------
        tf.Tensor:
            A batch of generated data.
        """
        return self.model(inputs)
