from tensorflow import keras
import tensorflow as tf

from indoxGen.synthCore.GAN.config import TabularGANConfig


class Discriminator(keras.Model):
    """
    Discriminator class for the GAN model, responsible for classifying real and generated data.

    Attributes:
    -----------
    config : TabularGANConfig
        Configuration object containing the parameters for the discriminator architecture.
    model : keras.Model
        The actual Keras model built using the specified layers in the configuration.
    """

    def __init__(self, config: TabularGANConfig):
        """
        Initializes the discriminator model based on the configuration provided.

        Parameters:
        -----------
        config : TabularGANConfig
            Configuration object containing the parameters for the discriminator architecture.
        """
        super(Discriminator, self).__init__()
        self.config = config
        self.model = self.build_model()

    def build_model(self) -> keras.Model:
        """
        Builds the discriminator model based on the configuration.

        Returns:
        --------
        keras.Model:
            A Keras model representing the discriminator architecture.
        """
        # Define the input layer with the shape of the discriminator's input dimension
        inputs = keras.Input(shape=(self.config.output_dim,))

        # First dense layer with LeakyReLU and dropout for regularization
        x = keras.layers.Dense(self.config.discriminator_layers[0],
                               kernel_initializer='he_normal')(inputs)
        x = keras.layers.LeakyReLU(negative_slope=0.2)(x)
        x = keras.layers.Dropout(0.3)(x)

        # Additional hidden layers with LeakyReLU and dropout
        for units in self.config.discriminator_layers[1:]:
            x = keras.layers.Dense(units, kernel_initializer='he_normal')(x)
            x = keras.layers.LeakyReLU(negative_slope=0.2)(x)
            x = keras.layers.Dropout(0.3)(x)

        # Output layer with a single unit for binary classification (real or fake)
        outputs = keras.layers.Dense(1, activation='sigmoid')(x)

        # Create and return the Keras model
        return keras.Model(inputs=inputs, outputs=outputs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass through the discriminator.

        Parameters:
        -----------
        inputs : tf.Tensor
            A batch of input data (either real or generated) to classify.

        Returns:
        --------
        tf.Tensor:
            A batch of predictions (real or fake) for each input sample.
        """
        return self.model(inputs)
