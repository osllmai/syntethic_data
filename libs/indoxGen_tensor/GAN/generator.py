import tensorflow as tf
from libs.indoxGen_tensor.GAN.config import TabularGANConfig


class Generator(tf.keras.Model):
    """
    Generator class for the GAN model, which takes in random noise and generates synthetic tabular data.

    Attributes:
    -----------
    config : TabularGANConfig
        Configuration object containing the parameters for the generator architecture.
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

    def build_model(self):
        """
        Builds the generator model based on the configuration.

        Returns:
        --------
        tf.keras.Sequential:
            A Keras Sequential model representing the generator architecture.
        """
        model = tf.keras.Sequential()

        # Input layer
        model.add(tf.keras.layers.Input(shape=(self.config.input_dim,)))

        for i, units in enumerate(self.config.generator_layers):
            model.add(tf.keras.layers.Dense(units, kernel_initializer='he_normal'))
            model.add(tf.keras.layers.LeakyReLU(negative_slope=0.2))
            model.add(tf.keras.layers.BatchNormalization())

            # Add residual connections for deeper networks
            if i > 0 and i % 2 == 0 and units == self.config.generator_layers[i - 2]:
                model.add(tf.keras.layers.Add())

        # Final layer to match the output dimension
        model.add(tf.keras.layers.Dense(self.config.output_dim, activation='tanh'))

        return model

    def call(self, inputs, training=False):
        """
        Forward pass through the generator.

        Parameters:
        -----------
        inputs : tf.Tensor
            A batch of input noise vectors to generate synthetic data from.
        training : bool
            Whether the model is in training mode or not.

        Returns:
        --------
        tf.Tensor:
            A batch of generated data.
        """
        return self.model(inputs, training=training)

    @tf.function
    def generate(self, num_samples: int):
        """
        Generates a specified number of synthetic samples.

        Parameters:
        -----------
        num_samples : int
            The number of synthetic samples to generate.

        Returns:
        --------
        tf.Tensor:
            A tensor of generated synthetic samples.
        """
        noise = tf.random.normal([num_samples, self.config.input_dim])
        return self(noise, training=False)