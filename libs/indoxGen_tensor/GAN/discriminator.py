import tensorflow as tf
from libs.indoxGen_tensor.GAN.config import TabularGANConfig


class Discriminator(tf.keras.Model):
    """
    Discriminator class for the GAN model, responsible for classifying real and generated data.

    Attributes:
    -----------
    config : TabularGANConfig
        Configuration object containing the parameters for the discriminator architecture.
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

    def build_model(self):
        """
        Builds the discriminator model based on the configuration.

        Returns:
        --------
        tf.keras.Sequential:
            A Keras Sequential model representing the discriminator architecture.
        """
        model = tf.keras.Sequential()

        # Input layer
        model.add(tf.keras.layers.Input(shape=(self.config.output_dim,)))

        for units in self.config.discriminator_layers:
            model.add(tf.keras.layers.Dense(units, kernel_initializer='he_normal'))
            model.add(tf.keras.layers.LeakyReLU(negative_slope=0.2))
            model.add(tf.keras.layers.LayerNormalization())
            model.add(tf.keras.layers.Dropout(0.3))

        # Use linear activation for WGAN-GP
        model.add(tf.keras.layers.Dense(1))

        return model

    def call(self, inputs, training=False):
        """
        Forward pass through the discriminator.

        Parameters:
        -----------
        inputs : tf.Tensor
            A batch of input data (either real or generated) to classify.
        training : bool
            Whether the model is in training mode or not.

        Returns:
        --------
        tf.Tensor:
            A batch of predictions (real or fake) for each input sample.
        """
        return self.model(inputs, training=training)

    def gradient_penalty(self, real_samples, fake_samples):
        """
        Calculates the gradient penalty for WGAN-GP.

        Parameters:
        -----------
        real_samples : tf.Tensor
            A batch of real data samples.
        fake_samples : tf.Tensor
            A batch of generated data samples.

        Returns:
        --------
        tf.Tensor:
            The calculated gradient penalty.
        """
        batch_size = tf.shape(real_samples)[0]
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            predictions = self(interpolated, training=True)

        gradients = tape.gradient(predictions, interpolated)
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)

        return gradient_penalty
