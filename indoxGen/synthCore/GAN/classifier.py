from tensorflow import keras
import tensorflow as tf
from indoxGen.synthCore.GAN.config import TabularGANConfig


class Classifier(keras.Model):

    """
    Classifier class for the GAN model, designed to classify generated tabular data into multiple classes.

    Attributes:
    -----------
    config : TabularGANConfig
        Configuration object containing the parameters for the classifier architecture.
    num_classes : int
        The number of output classes for classification.
    model : keras.Model
        The actual Keras model built using the specified layers in the configuration.
    """

    def __init__(self, config: TabularGANConfig, num_classes: int):
        """
        Initializes the classifier model based on the configuration and the number of output classes.

        Parameters:
        -----------
        config : TabularGANConfig
            Configuration object containing the parameters for the classifier architecture.
        num_classes : int
            The number of classes for the classification task.
        """
        super(Classifier, self).__init__()
        self.config = config
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self) -> keras.Model:
        """
        Builds the classifier model based on the configuration and the number of output classes.

        Returns:
        --------
        keras.Model:
            A Keras model representing the classifier architecture.
        """
        # Input layer
        inputs = keras.Input(shape=(self.config.output_dim,))

        # Hidden layers based on discriminator layers configuration
        x = inputs
        for units in self.config.discriminator_layers:
            x = keras.layers.Dense(units, kernel_initializer='he_normal')(x)
            x = keras.layers.LeakyReLU(negative_slope=0.2)(x)
            x = keras.layers.Dropout(0.5)(x)

        # Output layer with softmax activation for multi-class classification
        outputs = keras.layers.Dense(self.num_classes, activation='softmax')(x)

        # Create and return the model
        return keras.Model(inputs=inputs, outputs=outputs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass through the classifier.

        Parameters:
        -----------
        inputs : tf.Tensor
            A batch of input data to classify.

        Returns:
        --------
        tf.Tensor:
            A batch of class probabilities for each input sample.
        """
        return self.model(inputs)
