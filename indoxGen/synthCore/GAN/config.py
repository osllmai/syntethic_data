from typing import List


class TabularGANConfig:
    """
    Configuration class for setting up the parameters of a Tabular GAN model.

    Attributes:
    -----------
    input_dim : int
        The dimension of the input data (number of features).
    output_dim : int
        The dimension of the output data (number of features).
    generator_layers : List[int]
        A list of integers representing the number of neurons in each layer of the generator.
    discriminator_layers : List[int]
        A list of integers representing the number of neurons in each layer of the discriminator.
    learning_rate : float
        The learning rate for both the generator and discriminator.
    beta_1 : float
        The exponential decay rate for the first moment estimates (used in Adam optimizer).
    beta_2 : float
        The exponential decay rate for the second moment estimates (used in Adam optimizer).
    batch_size : int
        The number of samples per gradient update.
    epochs : int
        Number of epochs to train the model.
    n_critic : int
        The number of updates for the discriminator per update of the generator.
    """

    def __init__(self, input_dim: int, output_dim: int, generator_layers: List[int],
                 discriminator_layers: List[int], learning_rate: float, beta_1: float,
                 beta_2: float, batch_size: int, epochs: int, n_critic: int):
        """
        Initializes the GAN configuration with the necessary hyperparameters.

        Parameters:
        -----------
        input_dim : int
            Dimension of the input features.
        output_dim : int
            Dimension of the generated output.
        generator_layers : List[int]
            Layer sizes for the generator.
        discriminator_layers : List[int]
            Layer sizes for the discriminator.
        learning_rate : float
            Learning rate for the optimizer.
        beta_1 : float
            Beta 1 for Adam optimizer.
        beta_2 : float
            Beta 2 for Adam optimizer.
        batch_size : int
            Batch size used for training.
        epochs : int
            Number of training epochs.
        n_critic : int
            Number of discriminator updates per generator update (for WGAN-GP).
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.generator_layers = generator_layers
        self.discriminator_layers = discriminator_layers
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_critic = n_critic

        self._validate_config()

    def _validate_config(self):
        """
        Validates the configuration parameters to ensure they are valid for GAN training.
        Raises:
        -------
        ValueError:
            If any of the critical configurations such as input_dim, output_dim, or layer lists are invalid.
        """
        if not isinstance(self.input_dim, int) or self.input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")
        if not isinstance(self.output_dim, int) or self.output_dim <= 0:
            raise ValueError("output_dim must be a positive integer.")
        if not all(isinstance(layer, int) and layer > 0 for layer in self.generator_layers):
            raise ValueError("All generator layers must be positive integers.")
        if not all(isinstance(layer, int) and layer > 0 for layer in self.discriminator_layers):
            raise ValueError("All discriminator layers must be positive integers.")
        if self.learning_rate <= 0 or self.beta_1 <= 0 or self.beta_2 <= 0:
            raise ValueError("learning_rate, beta_1, and beta_2 must be positive.")
