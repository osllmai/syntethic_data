import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.indoxGen_torch.GAN.config import TabularGANConfig

class Classifier(nn.Module):
    """
    Classifier class for the GAN model, designed to classify generated tabular data into multiple classes.

    Attributes:
    -----------
    config : TabularGANConfig
        Configuration object containing the parameters for the classifier architecture.
    num_classes : int
        The number of output classes for classification.
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

    def build_model(self) -> nn.Module:
        """
        Builds the classifier model based on the configuration and the number of output classes.

        Returns:
        --------
        nn.Module:
            A PyTorch model representing the classifier architecture.
        """
        layers = []
        input_dim = self.config.output_dim

        # Hidden layers based on discriminator layers configuration
        for units in self.config.discriminator_layers:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
            layers.append(nn.Dropout(0.5))
            input_dim = units

        # Output layer with softmax activation for multi-class classification
        layers.append(nn.Linear(input_dim, self.num_classes))

        # Sequential model with all layers
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classifier.

        Parameters:
        -----------
        inputs : torch.Tensor
            A batch of input data to classify.

        Returns:
        --------
        torch.Tensor:
            A batch of class probabilities for each input sample.
        """
        return F.softmax(self.model(inputs), dim=1)
