import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.indoxGen_torch.GAN.config import TabularGANConfig

class Discriminator(nn.Module):
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

    def build_model(self) -> nn.Sequential:
        """
        Builds the discriminator model based on the configuration.

        Returns:
        --------
        nn.Sequential:
            A PyTorch Sequential model representing the discriminator architecture.
        """
        layers = []
        input_dim = self.config.output_dim

        # Input layer
        for units in self.config.discriminator_layers:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.LayerNorm(units))
            layers.append(nn.Dropout(0.3))
            input_dim = units

        # Use a single linear layer for the final output (WGAN-GP uses no activation here)
        layers.append(nn.Linear(input_dim, 1))

        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor, training: bool = False) -> torch.Tensor:
        """
        Forward pass through the discriminator.

        Parameters:
        -----------
        inputs : torch.Tensor
            A batch of input data (either real or generated) to classify.
        training : bool
            Whether the model is in training mode or not.

        Returns:
        --------
        torch.Tensor:
            A batch of predictions (real or fake) for each input sample.
        """
        return self.model(inputs)

    def gradient_penalty(self, real_samples: torch.Tensor, fake_samples: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Calculates the gradient penalty for WGAN-GP.

        Parameters:
        -----------
        real_samples : torch.Tensor
            A batch of real data samples.
        fake_samples : torch.Tensor
            A batch of generated data samples.

        Returns:
        --------
        torch.Tensor:
            The calculated gradient penalty.
        """
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, device=device).expand_as(real_samples)
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples

        interpolated = interpolated.requires_grad_(True)

        # Compute the predictions for interpolated data
        predictions = self(interpolated)

        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=predictions,
            inputs=interpolated,
            grad_outputs=torch.ones(predictions.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        slopes = torch.sqrt(torch.sum(gradients ** 2, dim=1))
        gradient_penalty = torch.mean((slopes - 1.0) ** 2)

        return gradient_penalty
