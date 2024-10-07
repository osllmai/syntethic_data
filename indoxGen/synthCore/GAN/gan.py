from typing import List

from tensorflow import keras
import pandas as pd
import numpy as np

from indoxGen.synthCore.GAN.config import TabularGANConfig
from indoxGen.synthCore.GAN.data_transformer import DataTransformer
from indoxGen.synthCore.GAN.tabular_gan import TabularGAN
from indoxGen.synthCore.GAN.utils import GANMonitor

from indoxGen.synthCore.GAN.config import TabularGANConfig



class TabularGANTrainer:
    """
    TabularGANTrainer class for training a Tabular GAN model on a provided dataset.
    This class handles data preprocessing, model creation, training, and result generation.
    """

    def __init__(self, config: TabularGANConfig, categorical_columns: list = None,
                 mixed_columns: dict = None, integer_columns: list = None):
        """
        Initializes the TabularGANTrainer with the necessary configuration and columns.

        Parameters:
        -----------
        config : TabularGANConfig
            Configuration object containing the parameters for the GAN architecture.
        categorical_columns : list, optional
            List of categorical columns to one-hot encode.
        mixed_columns : dict, optional
            Dictionary specifying constraints on mixed columns (e.g., 'positive', 'negative').
        integer_columns : list, optional
            List of integer columns for rounding during inverse transformation.
        """
        self.config = config
        self.categorical_columns = categorical_columns
        self.mixed_columns = mixed_columns
        self.integer_columns = integer_columns
        self.transformer = None
        self.gan = None
        self.history = None

    def prepare_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepares the data by fitting the transformers and transforming the input data.

        Parameters:
        -----------
        data : pd.DataFrame
            The raw tabular data to be processed.

        Returns:
        --------
        np.ndarray:
            Transformed data ready for GAN training.
        """
        self.transformer = DataTransformer(
            categorical_columns=self.categorical_columns,
            mixed_columns=self.mixed_columns,
            integer_columns=self.integer_columns
        )
        self.transformer.fit(data)
        return self.transformer.transform(data)

    def compile_gan(self):
        """
        Initializes and compiles the Tabular GAN model with Adam optimizers.
        """
        self.gan = TabularGAN(self.config)

        # Optimizer setup
        g_optimizer = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
        d_optimizer = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
        self.gan.compile(g_optimizer=g_optimizer, d_optimizer=d_optimizer)

        super(TabularGAN, self.gan).compile(
            loss=None, metrics=None, optimizer=g_optimizer
        )

    def train(self, data: pd.DataFrame, patience: int = 10):
        """
        Trains the Tabular GAN on the provided data with early stopping.

        Parameters:
        -----------
        data : pd.DataFrame
            The raw tabular data to be processed and used for training.
        patience : int, optional
            Number of epochs with no improvement after which training will be stopped.

        Returns:
        --------
        keras.callbacks.History:
            The history object storing training progress.
        """
        transformed_data = self.prepare_data(data)
        self.compile_gan()

        # Custom GAN monitor for early stopping
        gan_monitor = GANMonitor(patience=patience)

        # Train the GAN with early stopping
        self.history = self.gan.fit(
            transformed_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=[gan_monitor],
            verbose=1
        )

        return self.history

    def generate_samples(self, num_samples: int) -> pd.DataFrame:
        """
        Generates new samples using the trained GAN model and converts them back to the original format.

        Parameters:
        -----------
        num_samples : int
            The number of samples to generate.

        Returns:
        --------
        pd.DataFrame:
            Generated synthetic data in its original format.
        """
        if not self.gan or not self.transformer:
            raise ValueError("GAN model is not trained yet. Call `train` method first.")

        generated_data = self.gan.generate(num_samples)
        return self.transformer.inverse_transform(generated_data)

    def get_training_history(self) -> keras.callbacks.History:
        """
        Returns the training history.

        Returns:
        --------
        keras.callbacks.History:
            The history object containing training logs.
        """
        return self.history

    # def evaluate_synthetic_data(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
    #     # Save real and synthetic data
    #     real_data.to_csv("real_data.csv", index=False)
    #     synthetic_data.to_csv("synthetic_data.csv", index=False)
    #
    #     # Get utility metrics
    #     utility_metrics = get_utility_metrics(real_data, synthetic_data)
    #
    #     print("Utility Metrics:\n", utility_metrics)
    #
    #     # For statistical similarity and privacy metrics, call similar functions as needed
    #     # Example:
    #     stat_res = stat_sim("real_data.csv", "synthetic_data.csv", categorical_columns)
    #     print("Statistical Similarity Metrics:\n", stat_res)
    #
    #     privacy_res = privacy_metrics("real_data.csv", "synthetic_data.csv")
    #     print("Privacy Metrics:\n", privacy_res)
    #
    #     return utility_metrics, stat_res, privacy_res
