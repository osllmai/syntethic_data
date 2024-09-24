import pandas as pd
import json
import warnings
from loguru import logger
import sys
from itertools import product
from typing import List
warnings.filterwarnings("ignore")

# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout,
           format="<green>{level}</green>: <level>{message}</level>",
           level="INFO")
logger.add(sys.stdout,
           format="<red>{level}</red>: <level>{message}</level>",
           level="ERROR")

class DataFromAttributedPrompt:
    """Generates outputs from a given LLM based on attributes and a user setup."""

    def __init__(self, prompt_name: str, args: dict, outputs: dict):
        """
        Initializes the data generator with prompt configuration.

        Args:
            prompt_name (str): Name of the prompt.
            args (dict): Arguments for the prompt setup including LLM and instruction.
            outputs (dict): Configuration for the outputs.
        """
        self.prompt_name = prompt_name
        self.args = args
        self.outputs = outputs
        self.llm = args["llm"]
        self.n = args.get("n", 1)
        self.inputs = {}
        self.outputs = {}

    def register_input(self, name: str, help: str):
        """Registers an input for the generator."""
        self.inputs[name] = help

    def register_output(self, name: str, help: str):
        """Registers an output for the generator."""
        self.outputs[name] = help

    def prepare_prompts(self) -> List[str]:
        """
        Prepares multiple prompts based on the specified attributes.

        Returns:
            List[str]: A list of formatted prompts generated from attribute combinations.
        """
        base_instruction = self.args["instruction"]
        attributes = self.args["attributes"]

        attribute_combinations = product(*attributes.values())
        prompts = []

        for combination in attribute_combinations:
            attribute_dict = dict(zip(attributes.keys(), combination))
            formatted_prompt = base_instruction.format(**attribute_dict)
            prompts.append(formatted_prompt)

        logger.info(f"Generated {len(prompts)} prompts from attributes.")
        return prompts

    def run(self) -> pd.DataFrame:
        """Generates data based on the attribute setup and returns a DataFrame."""
        prompts = self.prepare_prompts()
        results = []

        for prompt in prompts:
            logger.info(f"Running prompt: {prompt}")
            try:
                generations = self.llm.chat(prompt=prompt, max_tokens=8000)
            except Exception as e:
                logger.error(f"Failed to generate text from LLM: {e}")
                raise ValueError(f"Failed to generate text from LLM: {e}")

            if isinstance(generations, list):
                full_response = ''.join(generations)
            else:
                full_response = generations

            try:
                result = json.loads(full_response)
                logger.info(f"Generated JSON result: {result}")
            except json.JSONDecodeError:
                result = {"response": full_response}

            results.append(result)

        df = pd.DataFrame(results) if results else pd.DataFrame()
        logger.info("No results generated." if df.empty else f"Generated DataFrame with {len(df)} records.")
        return df

    def save_to_excel(self, file_path: str, df: pd.DataFrame) -> None:
        """
        Saves the generated DataFrame to an Excel file.

        Args:
            file_path (str): The path where the Excel file will be saved.
            df (pd.DataFrame): The DataFrame to be saved.

        Raises:
            ValueError: If the DataFrame is empty or cannot be saved.
        """
        if df.empty:
            logger.error("DataFrame is empty. Cannot save to Excel.")
            raise ValueError("DataFrame is empty. Cannot save to Excel.")

        try:
            df.to_excel(file_path, index=False)
            logger.info(f"DataFrame saved to Excel file at: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save DataFrame to Excel: {e}")
            raise ValueError(f"Failed to save DataFrame to Excel: {e}")


__all__ = ["DataFromAttributedPrompt"]
