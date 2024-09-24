import pandas as pd
import json
import warnings
from loguru import logger
import sys
from typing import List, Dict

warnings.filterwarnings("ignore")

# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout,
           format="<green>{level}</green>: <level>{message}</level>",
           level="INFO")

logger.add(sys.stdout,
           format="<red>{level}</red>: <level>{message}</level>",
           level="ERROR")



class FewShotPrompt:
    """Generates outputs from a given LLM based on few-shot examples and a user setup."""

    def __init__(self, prompt_name: str, args: dict, outputs: dict, examples: List[Dict[str, str]]):
        super().__init__()
        self.prompt_name = prompt_name
        self.args = args
        self.outputs = outputs
        self.examples = examples
        self.llm = args["llm"]
        self.n = args.get("n", 1)

    def prepare_prompt(self) -> str:
        """Prepares the full setup including the examples."""
        few_shot_examples = ""
        for example in self.examples:
            input_example = example.get("input", "")
            output_example = example.get("output", "")
            few_shot_examples += f"Input: {input_example}\nOutput: {output_example}\n\n"

        full_prompt = f"{few_shot_examples}Input: {self.args['instruction']}\nOutput:"
        return full_prompt

    def run(self) -> pd.DataFrame:
        """Generates data based on the few-shot setup and returns a DataFrame."""
        full_prompt = self.prepare_prompt()

        try:
            generations = self.llm.chat(prompt=full_prompt, max_tokens=8000)
        except Exception as e:
            raise ValueError(f"Failed to generate text from LLM: {e}")

        if isinstance(generations, list):
            full_response = ''.join(generations)
        else:
            full_response = generations

        try:
            results = json.loads(full_response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}")

        if isinstance(results, dict):
            df = pd.DataFrame([results])
        else:
            df = pd.DataFrame(results)

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

__all__ = ["FewShotPrompt"]
