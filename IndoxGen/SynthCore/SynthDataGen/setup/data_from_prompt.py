import pandas as pd
import json
import warnings
from loguru import logger
import sys
import importlib



warnings.filterwarnings("ignore")

# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout,
           format="<green>{level}</green>: <level>{message}</level>",
           level="INFO")

logger.add(sys.stdout,
           format="<red>{level}</red>: <level>{message}</level>",
           level="ERROR")




class DataFromPrompt:
    """
    Generates outputs from a given LLM based on a setup and arguments.

    Attributes:
        prompt_name (str): The name of the prompt used for generation.
        args (dict): Dictionary of arguments for the LLM and generation setup.
        outputs (dict): Dictionary of expected outputs.
        llm: The language model to generate data.
        n (int): Number of generations to be produced.
        dataframe (pd.DataFrame): Optional pre-existing DataFrame for augmentation.
    """

    def __init__(self, prompt_name: str, args: dict, outputs: dict, dataframe: pd.DataFrame = None):
        """
        Initializes the DataFromPrompt class.

        Args:
            prompt_name (str): The name of the prompt for generating data.
            args (dict): Dictionary containing LLM and other arguments.
            outputs (dict): Dictionary specifying expected outputs.
            dataframe (pd.DataFrame, optional): DataFrame for augmenting data. Defaults to None.
        """
        self.prompt_name = prompt_name
        self.args = args
        self.outputs = outputs
        self.llm = args["llm"]
        self.n = args.get("n", 1)
        self.dataframe = dataframe

    def run(self) -> pd.DataFrame:
        """
        Generates data based on the setup and returns a DataFrame.

        If a DataFrame is provided, it uses it as a context to generate new data.
        If no DataFrame is provided, it generates new data from the LLM prompt.

        Returns:
            pd.DataFrame: A DataFrame containing the generated data.

        Raises:
            ValueError: If the LLM's JSON response cannot be parsed or is not valid.
        """
        if self.dataframe is not None:
            sample_row = self.dataframe.to_dict(orient='records')
            user_instruction = self.args["instruction"]

            prompt = f"{user_instruction} {self.prompt_name} : {json.dumps(sample_row)}"

            generations = self.llm.chat(prompt=prompt, max_tokens=8000)

            full_response = ''.join(generations)

            try:
                results = json.loads(full_response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                raise ValueError(f"Failed to parse JSON response: {e}")

            if isinstance(results, dict):
                df_new = pd.json_normalize(results)
                self.dataframe = pd.concat([df_new, self.dataframe], ignore_index=True)
            elif isinstance(results, list):
                df_new = pd.json_normalize(results)
                self.dataframe = pd.concat([df_new, self.dataframe], ignore_index=True)
            else:
                logger.error("Expected a dictionary or a list of dictionaries result from the LLM.")
                raise ValueError("Expected a dictionary or a list of dictionaries result from the LLM.")

            logger.info(f"Generated DataFrame with shape: {self.dataframe.shape}")
        else:
            prompt = self.args["instruction"]

            generations = self.llm.chat(prompt=prompt)

            full_response = ''.join(generations)

            try:
                results = json.loads(full_response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                raise ValueError(f"Failed to parse JSON response: {e}")

            if isinstance(results, dict):
                self.dataframe = pd.DataFrame([results][0])
            else:
                self.dataframe = pd.DataFrame(results)

            logger.info(f"Generated DataFrame with shape: {self.dataframe.shape}")

        return self.dataframe
    def save_to_excel(self, file_path: str) -> None:
        """
        Saves the generated DataFrame to an Excel file.

        Args:
            file_path (str): The path where the Excel file will be saved.

        Raises:
            ValueError: If the DataFrame is empty or cannot be saved.
        """
        if self.dataframe is None:
            logger.error("No DataFrame to save.")
            raise ValueError("No DataFrame to save.")

        if self.dataframe.empty:
            logger.error("DataFrame is empty. Cannot save to Excel.")
            raise ValueError("DataFrame is empty. Cannot save to Excel.")

        try:
            self.dataframe.to_excel(file_path, index=False)
            logger.info(f"DataFrame saved to Excel file at: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save DataFrame to Excel: {e}")
            raise ValueError(f"Failed to save DataFrame to Excel: {e}")
