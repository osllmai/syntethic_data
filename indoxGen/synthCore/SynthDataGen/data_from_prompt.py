import pandas as pd
import json
import warnings
from loguru import logger
import sys
from typing import Dict, Any, Optional

warnings.filterwarnings("ignore")

# Set up logging with different levels based on verbose flag
logger.remove()  # Remove the default logger
logger.add(sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO")
logger.add(sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR")


class DataFromPrompt:
    """
    Generates output from a given LLM based on user instructions.

    This class uses a language model to generate data based on user instructions,
    focusing on producing a single, comprehensive response.
    """

    def __init__(
            self,
            llm,
            user_instruction: str,
            example_data: Optional[pd.DataFrame] = None,
            verbose: int = 0
    ):
        """
        Initialize the DataFromPrompt class.

        Args:
            llm: Language model for generating data.
            user_instruction: Instruction for data generation.
            example_data: Optional pre-existing DataFrame for context.
            verbose: Verbosity level (0 for minimal output, 1 for detailed feedback).
        """
        self.llm = llm
        self.user_instruction = user_instruction
        self.dataframe = example_data
        self.verbose = verbose
        self.generated_data = None

        # Adjust logger level based on verbosity
        self._set_logging_level()

    def _set_logging_level(self) -> None:
        """Set the logging level based on verbosity."""
        if self.verbose >= 1:
            logger.enable(__name__)  # Enable detailed logging
            logger.info("Verbose mode activated.")
        else:
            logger.disable(__name__)  # Disable logging except for errors

    @staticmethod
    def get_instruction(user_prompt: str = "") -> str:
        """
        Return a comprehensive instruction for generating data.

        Args:
            user_prompt: User-provided prompt for data generation.

        Returns:
            str: Comprehensive instruction for data generation.
        """
        base_instruction = (
            "You are an advanced AI designed to generate unique and comprehensive data."
            " Your task is to carefully extract relevant information from the provided input"
            " and generate a highly detailed and structured response."
            " Ensure the output is creative, relevant, and entirely original."
            " The response should be in a format that can be parsed as a JSON object."
            " Avoid using common phrases like 'Here is', 'Generated data', or any similar expressions."
            " Only return the structured data without any additional text or explanations."
        )
        return f"{base_instruction} {user_prompt}".strip()

    def generate_data(self) -> pd.DataFrame:
        """
        Generate data based on the user instruction.

        Returns:
            pd.DataFrame: Generated data as a DataFrame.
        """
        generated = self._generate_data_point()
        if generated:
            self.generated_data = generated
            if self.verbose >= 1:
                logger.info(f"Generated data: {json.dumps(generated, indent=2)}")
        else:
            logger.warning("Failed to generate valid data.")

        return self.to_dataframe()

    def _generate_data_point(self) -> Dict[str, Any]:
        """Generate a single data point."""
        system_prompt = ("You are an advanced data generator. Create a comprehensive and realistic response based on "
                         "the given instruction. Your response must be a valid JSON object with all property names "
                         "enclosed in double quotes.")
        prompt = self._create_generation_prompt()

        for attempt in range(3):
            try:
                generated = self.llm.chat(prompt=prompt, system_prompt=system_prompt, max_tokens=8000)
                # Extract the JSON object
                start = generated.find('{')
                end = generated.rfind('}')
                if start != -1 and end != -1 and start < end:
                    json_str = generated[start:end + 1]
                    json_str = json_str.replace("'", '"')  # Ensure valid JSON format
                    data = json.loads(json_str)
                    return data
                else:
                    logger.warning(f"Failed to find valid JSON object in generated text (Attempt {attempt + 1}/3)")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse generated data (Attempt {attempt + 1}/3): {str(e)}")

            if attempt < 2:
                logger.info(f"Retrying generation (Attempt {attempt + 2}/3)...")

        logger.warning("Max attempts reached. Failed to generate valid data.")
        return {}

    def _create_generation_prompt(self) -> str:
        """Create a prompt for the generator LLM."""
        prompt = f"User instruction: {self.user_instruction}\n"
        prompt += ("Generate a comprehensive and structured response based on the instruction. "
                   "The response should be detailed, relevant, and in a format that can be parsed as a JSON object.\n")

        if self.dataframe is not None:
            # Get the columns from the DataFrame
            columns = self.dataframe.columns.tolist()
            sample_row = self.dataframe.to_dict(orient='records')

            # Add context with sample data
            prompt += f"\nContext (existing data sample): {json.dumps(sample_row)}\n"
            prompt += f"Only use the following columns for your response: {', '.join(columns)}.\n"

        prompt += (
            "\nGenerate a single, comprehensive response as a JSON object. Make sure the response contains "
            "only the specified columns and do not add any extra fields. The response should be directly relevant to the user instruction."
        )
        return prompt

    def to_dataframe(self) -> pd.DataFrame:
        """Convert generated data to a pandas DataFrame, handling various data structures."""
        if self.generated_data is None:
            logger.error("No data has been generated yet. Call generate_data() first.")
            return pd.DataFrame()

        # If the generated data is a list of dictionaries, directly convert to a DataFrame
        if isinstance(self.generated_data, list):
            if all(isinstance(item, dict) for item in self.generated_data):
                return pd.DataFrame(self.generated_data)

        # If the generated data is a dictionary, handle possible nested structures
        elif isinstance(self.generated_data, dict):
            # Flatten if the dictionary contains nested lists or dictionaries
            for key, value in self.generated_data.items():
                if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                    # Convert list of dictionaries into a DataFrame
                    return pd.DataFrame(value)
                elif isinstance(value, dict):
                    # Convert the inner dictionary into a DataFrame with one row
                    return pd.DataFrame([value])

            # If it's a simple dictionary, convert it into a single-row DataFrame
            return pd.DataFrame([self.generated_data])

        logger.error(f"Unexpected data type: {type(self.generated_data)}. Cannot convert to DataFrame.")
        return pd.DataFrame()

    def save_to_excel(self, file_path: str) -> None:
        """
        Saves the generated data to an Excel file.

        Args:
            file_path (str): The path where the Excel file will be saved.

        Raises:
            ValueError: If no data has been generated or it cannot be saved.
        """
        df = self.to_dataframe()

        if df.empty:
            logger.error("No data to save. Generate data first.")
            raise ValueError("No data to save. Generate data first.")

        try:
            df.to_excel(file_path, index=False)
            logger.info(f"Data saved to Excel file at: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save data to Excel: {e}")
            raise ValueError(f"Failed to save data to Excel: {e}")
