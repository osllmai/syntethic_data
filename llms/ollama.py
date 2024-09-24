from tenacity import retry, stop_after_attempt, wait_random_exponential
from loguru import logger
import sys
import ollama as ol

# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout,
           format="<green>{level}</green>: <level>{message}</level>",
           level="INFO")

logger.add(sys.stdout,
           format="<red>{level}</red>: <level>{message}</level>",
           level="ERROR")


class Ollama:
    def __init__(self, model):
        """
        Initializes the Ollama model with the specified model version.

        Args:
            model (str): Ollama model version.
        """
        try:
            logger.info(f"Initializing Ollama with model: {model}")
            self.model = model
            logger.info("Ollama initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Ollama: {e}")
            raise

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _generate_response(self, prompt):
        """
        Generates a response from the Ollama model.

        Args:
            prompt (str): The prompt to generate a response for.

        Returns:
            str: The generated response.
        """
        try:
            logger.info("Generating response")
            response = ol.generate(model=self.model, prompt=prompt)
            result = response["response"].strip().replace("\n", "").replace("\t", "")
            logger.info("Response generated successfully")
            return result
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def chat(self, prompt, system_prompt=""):
        """
        A method to send a combined system and user prompt to the Ollama model for generating responses.

        Args:
            prompt (str): The user prompt to be sent to the model.
            system_prompt (str, optional): An optional system prompt for guiding the model. Defaults to "".

        Returns:
            str: The generated response from the model.
        """
        combined_prompt = system_prompt + "\n" + prompt
        try:
            logger.info(f"Sending prompt to Ollama: {combined_prompt[:100]}...")  # Log truncated prompt
            return self._generate_response(combined_prompt)
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return str(e)
