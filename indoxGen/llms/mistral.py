from tenacity import retry, stop_after_attempt, wait_random_exponential
from loguru import logger
import sys

# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout,
           format="<green>{level}</green>: <level>{message}</level>",
           level="INFO")

logger.add(sys.stdout,
           format="<red>{level}</red>: <level>{message}</level>",
           level="ERROR")


class Mistral:
    def __init__(self, api_key, model="mistral-medium-latest"):
        """
        Initializes the Mistral AI model with the specified model version.

        Args:
            api_key (str): The API key for Mistral AI.
            model (str): The Mistral AI model version. Defaults to "mistral-medium-latest".
        """
        from mistralai import Mistral  # Assuming this is the Mistral API module

        try:
            logger.info(f"Initializing MistralAI with model: {model}")
            self.model = model
            self.client = Mistral(api_key=api_key)
            logger.info("MistralAI initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing MistralAI: {e}")
            raise

    def _run_mistral(self, user_message):
        """
        Runs the Mistral model to generate a response based on the user message.

        Args:
            user_message (str): The message to be processed by the Mistral model.

        Returns:
            str: The generated response.
        """
        try:
            messages = [
                {
                    "role": "user",
                    "content": user_message,
                },
            ]

            chat_response = self.client.chat.complete(
                model=self.model,
                messages=messages,
            )

            return chat_response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in run_mistral: {e}")
            return str(e)

    def chat(self, prompt, system_prompt=""):
        """
        A method to send a combined system and user prompt to the Mistral API.

        Args:
            prompt (str): The user prompt to be sent to the model.
            system_prompt (str, optional): An optional system prompt for guiding the model. Defaults to "".

        Returns:
            str: The generated response from the model.
        """
        combined_prompt = system_prompt + "\n" + prompt
        try:
            logger.info(f"Sending prompt to Mistral: {combined_prompt[:100]}...")  # Log truncated prompt
            response = self._run_mistral(combined_prompt)
            return response
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return str(e)
