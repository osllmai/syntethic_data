import requests
from loguru import logger
import sys

# Set up logging
logger.remove()
logger.add(sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO")
logger.add(sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR")


class HuggingFaceModel:
    def __init__(self, api_key, model="tiiuae/falcon-7b-instruct", prompt_template=""):
        """
        Initializes the Hugging Face model using the specified model via the Hugging Face Inference API.

        Args:
            api_key (str): The API key for Hugging Face.
            model (str, optional): The model version to use. Defaults to "tiiuae/falcon-7b-instruct".
            prompt_template (str, optional): The template for the prompt. Defaults to an empty string.
        """
        try:
            logger.info(f"Initializing HuggingFaceModel with model: {model}")
            self.api_key = api_key
            self.model = model
            self.prompt_template = prompt_template or "Context: {context}\nQuestion: {question}\nAnswer:"
            if not self.api_key:
                raise ValueError("A valid Hugging Face API key is required.")
            logger.info("HuggingFaceModel initialized successfully")
        except ValueError as ve:
            logger.error(f"ValueError during initialization: {ve}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during initialization: {e}")
            raise

    def _send_request(self, system_prompt, user_prompt, max_tokens, temperature, top_p, frequency_penalty, presence_penalty):
        """
        Sends a request to the Hugging Face API with the given prompt and additional parameters.

        Args:
            system_prompt (str): The system-level instruction for the assistant.
            user_prompt (str): The user's prompt.
            max_tokens (int): The maximum number of tokens in the response.
            temperature (float): Controls the randomness of the output.
            top_p (float): Nucleus sampling (top_p) parameter.
            frequency_penalty (float): Controls penalty for word frequency.
            presence_penalty (float): Controls penalty for new word introduction.

        Returns:
            str: The generated response.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "inputs": f"{system_prompt}\n{user_prompt}",
            "parameters": {
                "max_new_tokens": max_tokens or 8000,  # Reduced default max_tokens to avoid overloading
                "temperature": temperature or 0.7,    # Default temperature if not provided
                "top_p": top_p or 0.9,                # Default top_p if not provided
                "frequency_penalty": frequency_penalty or 0.0,
                "presence_penalty": presence_penalty or 0.0
            }
        }

        try:
            logger.info("Sending request to Hugging Face API")
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{self.model}",
                headers=headers,
                json=payload,
            )

            if response.status_code == 200:
                logger.info("Received successful response from Hugging Face API")
                answer_data = response.json()
                if isinstance(answer_data, list) and len(answer_data) > 0:
                    answer_data = answer_data[0]
                generated_text = answer_data.get("generated_text", "")
                return generated_text.strip()
            else:
                error_message = f"Error from Hugging Face API: {response.status_code}, {response.text}"
                logger.error(error_message)
                raise Exception(error_message)
        except Exception as e:
            logger.error(f"Error in _send_request: {e}")
            raise

    def chat(self, prompt, system_prompt="You are a helpful assistant", max_tokens=8000, temperature=0.7, top_p=0.9, frequency_penalty=0, presence_penalty=0):
        """
        Public method to interact with the model using chat messages, similar to IndoxApi and OpenAi.

        Args:
            prompt (str): The prompt to generate a response for.
            system_prompt (str): The system prompt.
            max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 500.
            temperature (float, optional): Controls the randomness of the output.
            top_p (float, optional): Nucleus sampling (top_p) parameter.
            frequency_penalty (float, optional): Penalty for frequency of words.
            presence_penalty (float, optional): Penalty for introducing new words.

        Returns:
            str: The generated response from the Hugging Face API.
        """
        try:
            logger.info("Generating response from Hugging Face model")
            return self._send_request(system_prompt=system_prompt, user_prompt=prompt,
                                      max_tokens=max_tokens, temperature=temperature,
                                      top_p=top_p, frequency_penalty=frequency_penalty,
                                      presence_penalty=presence_penalty)
        except Exception as e:
            logger.error(f"Error in chat method: {e}")
            return str(e)
