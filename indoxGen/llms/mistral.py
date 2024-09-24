import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential
from loguru import logger
import sys

# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO")
logger.add(sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR")


class Mistral:
    def __init__(self, api_key, model="mistral-medium-latest", prompt_template=""):
        """
        Initializes the Mistral API with the specified API key and model.

        Args:
            api_key (str): The API key for Mistral AI.
            model (str): The Mistral AI model version.
            prompt_template (str, optional): A template for prompts. Defaults to a basic format.
        """
        self.api_key = api_key
        self.model = model
        self.prompt_template = prompt_template or "Context: {context}\nQuestion: {question}\nAnswer:"

    def _send_request(self, system_prompt, user_prompt, max_tokens=1024, temperature=0.7, stream=False,
                      presence_penalty=0.0, frequency_penalty=0.0, top_p=1.0):
        """
        Sends a chat completion request to the Mistral API.

        Args:
            system_prompt (str): The system prompt for the model.
            user_prompt (str): The user input to be processed.
            max_tokens (int): The max number of tokens for the completion.
            temperature (float): The sampling temperature for randomness.
            stream (bool): Whether to stream results.
            presence_penalty (float): Presence penalty to encourage/discourage new tokens.
            frequency_penalty (float): Frequency penalty to reduce repetition.
            top_p (float): Nucleus sampling parameter.

        Returns:
            str: The generated response from the model.
        """
        url = 'https://api.mistralai.com/v1/chat'
        headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "stream": stream,
        }

        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            response_json = response.json()
            return response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            logger.error(f"Error from Mistral API: {response.status_code}, {response.text}")
            raise Exception(f"Mistral API request failed: {response.status_code}, {response.text}")

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(self, context, question, max_tokens=1024, temperature=0.7,
                                 stream=False, presence_penalty=0.0, frequency_penalty=0.0, top_p=1.0):
        """
        Attempts to generate an answer to a question based on the context.

        Args:
            context (str): The context for the prompt.
            question (str): The question to generate an answer for.

        Returns:
            str: The generated answer from the API.
        """
        system_prompt = "You are a helpful assistant."
        user_prompt = self.prompt_template.format(context=context, question=question)
        return self._send_request(system_prompt, user_prompt, max_tokens=max_tokens,
                                  temperature=temperature, stream=stream, presence_penalty=presence_penalty,
                                  frequency_penalty=frequency_penalty, top_p=top_p)

    def chat(self, prompt, system_prompt="You are a helpful assistant", max_tokens=1024, temperature=0.7, stream=False,
             presence_penalty=0.0, frequency_penalty=0.0, top_p=1.0):
        """
        Sends a chat prompt to the Mistral model and returns the response.

        Args:
            prompt (str): The user input for the chat.
            system_prompt (str, optional): The system-level prompt for the model.
            max_tokens (int, optional): Maximum number of tokens in the response. Defaults to 1024.
            temperature (float, optional): Temperature for randomness in responses. Defaults to 0.7.
            stream (bool, optional): Whether to stream results. Defaults to False.
            presence_penalty (float, optional): Penalty for repeating tokens. Defaults to 0.0.
            frequency_penalty (float, optional): Penalty for frequency of tokens. Defaults to 0.0.
            top_p (float, optional): Nucleus sampling probability. Defaults to 1.0.

        Returns:
            str: The generated response.
        """
        return self._send_request(system_prompt=system_prompt, user_prompt=prompt, max_tokens=max_tokens,
                                  temperature=temperature, stream=stream,
                                  presence_penalty=presence_penalty, frequency_penalty=frequency_penalty, top_p=top_p)
