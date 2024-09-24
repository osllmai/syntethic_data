from tenacity import retry, stop_after_attempt, wait_random_exponential
<<<<<<< HEAD:llms/googleai.py
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
=======
>>>>>>> 24d63bbe73e538f2fa1317b3a770801779266ec0:indoxGen/llms/googleai.py


class GoogleAi:
    def __init__(self, api_key, model="gemini-1.5-flash-latest"):
        """
        Initializes the Google AI model with the specified model version.

        Args:
            api_key (str): The API key for Google AI.
            model (str): The Google AI model version. Defaults to "gemini-1.5-flash-latest".
        """
        import google.generativeai as genai  # Assuming this is the Google API module for Gemini

        try:
            logger.info(f"Initializing GoogleAi with model: {model}")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
            logger.info("GoogleAi initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing GoogleAi: {e}")
            raise

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _generate_response(self, prompt):
        """
        Generates a response using the model.

        Args:
            prompt (str): The prompt to generate a response for.

        Returns:
            str: The generated response text.
        """
        try:
            logger.info("Generating response")
            response = self.model.generate_content(contents=prompt)
            return response.text.strip().replace("\n", "")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def chat(self, prompt, system_prompt=""):
        """
        A method to send a combined system and user prompt to the Google AI API for data generation or judging.

        Args:
            prompt (str): The user prompt to be sent to the model.
            system_prompt (str, optional): An optional system prompt for guiding the model. Defaults to "".

        Returns:
            str: The generated response from the model.
        """
<<<<<<< HEAD:llms/googleai.py
        combined_prompt = system_prompt + "\n" + prompt
        try:
            logger.info(f"Sending prompt to GoogleAi: {combined_prompt[:100]}...")  # Log truncated prompt
            return self._generate_response(combined_prompt)
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return str(e)
=======
        url = 'https://api.googleai.com/v1/chat'
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
            raise Exception(f"Google Gemini API request failed: {response.status_code}, {response.text}")

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
        Sends a chat prompt to the Google Gemini model and returns the response.

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
>>>>>>> 24d63bbe73e538f2fa1317b3a770801779266ec0:indoxGen/llms/googleai.py
