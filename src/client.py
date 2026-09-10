"""Client that talks to the manually-hosted transformers backend via HTTP."""

import requests

BASE_URL = "http://localhost:8000"


class ChatBot:
    """Chat client for the manual /generate backend."""

    def __init__(self, base_url: str = BASE_URL):
        """Initializes the ChatBot class.

        Args:
            base_url (str): The base URL of the backend_server.py Flask app.
        """
        self.base_url = base_url
        self.chat_history = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]

    def generate_response(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generates a response from the backend given a prompt.

        Args:
            prompt (str): The input prompt to generate a response for.
            max_new_tokens (int): The maximum number of NEW tokens to generate.

        Returns:
            str: The generated response.
        """
        self.chat_history.append({"role": "user", "content": prompt})

        resp = requests.post(
            f"{self.base_url}/generate",
            json={"messages": self.chat_history, "max_new_tokens": max_new_tokens},
            timeout=120,
        )
        resp.raise_for_status()
        response = resp.json()["response"]

        self.chat_history.append({"role": "assistant", "content": response})
        return response


if __name__ == "__main__":
    bot = ChatBot()
    print(bot.generate_response("What is the Capital of France?"))