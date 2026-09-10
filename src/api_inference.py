"""The module talks to the vLLM-hosted model via its OpenAI-compatible API."""

from openai import OpenAI


class ChatBot:
    """Chat client that hits the local vLLM server."""

    def __init__(self, model_name: str = "Qwen/Qwen3.5-2B", base_url: str = "http://localhost:8000/v1"):
        """Initializes the ChatBot class.

        Args:
            model_name (str): The model identifier as registered on the vLLM server.
            base_url (str): The base URL of the vLLM OpenAI-compatible server.
        """
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")

        self.chat_history = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]

    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generates a response from the model given a prompt.

        Args:
            prompt (str): The input prompt to generate a response for.
            max_tokens (int): The maximum number of NEW tokens to generate.

        Returns:
            str: The generated response.
        """
        self.chat_history.append({"role": "user", "content": prompt})

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.chat_history,
            max_tokens=max_tokens,
        )

        response = completion.choices[0].message.content
        self.chat_history.append({"role": "assistant", "content": response})
        return response

if __name__ == "__main__":
    bot = ChatBot()
    print(bot.generate_response("What is the Capital of France?"))