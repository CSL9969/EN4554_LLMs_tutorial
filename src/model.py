"""The module loads the model and the tokenizer."""

from transformers import AutoTokenizer, AutoModelForCausalLM


class ChatBot:
    """The class loads the model and the tokenizer."""

    def __init__(self, model_name: str | None = None):
        """Initializes the ChatBot class.

        Args:
            model_name (str | None): The name of the model to load.
        """
        model_name = "Qwen/Qwen3.5-2B" if model_name is None else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.chat_history = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]

    def generate_response(self, prompt: str, max_new_tokens: int = 1000) -> str:
        """Generates a response from the model given a prompt.

        Args:
            prompt (str): The input prompt to generate a response for.
            max_new_tokens (int): The maximum number of NEW tokens to generate
                (independent of how long the conversation so far is).

        Returns:
            str: The generated response.
        """
        self.chat_history.append({"role": "user", "content": prompt})

        inputs = self.tokenizer.apply_chat_template(
            self.chat_history,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Slice off the input tokens so we only decode the newly generated part.
        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        self.chat_history.append({"role": "assistant", "content": response})
        return response

    def reset_chat_history(self):
        """Resets the chat history to the initial system message."""
        self.chat_history = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]

    def get_formatted_input(self) -> str:
        """Returns the formatted input for the model based on the current chat history.

        Returns:
            str: The formatted input string.
        """
        return self.tokenizer.apply_chat_template(
            self.chat_history,
            add_generation_prompt=False,
            tokenize=False,
            return_dict=False,
        )

if __name__ == "__main__":
    bot = ChatBot()
    print(bot.generate_response("What is the Capital of France?"))
    print(bot.get_formatted_input())