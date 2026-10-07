"""A backend that loads a transformers model once and serves it over HTTP."""

import argparse

import torch
from flask import Flask, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen3.5-0.8B"

app = Flask(__name__)

tokenizer = None
model = None


def load_model(model_name: str = MODEL_NAME, load_in_4bit: bool = False):
    """Loads the tokenizer and model.

    Args:
        model_name (str): The name of the model to load.
        load_in_4bit (bool): If True, loads the model in 4-bit (NF4) using bitsandbytes
            to reduce GPU memory. Defaults to False (bf16).

    Returns:
        tuple: The loaded (tokenizer, model).
    """
    print(f"Loading {model_name} ({'4-bit' if load_in_4bit else 'bf16'})...")
    tok = AutoTokenizer.from_pretrained(model_name)

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="cuda",  # load the whole model on the GPU (no CPU offloading)
    )
    mdl.eval()
    print("Model loaded. Server ready.")
    return tok, mdl


@app.route("/generate", methods=["POST"])
def generate():
    """Generates a response for a list of chat messages.

    Expects JSON body: {"messages": [...], "max_new_tokens": 512}
    Returns JSON body: {"response": "..."}
    """
    data = request.get_json(silent=True)
    if not data or "messages" not in data:
        return jsonify({"error": "Missing 'messages' field in request body"}), 400

    messages = data["messages"]
    max_new_tokens = data.get("max_new_tokens", 512)

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return jsonify({"response": response_text})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve a transformers model over HTTP.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load the model in 4-bit to reduce GPU memory.")
    args = parser.parse_args()

    tokenizer, model = load_model(load_in_4bit=args.load_in_4bit)
    app.run(host="0.0.0.0", port=8000)