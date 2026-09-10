"""A backend that loads a transformers model once and serves it over HTTP."""

import torch
from flask import Flask, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3.5-2B"

app = Flask(__name__)

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
print("Model loaded. Server ready.")


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
    app.run(host="0.0.0.0", port=8000)