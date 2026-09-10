"""Fine-tunes Qwen/Qwen3.5-2B with LoRA on a chat-format dataset.

Expected dataset format: a JSONL file where each line is a JSON object with a
"messages" field, e.g.:

    {"messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."}
    ]}

Usage:
    python fine_tune.py --dataset ./data/train.jsonl --output_dir ./qwen-lora-out
"""

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "Qwen/Qwen3.5-2B"


def load_model_and_tokenizer(model_name: str):
    """Loads the base model and tokenizer for fine-tuning."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer


def apply_lora(model, r: int, alpha: int, dropout: float):
    """Wraps the base model with a LoRA adapter.

    Args:
        model: The base causal LM.
        r (int): LoRA rank — higher means more trainable capacity (and memory).
        alpha (int): LoRA scaling factor, typically 2x the rank.
        dropout (float): Dropout applied inside the LoRA layers.

    Returns:
        The model wrapped with LoRA adapters.
    """
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # Standard attention + MLP projection targets for Qwen-style architectures.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def formatting_func(example: dict, tokenizer) -> str:
    """Formats one dataset example into a single training string via the chat template."""
    return tokenizer.apply_chat_template(example["messages"], tokenize=False)


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen3.5-2B on a chat dataset.")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--dataset", required=True, help="Path to a JSONL file with a 'messages' field per line.")
    parser.add_argument("--output_dir", default="./qwen-lora-out")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    model = apply_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)

    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset("json", data_files=args.dataset, split="train")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_seq_length,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=lambda example: formatting_func(example, tokenizer),
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving LoRA adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()