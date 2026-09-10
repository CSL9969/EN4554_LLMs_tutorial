"""Full fine-tuning for Qwen/Qwen3.5-2B.

Default dataset: tatsu-lab/alpaca - the standard 52k-example instruction-
following dataset, commonly used as a baseline for instruction fine-tuning.
Pass --dataset to use your own JSONL file instead.

Requires:
    pip install transformers trl datasets accelerate

Usage:
    # Fine-tune on Alpaca (default):
    python full_fine_tune.py --output_dir ./qwen-full-out

    # Fine-tune on your own data:
    python full_fine_tune.py --dataset ./data/train.jsonl --output_dir ./qwen-full-out
"""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "Qwen/Qwen3.5-2B"
DEFAULT_DATASET = "tatsu-lab/alpaca"


def load_model_and_tokenizer(model_name: str):
    """Loads the base model and tokenizer for full fine-tuning."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer


def alpaca_to_messages(example: dict) -> dict:
    """Converts one Alpaca-format example into the chat "messages" format."""
    instruction = example["instruction"]
    extra_input = example.get("input", "")
    user_content = f"{instruction}\n\n{extra_input}" if extra_input else instruction
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["output"]},
        ]
    }


def load_training_dataset(dataset_path: str | None):
    """Loads either the user's custom JSONL dataset or the default Alpaca dataset."""
    if dataset_path:
        print(f"Loading custom dataset: {dataset_path}")
        return load_dataset("json", data_files=dataset_path, split="train")

    print(f"No --dataset given, using default: {DEFAULT_DATASET}")
    dataset = load_dataset(DEFAULT_DATASET, split="train")
    return dataset.map(alpaca_to_messages, remove_columns=dataset.column_names)


def formatting_func(example: dict, tokenizer) -> str:
    """Formats one dataset example into a single training string via the chat template."""
    return tokenizer.apply_chat_template(example["messages"], tokenize=False)


def main():
    parser = argparse.ArgumentParser(description="Full fine-tune Qwen3.5-2B.")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--dataset", default=None, help="Path to a JSONL file with a 'messages' field per line. Defaults to tatsu-lab/alpaca.")
    parser.add_argument("--output_dir", default="./qwen-full-out")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)  # lower LR than LoRA - all weights move now
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--optim", default="adamw_torch", help="Use 'paged_adamw_8bit' (needs bitsandbytes) for lower memory.")
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    dataset = load_training_dataset(args.dataset)
    print(f"Training examples: {len(dataset)}")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_seq_length,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
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

    print("Starting full fine-tuning...")
    trainer.train()

    print(f"Saving fine-tuned model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()