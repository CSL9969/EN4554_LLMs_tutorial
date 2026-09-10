# LLM Inference & Fine-Tuning Toolkit

This repository contains code for:
- LLM inference using Transformers
- LLM inference using vLLM
- Manually hosting models via Flask
- Retrieval-Augmented Generation (RAG)
- Agentic pipelines (LangChain / LangGraph)
- Full fine-tuning and LoRA/QLoRA fine-tuning

## Requirements

- Python 3.12
- NVIDIA GPU with CUDA support (recommended for inference and fine-tuning)
- CUDA 13.0 (this project was developed and tested with `torch==2.13.0+cu130`)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/CSL9969/EN4554_LLMs_tutorial.git
cd <EN4554_LLMs_tutorial>
```

### 2. Create and activate a virtual environment

Using conda:
```bash
conda create -n llm_demo python=3.12
conda activate llm_demo
```

Or using venv:
```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install PyTorch (match your CUDA version first)

`requirements.txt` pins `torch==2.13.0`, but PyTorch builds are CUDA-version specific and not hosted on the default PyPI index. Install torch separately first, matching your system's CUDA version, using the official selector:

👉 https://pytorch.org/get-started/locally/

Verify the install:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
`torch.cuda.is_available()` should print `True` if your GPU is detected correctly.

### 4. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

> Note: If `torch`, `torchvision`, and `torchaudio` are already installed from step 3 with a matching version, pip will skip reinstalling them. If you see a version conflict, remove the `torch==`, `torchvision==`, and `torchaudio==` lines from `requirements.txt` before running this step.
