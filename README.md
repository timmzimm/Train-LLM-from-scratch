# GPT-2 Style LLM Training from Scratch

**Inspired by [Sebastian Raschka's](https://github.com/rasbt/LLMs-from-scratch) repository and his book, but not borrowed from it.**  
This project implements a simplified GPT-2 style model (~120M parameters) from scratch and trains it on a large instruction dataset. The code is modular, uses configuration files, and aims to be educational—demonstrating the full pipeline without external large language model libraries.

## Overview

This repository showcases:

- **Custom BPE Tokenization:** A simplified Byte-Pair Encoding tokenizer inspired by GPT-2.
- **GPT-2 Style Model Architecture:** Multi-head causal self-attention, residual connections, and layer norms.
- **Config-Driven Pipeline:** All parameters are controlled via JSON configs in the `config/` directory.

## Dataset

We train on the [Orca Agent Instruct Dataset (1M)](https://huggingface.co/datasets/microsoft/orca-agentinstruct-1M-v1) by Microsoft. This dataset:

- Contains ~1 million user-assistant message pairs.
- Covers a wide range of instructions and responses.
- Provides realistic conversational patterns for better model adaptation.

## Requirements

- **Python:** 3.11.10
- Other dependencies listed in `requirements.txt`.

## Installation & Usage

**Clone the repository:**
   ```bash
   git clone https://github.com/timmzimm/Train-LLM-from-scratch.git
   cd Train-LLM-from-scratch
   pip install -r requirements.txt
   ```

## Acknowledgments
Sebastian Raschka: His transparent approach to LLMs was an inspiration.
Microsoft & Hugging Face: For providing and hosting the Orca dataset


