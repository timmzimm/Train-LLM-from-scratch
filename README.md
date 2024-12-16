GPT-2 Style LLM Training from Scratch

Inspired by Sebastian Raschka's repository and his book, but not borrowed from it. This project implements a simplified GPT-2 style model and trains it from scratch on a large instruction dataset. The code is structured, modular, and configured through dedicated JSON files.

<p align="center"> <img src="https://user-images.githubusercontent.com/your_image.png" alt="Model Training Illustration" width="500"> </p>
Overview
This repository demonstrates how to:

Tokenize raw text using a simplified Byte-Pair Encoding (BPE) tokenizer inspired by GPT-2.
Train a GPT-2 style model from scratch, including the multi-head causal self-attention architecture, residual connections, and layer normalization.
Use configuration files (.json in config/) to control data preprocessing, model architecture, and training hyperparameters.
The approach is educational, helping you understand the full pipeline of building and training a GPT-like model without relying on external libraries like transformers.

Dataset
We use the Orca Agent Instruct Dataset (1M) from Microsoft. This dataset provides a variety of user-assistant interactions that can serve as a foundation for instruction-tuned language models. It contains messages with user and assistant roles, enabling models to learn from diverse conversational prompts and responses.

Key Features:

High-quality instructions: The dataset is curated to reflect a wide range of tasks.
Large scale: With around 1 million samples, it supports extensive training at scale.
Realistic dialogues: It mimics real assistant-user interactions, improving model adaptation to instruction-like queries.
Project Structure
lua
Copy code
project/
|-- main.py                  # Entry point for training
|-- config/
|   |-- dataset_config.json
|   |-- model_config.json
|   `-- training_config.json
|-- notebooks/               # For analysis and experiments (Jupyter)
|-- src/
|   |-- tokenization/
|   |   `-- bpe_tokenizer.py
|   |-- model/
|   |   `-- gpt2.py
|   |-- data/
|   |   `-- dataset.py
|   |-- training/
|   |   `-- train.py
|   `-- utils/
|       `-- evaluation.py
|-- model_checkpoint/
|   |-- tokenizer.json       # Saved tokenizer
|   `-- model.pt             # Model weights (not committed by default)
|-- requirements.txt
|-- README.md
|-- .gitignore
Requirements
Python: 3.11.10
Dependencies listed in requirements.txt.
Installation & Usage
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/your-repo.git
Change directory:

bash
Copy code
cd your-repo
Install requirements:

bash
Copy code
pip install -r requirements.txt
Run training:

bash
Copy code
python main.py
This will:

Load configurations from config/*.json
Download and preprocess the dataset.
Train the tokenizer and model.
Save the trained model and tokenizer under model_checkpoint/.
Note: By default, the model.pt file is not tracked by Git due to .gitignore settings.

Acknowledgments
Sebastian Raschka: For inspiration from his repository and book, which laid out a transparent approach to LLMs. This code is newly implemented and not borrowed directly.
Microsoft & Hugging Face: For providing and hosting the Orca dataset.
