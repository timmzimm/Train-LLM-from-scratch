import json
import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

# Оставляем ваши кастомные токенизаторы
from src.tokenization.bpe_tokenizer import BPETokenizer
from src.tokenization.bytelevel_bpe_tokenizer import ByteLevelBPETokenizer

# Импортируем Hugging Face токенизатор
from transformers import GPT2TokenizerFast

from src.model.gpt2 import GPT2Config, GPT2Model
from src.data.dataset import TextDataset, extract_texts
from src.training.train import train
from src.utils.evaluation import evaluate

def main():
    """
    Main entry point for training the GPT-2 style model with multiple tokenization options:
      - huggingface (GPT2TokenizerFast) — default
      - char        (our char-level BPE)
      - byte        (our byte-level BPE)
    """
    # Load configs
    with open("config/dataset_config.json", "r") as f:
        dataset_config = json.load(f)
    with open("config/model_config.json", "r") as f:
        model_config_data = json.load(f)
    with open("config/training_config.json", "r") as f:
        train_config = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --------------------------------------------------
    # 1) Load dataset & extract raw texts (omitted details)
    # --------------------------------------------------
    dataset = load_dataset(dataset_config["dataset_name"])
    ds_train = dataset["analytical_reasoning"]  # example
    ds_val = dataset["text_modification"]       # example
    train_texts = extract_texts(ds_train)
    val_texts = extract_texts(ds_val)

    # --------------------------------------------------
    # 2) Initialize Tokenizer
    # --------------------------------------------------
    tokenization_type = train_config["tokenization_type"]  # "huggingface", "char", or "byte"
    
    if tokenization_type == "huggingface":
        hf_tokenizer_name = train_config["hf_tokenizer_name"]  # e.g. "gpt2"
        print(f"Loading Hugging Face tokenizer: {hf_tokenizer_name}")
        tokenizer = GPT2TokenizerFast.from_pretrained(hf_tokenizer_name)
        # Add special tokens if needed (if not already in tokenizer)
        # For example, <|endoftext|> is usually in GPT2 vocab, but let's ensure:
        if train_config["special_tokens"]:
            tokenizer.add_special_tokens({"additional_special_tokens": train_config["special_tokens"]})
        
        # We'll skip custom training here, because HF tokenizer is pretrained
        tokenizer_filename = "tokenizer_hf"

    elif tokenization_type == "char":
        print("Initializing char-level BPE tokenizer.")
        tokenizer = BPETokenizer(
            special_tokens=train_config["special_tokens"],
            vocab_size_limit=train_config["vocab_size_limit"],
            merges_count=train_config["merges_count"]
        )
        print("Training char-level BPE on train_texts...")
        tokenizer.train(train_texts)
        tokenizer_filename = "tokenizer_char"

    elif tokenization_type == "byte":
        print("Initializing byte-level BPE tokenizer.")
        tokenizer = ByteLevelBPETokenizer(
            special_tokens=train_config["special_tokens"],
            vocab_size_limit=train_config["vocab_size_limit"],
            merges_count=train_config["merges_count"]
        )
        print("Training byte-level BPE on train_texts...")
        tokenizer.train(train_texts)
        tokenizer_filename = "tokenizer_byte"

    else:
        raise ValueError(f"Unknown tokenization_type: {tokenization_type}")

    # --------------------------------------------------
    # 3) Encode texts
    # --------------------------------------------------
    if tokenization_type == "huggingface":
        # Hugging Face tokenizer usually encodes via tokenizer.encode(...)
        # It returns a list of IDs
        train_ids = []
        for text in train_texts:
            train_ids.extend(tokenizer.encode(text))
        val_ids = []
        for text in val_texts:
            val_ids.extend(tokenizer.encode(text))

        # The tokenizer vocabulary size is from the HF object
        vocab_size = tokenizer.vocab_size + len(train_config["special_tokens"] or [])
        # because add_special_tokens() might have expanded the vocab

    else:
        # char or byte (your custom approach)
        train_ids = []
        for text in train_texts:
            train_ids.extend(tokenizer.encode(text))
        val_ids = []
        for text in val_texts:
            val_ids.extend(tokenizer.encode(text))
        vocab_size = tokenizer.vocab.vocab_size

    # --------------------------------------------------
    # 4) Create PyTorch Datasets
    # --------------------------------------------------
    block_size = train_config["block_size"]
    train_dataset = TextDataset(train_ids, block_size)
    val_dataset = TextDataset(val_ids, block_size) if val_ids else None

    train_loader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_config["batch_size"], shuffle=False) if val_dataset else []

    print(f"Train dataset sequences: {len(train_dataset)}")
    print(f"Val dataset sequences: {len(val_dataset) if val_dataset else 0}")

    # --------------------------------------------------
    # 5) Initialize GPT-2 config and model
    # --------------------------------------------------
    config = GPT2Config(
        vocab_size=vocab_size,
        n_ctx=model_config_data["n_ctx"],
        n_embd=model_config_data["n_embd"],
        n_layer=model_config_data["n_layer"],
        n_head=model_config_data["n_head"]
    )
    model = GPT2Model(config).to(device)
    print("Model created with", sum(p.numel() for p in model.parameters()), "parameters.")

    # --------------------------------------------------
    # 6) Train
    # --------------------------------------------------
    train(model, train_loader, val_loader, device, train_config)

    # --------------------------------------------------
    # 7) Save model & tokenizer
    # --------------------------------------------------
    print("Saving model...")
    os.makedirs("model_checkpoint", exist_ok=True)
    torch.save(model.state_dict(), "model_checkpoint/model.pt")

    if tokenization_type == "huggingface":
        # We can save the HF tokenizer via save_pretrained
        tokenizer.save_pretrained("model_checkpoint/hf_tokenizer")
    else:
        # Save your custom tokenizer merges etc.
        # Note that we skip .json in the var so we can differentiate among types
        with open(f"model_checkpoint/{tokenizer_filename}.json", "w") as f:
            json.dump({
                "special_tokens": tokenizer.vocab.special_tokens,
                "token2id": tokenizer.vocab.token2id,
                "merges": tokenizer.vocab.merges
            }, f)

    print("Done!")

if __name__ == "__main__":
    main()



