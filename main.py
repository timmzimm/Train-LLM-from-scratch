import json
import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from src.tokenization.bpe_tokenizer import BPETokenizer
from src.tokenization.bytelevel_bpe_tokenizer import ByteLevelBPETokenizer
from src.model.gpt2 import GPT2Config, GPT2Model
from src.data.dataset import TextDataset, extract_texts
from src.training.train import train
from src.utils.evaluation import evaluate

def main():
    """
    Main entry point for training the GPT-2 style model.

    Steps:
    - Load configuration files.
    - Load and preprocess the dataset.
    - Train the tokenizer on the training texts.
    - Encode texts using the trained tokenizer.
    - Create PyTorch datasets and loaders.
    - Initialize and train the GPT-2 model.
    - Save the final tokenizer and model checkpoint.
    """
    # Load configurations
    with open("config/dataset_config.json", "r") as f:
        dataset_config = json.load(f)
    with open("config/model_config.json", "r") as f:
        model_config_data = json.load(f)
    with open("config/training_config.json", "r") as f:
        train_config = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading dataset...")
    dataset = load_dataset(dataset_config["dataset_name"])
    ds_train = dataset[dataset_config["train_split"]]
    ds_val = dataset[dataset_config["val_split"]]

    print("Extracting training texts...")
    train_texts = extract_texts(ds_train, max_samples=dataset_config["max_train_samples"])
    print(f"Collected {len(train_texts)} training samples.")

    print("Extracting validation texts...")
    val_texts = extract_texts(ds_val, max_samples=dataset_config["max_val_samples"])
    print(f"Collected {len(val_texts)} validation samples.")

    if not train_texts:
        raise ValueError("No training data extracted. Check the dataset and extraction logic.")
    
    

    print("Training tokenizer...")
    tokenization_type = train_config["tokenization_type"]  # "char" or "byte"
    if tokenization_type == "char":
        tokenizer = BPETokenizer(
            special_tokens=train_config["special_tokens"],
            vocab_size_limit=train_config["vocab_size_limit"],
            merges_count=train_config["merges_count"]
        )
        tokenizer_filename = "tokenizer_char.json"
    elif tokenization_type == "byte":
        tokenizer = ByteLevelBPETokenizer(
            special_tokens=train_config["special_tokens"],
            vocab_size_limit=train_config["vocab_size_limit"],
            merges_count=train_config["merges_count"]
        )
        tokenizer_filename = "tokenizer_byte.json"
    else:
        raise ValueError(f"Unknown tokenization_type: {tokenization_type}")
    # tokenizer = BPETokenizer(
    #     special_tokens=train_config["special_tokens"],
    #     vocab_size_limit=train_config["vocab_size_limit"],
    #     merges_count=train_config["merges_count"]
    # )
    tokenizer.train(train_texts)
    print("Tokenizer trained.")

    print("Encoding train set...")
    train_ids = []
    for text in train_texts:
        train_ids.extend(tokenizer.encode(text))

    print("Encoding val set...")
    val_ids = []
    for text in val_texts:
        val_ids.extend(tokenizer.encode(text))

    if not train_ids:
        raise ValueError("No tokens in training data after tokenization.")

    if not val_ids:
        print("No tokens in validation data. Will train without validation.")

    block_size = train_config["block_size"]
    train_dataset = TextDataset(train_ids, block_size)
    val_dataset = TextDataset(val_ids, block_size) if val_ids else None

    train_loader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_config["batch_size"], shuffle=False) if val_dataset else []

    print(f"Train dataset sequences: {len(train_dataset)}, Val dataset sequences: {len(val_dataset) if val_dataset else 0}")

    # Initialize GPT-2 config
    config = GPT2Config(
        vocab_size=tokenizer.vocab.vocab_size,
        n_ctx=model_config_data["n_ctx"],
        n_embd=model_config_data["n_embd"],
        n_layer=model_config_data["n_layer"],
        n_head=model_config_data["n_head"]
    )

    model = GPT2Model(config).to(device)
    print("Model created with", sum(p.numel() for p in model.parameters()), "parameters.")

    print("Starting training...")
    # main.py (excerpt where we call train)


    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        train_config=train_config
    )


    print("Saving model and tokenizer...")
    os.makedirs("model_checkpoint", exist_ok=True)
    torch.save(model.state_dict(), "model_checkpoint/model.pt")

    with open(f"model_checkpoint/{tokenizer_filename}", "w") as f:
        json.dump({
            "special_tokens": tokenizer.vocab.special_tokens,
            "token2id": tokenizer.vocab.token2id,
            "merges": tokenizer.vocab.merges
        }, f)

    print("Training complete. Model and tokenizer saved.")

if __name__ == "__main__":
    main()

