import os
import json
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import GPT2TokenizerFast
from src.tokenization.bpe_tokenizer import BPETokenizer
from src.tokenization.bytelevel_bpe_tokenizer import ByteLevelBPETokenizer
from src.data.dataset import TextDataset, extract_texts
from src.data.load_data import load_and_merge_splits
from src.model.gpt2 import GPT2Config, GPT2Model
from src.training.train import train

def run_training_pipeline_single(train_config):
    """
    Single-GPU or CPU training pipeline.
    """
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    local_rank = 0
    _run_pipeline_core(train_config, device, local_rank, ddp=False)

def run_training_pipeline_ddp(train_config, local_rank):
    """
    Multi-GPU DDP training pipeline for a single process (GPU).
    local_rank is set by PyTorch (0..N-1).
    """
    gpu_ids = train_config["gpu_ids"]
    device_id = gpu_ids[local_rank]
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    _run_pipeline_core(train_config, device, local_rank, ddp=True)

def _run_pipeline_core(train_config, device, local_rank, ddp=False):
    """
    Core logic:
    1) Load dataset config and merge all splits.
    2) Extract texts from train/val (test is optional).
    3) Tokenize texts (choose huggingface/char/byte).
    4) Create DataLoader (distributed if ddp).
    5) Create GPT-2 model, DDP if needed.
    6) Run training.
    7) Save model (on rank=0).
    """

    # 1) Load & merge splits
    train_ds_full, val_ds_full, test_ds_full = load_and_merge_splits("config/dataset_config.json")
    train_texts = extract_texts(train_ds_full)
    val_texts   = extract_texts(val_ds_full)
    
    # Limit data for quick test (optional)
    max_train = 1000  # or any number you want
    max_val   = 200
    train_texts = train_texts[:max_train]
    val_texts   = val_texts[:max_val]

    # 2) Initialize tokenizer
    tokenization_type = train_config["tokenization_type"]
    if tokenization_type == "huggingface":
        tokenizer = GPT2TokenizerFast.from_pretrained(train_config["hf_tokenizer_name"])
        vocab_size = tokenizer.vocab_size

        def encode_func(txt):
            return tokenizer.encode(txt)

    elif tokenization_type == "char":
        tokenizer = BPETokenizer(
            special_tokens=train_config["special_tokens"],
            vocab_size_limit=train_config["vocab_size_limit"],
            merges_count=train_config["merges_count"]
        )
        tokenizer.train(train_texts)
        vocab_size = tokenizer.vocab.vocab_size

        def encode_func(txt):
            return tokenizer.encode(txt)

    elif tokenization_type == "byte":
        tokenizer = ByteLevelBPETokenizer(
            special_tokens=train_config["special_tokens"],
            vocab_size_limit=train_config["vocab_size_limit"],
            merges_count=train_config["merges_count"]
        )
        tokenizer.train(train_texts)
        vocab_size = tokenizer.vocab.vocab_size

        def encode_func(txt):
            return tokenizer.encode(txt)
    else:
        raise ValueError(f"Unknown tokenization_type: {tokenization_type}")

   # 3) Encode data with tqdm progress

    train_ids = []
    for t in tqdm(train_texts, desc="Encoding train texts"):
        train_ids.extend(encode_func(t))

    val_ids = []
    for t in tqdm(val_texts, desc="Encoding val texts"):
        val_ids.extend(encode_func(t))

    # 4) Create PyTorch datasets & (optionally) distributed sampler
    block_size = train_config["block_size"]
    train_dataset = TextDataset(train_ids, block_size)
    val_dataset   = TextDataset(val_ids,   block_size)

    if ddp:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler   = DistributedSampler(val_dataset, shuffle=False)
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_config["batch_size"])
        val_loader   = DataLoader(val_dataset,   sampler=val_sampler,   batch_size=train_config["batch_size"])
    else:
        train_loader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=train_config["batch_size"], shuffle=False)

    # 5) Build model
    with open("config/model_config.json", "r") as f:
        model_config_data = json.load(f)

    gpt2_cfg = GPT2Config(
        vocab_size=vocab_size,
        n_ctx=model_config_data["n_ctx"],
        n_embd=model_config_data["n_embd"],
        n_layer=model_config_data["n_layer"],
        n_head=model_config_data["n_head"]
    )
    model = GPT2Model(gpt2_cfg).to(device)

    if ddp:
        model = DDP(model, device_ids=[device.index], output_device=device.index)

    # 6) Train
    train(model, train_loader, val_loader, device, train_config)

    # 7) Save model on rank=0
    if (not ddp) or (ddp and local_rank == 0):
        _save_model_and_tokenizer(model, tokenizer, tokenization_type)

def _save_model_and_tokenizer(model, tokenizer, tokenization_type):
    """
    Saves the model checkpoint and tokenizer (if any).
    """
    os.makedirs("model_checkpoint", exist_ok=True)
    torch.save(model.state_dict(), "model_checkpoint/model.pt")

    if tokenization_type == "huggingface":
        # GPT2TokenizerFast can save_pretrained
        tokenizer.save_pretrained("model_checkpoint/hf_tokenizer")
    else:
        # For char/byte custom BPE
        # If you want merges & token2id, add them here
        pass

    print("[Pipeline] Model saved.")



