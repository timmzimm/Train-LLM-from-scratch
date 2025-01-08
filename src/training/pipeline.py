import os
import json
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# Импорт модулей вашего проекта
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from src.tokenization.bpe_tokenizer import BPETokenizer
from src.tokenization.bytelevel_bpe_tokenizer import ByteLevelBPETokenizer
from src.data.dataset import TextDataset, extract_texts
from src.model.gpt2 import GPT2Config, GPT2Model
from src.training.train import train
# ... etc.

def run_training_pipeline_single(train_config):
    """
    Single-GPU (or CPU) pipeline:
    - device = cuda:0 if available, else cpu
    - loads dataset
    - prepares model
    - trains
    - saves checkpoint
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    local_rank = 0  # для совместимости

    _run_pipeline_core(train_config, device, local_rank, ddp=False)


def run_training_pipeline_ddp(local_rank, train_config):
    """
    DDP pipeline for a single process (one GPU).
    Typically called from ddp_main().

    We map local_rank to an actual GPU index from train_config["gpu_ids"].
    """
    gpu_ids = train_config["gpu_ids"]
    device_id = gpu_ids[local_rank]  # 0 -> gpu_ids[0], 1 -> gpu_ids[1], etc.
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    _run_pipeline_core(train_config, device, local_rank, ddp=True)


def _run_pipeline_core(train_config, device, local_rank, ddp=False):
    """
    The actual pipeline code:
    1) Load dataset
    2) Extract texts
    3) Initialize tokenizer
    4) Encode data
    5) Build model (optionally wrap in DDP)
    6) Train
    7) Save model (if local_rank == 0)
    """

    # ----------------------------------------------------------------
    # Load dataset config & model config
    # ----------------------------------------------------------------
    with open("config/dataset_config.json", "r") as f:
        dataset_config = json.load(f)
    with open("config/model_config.json", "r") as f:
        model_config_data = json.load(f)

    # ----------------------------------------------------------------
    # 1) Load dataset & splits (example usage)
    # ----------------------------------------------------------------
    ds = load_dataset(dataset_config["dataset_name"])
    ds_train = ds[dataset_config["train_split"]]
    ds_val = ds[dataset_config["val_split"]]

    # extract text
    train_texts = extract_texts(ds_train)
    val_texts = extract_texts(ds_val)

    # ----------------------------------------------------------------
    # 2) Initialize tokenizer (huggingface/char/byte)
    # ----------------------------------------------------------------
    tokenization_type = train_config["tokenization_type"]
    if tokenization_type == "huggingface":
        hf_tokenizer_name = train_config["hf_tokenizer_name"]
        tokenizer = GPT2TokenizerFast.from_pretrained(hf_tokenizer_name)
        # optionally add special tokens
        # tokenizer.add_special_tokens(...)
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

    # ----------------------------------------------------------------
    # 3) Encode data
    # ----------------------------------------------------------------
    train_ids = []
    for text in train_texts:
        train_ids.extend(encode_func(text))
    val_ids = []
    for text in val_texts:
        val_ids.extend(encode_func(text))

    # ----------------------------------------------------------------
    # 4) Create PyTorch datasets & loaders
    # ----------------------------------------------------------------
    block_size = train_config["block_size"]
    train_dataset = TextDataset(train_ids, block_size)
    val_dataset = TextDataset(val_ids, block_size)

    if ddp:
        # Each process sees a subset of data
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_config["batch_size"], sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=train_config["batch_size"], sampler=val_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=train_config["batch_size"], shuffle=False)

    # ----------------------------------------------------------------
    # 5) Create model
    # ----------------------------------------------------------------
    config = GPT2Config(
        vocab_size=vocab_size,
        n_ctx=model_config_data["n_ctx"],
        n_embd=model_config_data["n_embd"],
        n_layer=model_config_data["n_layer"],
        n_head=model_config_data["n_head"]
    )
    model = GPT2Model(config).to(device)

    if ddp:
        model = DDP(model, device_ids=[device.index], output_device=device.index)

    # ----------------------------------------------------------------
    # 6) Train
    # ----------------------------------------------------------------
    train(model, train_loader, val_loader, device, train_config)

    # ----------------------------------------------------------------
    # 7) Save model (only rank 0 usually)
    # ----------------------------------------------------------------
    if (not ddp) or (ddp and local_rank == 0):
        # save checkpoint
        os.makedirs("model_checkpoint", exist_ok=True)
        torch.save(model.state_dict(), "model_checkpoint/model.pt")

        # save tokenizer
        if tokenization_type == "huggingface":
            tokenizer.save_pretrained("model_checkpoint/hf_tokenizer")
        else:
            # Save merges / vocab
            # ...
            pass

        print("[Pipeline] Model saved.")


