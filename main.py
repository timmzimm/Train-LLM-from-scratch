import os
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.model.gpt2 import GPT2Config, GPT2Model
from src.tokenization.bpe_tokenizer import BPETokenizer
from src.tokenization.bytelevel_bpe_tokenizer import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast
from src.data.dataset import TextDataset, extract_texts
from src.training.train import train
from src.utils.evaluation import evaluate

from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    """
    Main entry point for distributed or single-GPU training, depending on config.
    We'll parse the config, init process group (if distributed), etc.
    """
    with open("config/training_config.json", "r") as f:
        train_config = json.load(f)

    distributed = train_config.get("distributed", False)
    
    if distributed:
        # We launch main_worker on each process
        # The recommended approach is to call this script with torchrun or the launch utility
        # which sets LOCAL_RANK and WORLD_SIZE automatically
        world_size = len(train_config["gpu_ids"])  # e.g. 2
        # main_worker will run in each process
        # local_rank is read from environment
        main_worker_ddp(world_size, train_config)
    else:
        # Single-GPU or CPU fallback
        single_worker(train_config)

def main_worker_ddp(world_size, train_config):
    """
    Worker function for DDP. Each process has its own local_rank set by torchrun.
    """
    # local_rank is the index of the GPU on this node
    local_rank = int(os.environ["LOCAL_RANK"])  # 0 or 1, etc.
    gpu_ids = train_config["gpu_ids"]           # e.g. [0,1]
    device_id = gpu_ids[local_rank]             # map rank -> actual GPU index

    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        init_method="env://"
    )

    # Set device
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    # Then do everything else: load dataset, create model, wrap with DDP, train...
    # We'll call a helper that does the pipeline
    run_training_pipeline(train_config, device, local_rank, ddp=True)

    # Cleanup
    dist.destroy_process_group()

def single_worker(train_config):
    """
    Single-GPU or CPU training fallback.
    """
    # By default, if we only have one GPU, we use device = 'cuda:0' if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_training_pipeline(train_config, device, local_rank=0, ddp=False)


def run_training_pipeline(train_config, device, local_rank, ddp=False):
    """
    Here we do what main() used to do: load data, create model, train, etc.
    ddp indicates if we're in DistributedDataParallel mode or not.
    """
    # Load dataset, model_config, dataset_config
    with open("config/dataset_config.json", "r") as f:
        dataset_config = json.load(f)
    with open("config/model_config.json", "r") as f:
        model_config_data = json.load(f)

    # For brevity, let's assume we only do small subset of the pipeline
    # (In real code, you'd load your dataset, extract texts, etc.)
    # Example:
    from datasets import load_dataset
    ds = load_dataset(dataset_config["dataset_name"])
    train_ds = ds["analytical_reasoning"]
    val_ds = ds["text_modification"]

    train_texts = extract_texts(train_ds)
    val_texts = extract_texts(val_ds)

    # Initialize tokenizer
    tokenization_type = train_config["tokenization_type"]
    if tokenization_type == "huggingface":
        tokenizer = GPT2TokenizerFast.from_pretrained(train_config["hf_tokenizer_name"])
        # add special tokens if needed...
        vocab_size = tokenizer.vocab_size
        def encode_fn(txt):
            return tokenizer.encode(txt)
    elif tokenization_type == "char":
        # ...
        pass
    elif tokenization_type == "byte":
        # ...
        pass
    
    # Convert texts to IDs
    train_ids = []
    for t in train_texts:
        train_ids.extend(encode_fn(t))
    val_ids = []
    for t in val_texts:
        val_ids.extend(encode_fn(t))

    # Build dataset & sampler
    from src.data.dataset import TextDataset
    block_size = train_config["block_size"]
    train_dataset = TextDataset(train_ids, block_size)
    val_dataset = TextDataset(val_ids, block_size)

    if ddp:
        # distributed sampler
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_config["batch_size"])
        val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=train_config["batch_size"])
    else:
        train_loader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=train_config["batch_size"], shuffle=False)

    # Initialize model
    from src.model.gpt2 import GPT2Model, GPT2Config
    config = GPT2Config(
        vocab_size=vocab_size,
        n_ctx=model_config_data["n_ctx"],
        n_embd=model_config_data["n_embd"],
        n_layer=model_config_data["n_layer"],
        n_head=model_config_data["n_head"]
    )
    model = GPT2Model(config).to(device)

    # If ddp, wrap model
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[device.index], output_device=device.index)

    # Now train
    from src.training.train import train
    train(model, train_loader, val_loader, device, train_config)

    if local_rank == 0:
        # Usually only rank 0 saves
        os.makedirs("model_checkpoint", exist_ok=True)
        torch.save(model.state_dict(), "model_checkpoint/model.pt")
        print("Model saved by rank 0")

    # Done

if __name__ == "__main__":
    main()




