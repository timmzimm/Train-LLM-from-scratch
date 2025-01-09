import os
import json
import torch
import torch.distributed as dist

from src.training.pipeline import run_training_pipeline_single, run_training_pipeline_ddp

def main():
    """
    Entry point. Reads train_config and decides if we do single-GPU/CPU or DDP.
    """
    with open("config/training_config.json", "r") as f:
        train_config = json.load(f)

    distributed = train_config.get("distributed", False)
    if distributed:
        # Launch multi-GPU with PyTorch DDP
        world_size = len(train_config["gpu_ids"])
        ddp_main(world_size, train_config)
    else:
        # Single-GPU or CPU fallback
        run_training_pipeline_single(train_config)

def ddp_main(world_size, train_config):
    """
    Initializes the process group, runs the pipeline on each process, then cleans up.
    We assume you're calling:
      torchrun --nproc_per_node={world_size} main.py
    """
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    run_training_pipeline_ddp(train_config, local_rank)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()





