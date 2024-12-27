import json
import math
from datasets import load_dataset, concatenate_datasets

def load_and_merge_splits(config_path: str):
    """
    Loads the orca-agentinstruct dataset from Hugging Face.
    Splits each subset (from 'all_splits') into train/val/test
    in the ratio (train_ratio, val_ratio, test_ratio),
    then concatenates them across all subsets.

    Returns:
        train_ds, val_ds, test_ds (huggingface Datasets)
    """
    with open(config_path, "r") as f:
        dataset_config = json.load(f)

    dataset_name = dataset_config["dataset_name"]
    all_splits = dataset_config["all_splits"]
    train_ratio = dataset_config["train_ratio"]
    val_ratio = dataset_config["val_ratio"]
    test_ratio = dataset_config["test_ratio"]

    # Sanity check
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-7:
        raise ValueError("Sum of train_ratio, val_ratio, and test_ratio must be 1.0")

    # Load the entire dataset
    print(f"Loading entire dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)

    train_sets = []
    val_sets = []
    test_sets = []

    for split_name in all_splits:
        if split_name not in dataset:
            print(f"Warning: split '{split_name}' not found. Skipping.")
            continue

        ds_current = dataset[split_name]
        # 1) first "cut" test portion
        test_split = ds_current.train_test_split(test_size=test_ratio, seed=42)
        test_ds = test_split["test"]
        rest_ds = test_split["train"]

        # 2) now from rest, cut val portion
        # val_ratio is fraction of total, so relative fraction from rest is val_ratio / (train_ratio + val_ratio)
        total_tv = train_ratio + val_ratio
        if abs(total_tv) < 1e-9:
            raise ValueError("train_ratio + val_ratio cannot be 0.")
        relative_val_ratio = val_ratio / total_tv

        tv_split = rest_ds.train_test_split(test_size=relative_val_ratio, seed=42)
        cur_train = tv_split["train"]
        cur_val = tv_split["test"]

        train_sets.append(cur_train)
        val_sets.append(cur_val)
        test_sets.append(test_ds)

    if not train_sets:
        raise ValueError("No valid splits found. Check your all_splits in dataset_config.")

    # Combine
    train_ds = concatenate_datasets(train_sets)
    val_ds = concatenate_datasets(val_sets)
    test_ds = concatenate_datasets(test_sets)

    print(f"Final dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_ds, val_ds, test_ds
