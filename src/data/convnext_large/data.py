import datasets
import src.config as config
import torch.utils.data
import os
from huggingface_hub import hf_hub_download

DATASET_NAME = "InfImagine/FakeImageDataset"
CACHE_DIR = "/home/slurm/comp597/students/afuad/.cache/huggingface"
# The base path for the shards
SHARD_BASE = "ImageData/val/IF-CC95K/IF-CC95K.tar.gz"
# There are 4 shards: .0, .1, .2, .3
NUM_SHARDS = 4 

def load_data(conf: config.Config) -> torch.utils.data.Dataset:
    print(f"Loading FakeImageDataset...")
    
    # 1. Download all shards
    shard_paths = []
    for i in range(NUM_SHARDS):
        filename = f"{SHARD_BASE}.{i}"
        print(f"Downloading shard {i+1}/{NUM_SHARDS}: {filename}")
        path = hf_hub_download(
            repo_id=DATASET_NAME,
            filename=filename,
            repo_type="dataset",
            cache_dir=CACHE_DIR
        )
        shard_paths.append(path)

    # 2. Merge shards into a single tar file if not already done
    # We place the merged file in the CACHE_DIR to avoid re-merging
    merged_tar_path = os.path.join(CACHE_DIR, "IF-CC95K_merged.tar.gz")
    
    if not os.path.exists(merged_tar_path):
        print(f"Merging shards into {merged_tar_path}...")
        with open(merged_tar_path, 'wb') as outfile:
            for shard in shard_paths:
                with open(shard, 'rb') as infile:
                    # Use a buffer to handle large file merge efficiently
                    while chunk := infile.read(1024 * 1024): 
                        outfile.write(chunk)
        print("Merge complete.")
    else:
        print("Merged tar already exists, skipping merge.")

    # 3. Load using webdataset loader
    try:
        print("Loading webdataset from merged tar file...")
        dataset = datasets.load_dataset(
            "webdataset",
            data_files={"train": merged_tar_path},
            split="train",
            cache_dir=CACHE_DIR,
        )
        
        print(f"Successfully loaded {len(dataset)} samples")
        return dataset
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise