import time
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from load import FaceDataset



# PATHS


DATASET_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = DATASET_DIR / "processed" / "train.pt"



# LOAD DATASET


dataset = FaceDataset(TRAIN_FILE)



# TEST BATCH SIZES


batch_sizes = [8, 64]



print("BATCH SIZE COMPARISON")

print(f"Total training samples: {len(dataset)}")


for batch_size in batch_sizes:

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    start_time = time.time()

    batch_count = 0

    for images, labels, paths in loader:

        batch_count += 1

    end_time = time.time()

    elapsed_time = end_time - start_time

  
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {batch_count}")
    print(f"Time taken: {elapsed_time:.4f} seconds")



print("COMPARISON")


print("""
Batch size 8:
- Uses less memory
- More batches
- Suitable for systems with lower memory

Batch size 64:
- Uses more memory
- Fewer batches
- Can process more samples at once
- Requires more memory
""")