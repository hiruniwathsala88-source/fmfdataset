import json
import random
from pathlib import Path
from sklearn.model_selection import train_test_split



# PATHS


DATASET_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = DATASET_DIR / "raw"
CONFIG_FILE = DATASET_DIR / "config.json"
OUTPUT_FILE = DATASET_DIR / "split.json"



# RANDOM SEED


SEED = 42

random.seed(SEED)



# LOAD CLASS NAMES


with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

class_names = config["class_names"]

print("Classes:")
print(class_names)



# COLLECT IMAGE PATHS AND LABELS


image_paths = []
labels = []


for label, class_name in enumerate(class_names):

    class_folder = RAW_DIR / class_name

    images = []

    images.extend(class_folder.glob("*.jpg"))
    images.extend(class_folder.glob("*.jpeg"))
    images.extend(class_folder.glob("*.png"))

    print(f"{class_name}: {len(images)} images")

    for image_path in images:

        image_paths.append(str(image_path))
        labels.append(label)



# TRAIN / VALIDATION SPLIT


train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.20,
    random_state=SEED,
    stratify=labels
)



# SAVE SPLIT


split_data = {

    "train": {
        "paths": train_paths,
        "labels": train_labels
    },

    "val": {
        "paths": val_paths,
        "labels": val_labels
    },

    "class_names": class_names
}


with open(OUTPUT_FILE, "w") as f:

    json.dump(
        split_data,
        f,
        indent=4
    )



# RESULTS


print("\n-----------------------------")
print("TRAIN / VALIDATION SPLIT")
print("-----------------------------")

print(f"Total images      : {len(image_paths)}")
print(f"Training images   : {len(train_paths)}")
print(f"Validation images : {len(val_paths)}")

print("\nSplit file saved to:")
print(OUTPUT_FILE)