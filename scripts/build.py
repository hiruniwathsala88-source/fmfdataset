import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms



# PATHS


DATASET_DIR = Path(__file__).resolve().parent.parent

SPLIT_FILE = DATASET_DIR / "split.json"
OUTPUT_DIR = DATASET_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



# SETTINGS


IMAGE_SIZE = 224

# Number of augmented versions for each training image
AUGMENTATIONS_PER_IMAGE = 6



# LOAD split.json


with open(SPLIT_FILE, "r") as f:
    split_data = json.load(f)

class_names = split_data["class_names"]

train_paths = split_data["train"]["paths"]
train_labels = split_data["train"]["labels"]

val_paths = split_data["val"]["paths"]
val_labels = split_data["val"]["labels"]


print("Classes:", class_names)
print("Training images:", len(train_paths))
print("Validation images:", len(val_paths))



# TRAIN AUGMENTATION


train_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    # Horizontal flip
    transforms.RandomHorizontalFlip(p=0.5),

    # Rotation ±15 degrees
    transforms.RandomRotation(15),

    # Brightness and contrast
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    # Gaussian blur
    transforms.GaussianBlur(
        kernel_size=3
    ),

    # Perspective
    transforms.RandomPerspective(
        distortion_scale=0.2,
        p=0.5
    ),

    # Convert to Tensor
    transforms.ToTensor()
])



# VALIDATION TRANSFORM


val_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor()
])



# BUILD TRAIN DATA


train_images = []
train_targets = []
train_source_paths = []


print("\nBuilding training dataset...")


for path, label in zip(train_paths, train_labels):

    image_path = Path(path)

    image = Image.open(image_path).convert("RGB")

    # Create 6 augmented versions
    for i in range(AUGMENTATIONS_PER_IMAGE):

        augmented_image = train_transform(image)

        train_images.append(augmented_image)
        train_targets.append(label)
        train_source_paths.append(str(image_path))


print("Training samples created:", len(train_images))


# Convert list to tensor
train_images = torch.stack(train_images)

train_targets = torch.tensor(
    train_targets,
    dtype=torch.long
)



# BUILD VALIDATION DATA


val_images = []
val_targets = []
val_source_paths = []


print("\nBuilding validation dataset...")


for path, label in zip(val_paths, val_labels):

    image_path = Path(path)

    image = Image.open(image_path).convert("RGB")

    # Only resize + tensor
    image_tensor = val_transform(image)

    val_images.append(image_tensor)
    val_targets.append(label)
    val_source_paths.append(str(image_path))


print("Validation samples created:", len(val_images))


# Convert list to tensor
val_images = torch.stack(val_images)

val_targets = torch.tensor(
    val_targets,
    dtype=torch.long
)



# SAVE TRAIN.PT


train_data = {
    "images": train_images,
    "labels": train_targets,
    "paths": train_source_paths,
    "class_names": class_names
}


train_file = OUTPUT_DIR / "train.pt"

torch.save(
    train_data,
    train_file
)



# SAVE VAL.PT


val_data = {
    "images": val_images,
    "labels": val_targets,
    "paths": val_source_paths,
    "class_names": class_names
}


val_file = OUTPUT_DIR / "val.pt"

torch.save(
    val_data,
    val_file
)



# FINAL RESULTS

print("\n================================")
print("TASK 5 COMPLETED")
print("================================")

print("\nTrain:")
print("Images shape:", train_images.shape)
print("Labels shape:", train_targets.shape)
print("Number of samples:", len(train_targets))

print("\nValidation:")
print("Images shape:", val_images.shape)
print("Labels shape:", val_targets.shape)
print("Number of samples:", len(val_targets))

print("\nFiles saved:")

print(train_file)
print(val_file)