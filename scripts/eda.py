import torch
from pathlib import Path
import matplotlib.pyplot as plt
import random



# PATHS


DATASET_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = DATASET_DIR / "processed" / "train.pt"
VAL_FILE = DATASET_DIR / "processed" / "val.pt"



# LOAD DATA


train_data = torch.load(TRAIN_FILE)
val_data = torch.load(VAL_FILE)


train_images = train_data["images"]
train_labels = train_data["labels"]
class_names = train_data["class_names"]

val_images = val_data["images"]
val_labels = val_data["labels"]



# PRINT SHAPES



print("TASK 6 - EDA")


print("\nTrain image shape:")
print(train_images.shape)

print("\nTrain label shape:")
print(train_labels.shape)

print("\nValidation image shape:")
print(val_images.shape)

print("\nValidation label shape:")
print(val_labels.shape)



# CLASS DISTRIBUTION



print("CLASS DISTRIBUTION")


for class_id, class_name in enumerate(class_names):

    train_count = (train_labels == class_id).sum().item()
    val_count = (val_labels == class_id).sum().item()

    print(
        f"{class_name}: "
        f"Train = {train_count}, "
        f"Validation = {val_count}"
    )



# BAR CHART


train_counts = []

val_counts = []

for class_id in range(len(class_names)):

    train_counts.append(
        (train_labels == class_id).sum().item()
    )

    val_counts.append(
        (val_labels == class_id).sum().item()
    )


x = range(len(class_names))

plt.figure(figsize=(8, 5))

plt.bar(
    [i - 0.2 for i in x],
    train_counts,
    width=0.4,
    label="Train"
)

plt.bar(
    [i + 0.2 for i in x],
    val_counts,
    width=0.4,
    label="Validation"
)

plt.xticks(
    list(x),
    class_names
)

plt.xlabel("Class")
plt.ylabel("Number of Images")

plt.title("Train vs Validation Class Distribution")

plt.legend()

plt.tight_layout()


chart_path = DATASET_DIR / "class_distribution.png"

plt.savefig(chart_path)

print("\nBar chart saved to:")
print(chart_path)

plt.show()



# RANDOM 16 IMAGES


print("\nCreating random image grid...")


num_images = min(16, len(train_images))

indices = random.sample(
    range(len(train_images)),
    num_images
)


plt.figure(figsize=(12, 10))


for i, index in enumerate(indices):

    image = train_images[index]

    label = train_labels[index].item()

    # Tensor: [3, 224, 224]
    # Convert to: [224, 224, 3]

    image = image.permute(1, 2, 0)

    plt.subplot(4, 4, i + 1)

    plt.imshow(image)

    plt.title(class_names[label])

    plt.axis("off")


plt.tight_layout()


grid_path = DATASET_DIR / "random_16_images.png"

plt.savefig(grid_path)

print("Random image grid saved to:")
print(grid_path)

plt.show()