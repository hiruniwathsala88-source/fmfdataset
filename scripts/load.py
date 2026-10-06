import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path



# PATH


DATASET_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = DATASET_DIR / "processed" / "train.pt"
VAL_FILE = DATASET_DIR / "processed" / "val.pt"



# IMAGE NORMALIZATION


imagenet_normalization = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)



# DATASET CLASS


class FaceDataset(Dataset):

    def __init__(self, pt_file):

        data = torch.load(pt_file)

        self.images = data["images"]
        self.labels = data["labels"]
        self.paths = data["paths"]
        self.class_names = data["class_names"]

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        image = self.images[index]

        label = self.labels[index]

        path = self.paths[index]

        # ImageNet normalization
        image = imagenet_normalization(image)

        return image, label, path



# CREATE DATALOADERS


def create_dataloaders(batch_size=8):

    train_dataset = FaceDataset(TRAIN_FILE)

    val_dataset = FaceDataset(VAL_FILE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader



# TEST


if __name__ == "__main__":

    train_loader, val_loader = create_dataloaders(
        batch_size=8
    )

    
    print("TASK 7")
    

    print("\nTrain dataset size:")
    print(len(train_loader.dataset))

    print("\nValidation dataset size:")
    print(len(val_loader.dataset))

    print("\nBatch size:")
    print(train_loader.batch_size)

    # Get one batch
    images, labels, paths = next(iter(train_loader))

    print("\nOne training batch:")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

    print("\nFirst image path:")
    print(paths[0])