import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt


class FaceAugmentor:

    def __init__(self):

        self.augmentations = transforms.Compose([
            transforms.Resize((224, 224)),

            # Horizontal Flip
            transforms.RandomHorizontalFlip(p=0.5),

            # Rotation ±15 degrees
            transforms.RandomRotation(15),

            # Brightness and Contrast
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2
            ),

            # Gaussian Blur
            transforms.GaussianBlur(
                kernel_size=3
            ),

            # Perspective transformation
            transforms.RandomPerspective(
                distortion_scale=0.2,
                p=0.5
            ),

            # Convert image to Tensor
            transforms.ToTensor()
        ])

    def augment_image(self, image):
        return self.augmentations(image)

    def get_random_augmentations(self, image, n=6):

        augmented_images = []

        for _ in range(n):
            augmented = self.augment_image(image)
            augmented_images.append(augmented)

        return augmented_images

    def visualize(self, image_path):

        image = Image.open(image_path).convert("RGB")

        augmented_images = self.get_random_augmentations(
            image,
            n=6
        )

        plt.figure(figsize=(12, 8))

        # Original image
        plt.subplot(2, 4, 1)
        plt.imshow(image)
        plt.title("Original")
        plt.axis("off")

        # Augmented images
        for i, img in enumerate(augmented_images):

            plt.subplot(2, 4, i + 2)

            img = img.permute(1, 2, 0)

            plt.imshow(img)

            plt.title(f"Augmented {i + 1}")
            plt.axis("off")

        plt.tight_layout()

        output_path = (
            Path(__file__).resolve().parent.parent
            / "augmentation_visualization.png"
        )

        plt.savefig(output_path)

        print(f"Visualization saved to:")
        print(output_path)

        plt.show()



# TEST AUGMENTATION


if __name__ == "__main__":

    DATASET_DIR = Path(__file__).resolve().parent.parent

    # Use one existing image
    image_folder = DATASET_DIR / "raw" / "aroshi"

    images = list(image_folder.glob("*.jpg"))

    if len(images) == 0:
        images = list(image_folder.glob("*.jpeg"))

    if len(images) == 0:
        images = list(image_folder.glob("*.png"))

    if len(images) == 0:
        print("No images found in aroshi folder.")
        exit()

    image_path = images[0]

    print("Using image:")
    print(image_path)

    augmentor = FaceAugmentor()

    augmentor.visualize(image_path)