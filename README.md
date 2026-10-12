# Face Recognition Dataset (PyTorch)

Deep Learning module project: preparing a face dataset with 3 classes (`aroshi`, `hiruni`, `unknown`) and loading it with PyTorch.

## What I did
- Collected face photos with a phone camera and cropped the faces using OpenCV (Haar Cascade)
- Augmented the training images (flip, rotation, brightness, contrast, blur, perspective)
- Split the data 80/20 into training and validation sets
- Built `.pt` tensor datasets (224 x 224 images)
- Analysed the dataset and created a custom PyTorch `Dataset` and `DataLoader`

## Dataset
| Item | Value |
|---|---|
| Raw images | 51 (17 per class) |
| Training samples | 240 (40 images x 6 augmentations) |
| Validation samples | 11 |

## Files
- `scripts/`: Python scripts
- `config.json`: class names
- `split.json`: train/validation split
- `Face Recognition Report.pdf`: full lab report


