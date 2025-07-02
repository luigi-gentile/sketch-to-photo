import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


class SketchPhotoDataset:
    """
    Dataset class for paired sketch and photo images.
    Loads images from 'sketches' and 'photos' folders under a specified split directory.
    """

    def __init__(self, base_path, split='train', transform=None):
        # Path to the folder containing sketches for the chosen split (train/val/test)
        self.sketch_folder = os.path.join(base_path, split, 'sketches')
        # Path to the folder containing photos for the chosen split
        self.photo_folder = os.path.join(base_path, split, 'photos')
        # Transformations to apply to both sketch and photo images (if any)
        self.transform = transform

        # List all image filenames (jpg, jpeg, png) sorted alphabetically in photo folder
        self.image_names = sorted([
            f for f in os.listdir(self.photo_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

    def __len__(self):
        # Return total number of image pairs in the dataset
        return len(self.image_names)

    def __getitem__(self, index):
        # Get the filename of the image at the given index
        name = self.image_names[index]

        # Build full file paths for sketch and photo images
        sketch_path = os.path.join(self.sketch_folder, name)
        photo_path = os.path.join(self.photo_folder, name)

        # Open and convert images to RGB (3-channel)
        sketch_img = Image.open(sketch_path).convert('RGB')
        photo_img = Image.open(photo_path).convert('RGB')

        # Apply the same transform to both sketch and photo if transform is provided
        if self.transform:
            sketch_img = self.transform(sketch_img)
            photo_img = self.transform(photo_img)

        # Return a dictionary with sketch and photo tensors
        return {'sketch': sketch_img, 'photo': photo_img}


def create_dataloader(base_path, split='train', batch_size=16, img_size=256, shuffle=True, workers=0):
    """
    Utility function to create a DataLoader for the SketchPhotoDataset.
    Applies standard resizing and normalization transforms.

    Args:
        base_path (str): Root path to the processed dataset.
        split (str): Dataset split - 'train', 'val', or 'test'.
        batch_size (int): Number of samples per batch.
        img_size (int): Size to resize images (height and width).
        shuffle (bool): Whether to shuffle the data each epoch.
        workers (int): Number of subprocesses to use for data loading.

    Returns:
        DataLoader: PyTorch DataLoader yielding batches of sketch-photo pairs.
    """
    # Compose transformations: resize images, convert to tensor, normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create dataset instance with specified transforms
    dataset = SketchPhotoDataset(base_path, split, transform)

    # Create DataLoader for batching, shuffling, and parallel data loading
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available()  # Optimize GPU data transfer if CUDA available
    )

    return loader


if __name__ == "__main__":
    # Example usage and quick test of the dataloader

    # Define path to processed data
    data_path = "../data/processed"
    # Create a DataLoader for training split with batch size 4 and image size 128x128
    loader = create_dataloader(data_path, split='train', batch_size=4, img_size=128)

    # Iterate through one batch and print shapes of sketch and photo tensors
    for batch in loader:
        print(f"Sketch batch shape: {batch['sketch'].shape}")  # Expected: [batch_size, 3, H, W]
        print(f"Photo batch shape: {batch['photo'].shape}")
        break
