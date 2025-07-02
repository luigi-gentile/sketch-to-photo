import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import math


class SketchPhotoDataset:
    """
    Dataset class for paired sketch and photo images.
    Loads images from 'sketches' and 'photos' folders under a specified split directory.
    """

    def __init__(self, base_path, split='train', transform=None):
        self.sketch_folder = os.path.join(base_path, split, 'sketches')
        self.photo_folder = os.path.join(base_path, split, 'photos')
        self.transform = transform

        self.image_names = sorted([
            f for f in os.listdir(self.photo_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        name = self.image_names[index]

        sketch_path = os.path.join(self.sketch_folder, name)
        photo_path = os.path.join(self.photo_folder, name)

        sketch_img = Image.open(sketch_path).convert('RGB')
        photo_img = Image.open(photo_path).convert('RGB')

        if self.transform:
            sketch_img = self.transform(sketch_img)
            photo_img = self.transform(photo_img)

        return {'sketch': sketch_img, 'photo': photo_img}


def create_dataloader(base_path, split='train', batch_size=16, img_size=256, shuffle=True, workers=0, subset_fraction=1.0):
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
        subset_fraction (float): Fraction of dataset to load (0 < subset_fraction <= 1).

    Returns:
        DataLoader: PyTorch DataLoader yielding batches of sketch-photo pairs.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = SketchPhotoDataset(base_path, split, transform)

    if subset_fraction < 1.0:
        subset_len = math.floor(len(dataset) * subset_fraction)
        indices = list(range(subset_len))
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available()
    )

    return loader


if __name__ == "__main__":
    data_path = "../data/processed"
    loader = create_dataloader(data_path, split='train', batch_size=4, img_size=128, subset_fraction=0.25)

    for batch in loader:
        print(f"Sketch batch shape: {batch['sketch'].shape}")
        print(f"Photo batch shape: {batch['photo'].shape}")
        break
