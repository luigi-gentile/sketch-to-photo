import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid, save_image
import numpy as np

def show_images(images, nrow=4, title=None):
    """
    Visualizza una griglia di immagini.
    images: Tensor di shape (B, C, H, W)
    """
    grid_img = make_grid(images, nrow=nrow, normalize=True, scale_each=True)
    npimg = grid_img.cpu().numpy()
    plt.figure(figsize=(10, 5))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def save_sample_images(sketches, fake_photos, filename, nrow=4):
    """
    Salva le immagini generate insieme agli sketch come un'unica immagine.
    """
    sample_grid = torch.cat([sketches, fake_photos], dim=0)
    save_image(sample_grid * 0.5 + 0.5, filename, nrow=nrow)
    print(f"Saved sample image to {filename}")

if __name__ == "__main__":
    print("Eseguo test visualizzazione immagini...")

    fake_images = torch.randn(8, 3, 128, 128)

    show_images(fake_images, nrow=4, title="Test immagini random")
    save_sample_images(fake_images, fake_images, "results/test_sample.png", nrow=4)
    print("Immagini di test salvate in results/test_sample.png")
