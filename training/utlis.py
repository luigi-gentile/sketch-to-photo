import torch
import os
import random
import numpy as np

def save_checkpoint(model, optimizer, epoch, path):
    """
    Save a checkpoint containing the model and optimizer states.

    Args:
        model (torch.nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer to save.
        epoch (int): Current epoch number.
        path (str): File path to save the checkpoint.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    print(f"[INFO] Checkpoint saved at epoch {epoch} to {path}")

def load_checkpoint(model, optimizer, path, device):
    """
    Load a checkpoint and update model and optimizer states.

    Args:
        model (torch.nn.Module): Model to load the weights into.
        optimizer (torch.optim.Optimizer): Optimizer to load the state into.
        path (str): Checkpoint file path.
        device (torch.device): Device to load the checkpoint on.

    Returns:
        int: The epoch number to resume from.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"[INFO] Checkpoint loaded from {path}, resuming at epoch {epoch}")
    return epoch

def set_seed(seed=42):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Seed set to {seed}")

def progress_bar(current, total, bar_length=40):
    """
    Display a textual progress bar in the console.

    Args:
        current (int): Current progress (e.g. current batch).
        total (int): Total steps (e.g. total batches).
        bar_length (int): Length of the progress bar.
    """
    fraction = current / total
    filled_length = int(bar_length * fraction)
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    print(f'\rProgress: |{bar}| {current}/{total} ({fraction*100:.1f}%)', end='')
    if current == total:
        print()  # newline

if __name__ == "__main__":
    # Quick test of the utility functions
    set_seed(123)

    # Dummy model and optimizer for checkpoint testing
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters())

    save_path = "../checkpoints/test_checkpoint.pth"

    # Save checkpoint
    save_checkpoint(model, optimizer, epoch=5, path=save_path)

    # Load checkpoint
    epoch = load_checkpoint(model, optimizer, save_path, device=torch.device('cpu'))

    # Test progress bar
    import time
    for i in range(41):
        progress_bar(i, 40)
        time.sleep(0.05)
