import torch
import torch.nn as nn

class GANLoss(nn.Module):
    """
    Adversarial loss for GAN training.
    Uses BCEWithLogitsLoss for stability.
    """

    def __init__(self):
        super(GANLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, preds, target_is_real):
        """
        preds: output del discriminator (logits)
        target_is_real: bool, True se l’input è reale, False se fake
        """
        target = torch.ones_like(preds) if target_is_real else torch.zeros_like(preds)
        return self.loss(preds, target)


def l1_loss(predicted, target):
    """
    L1 loss between predicted and target images.
    """
    return torch.mean(torch.abs(predicted - target))


if __name__ == "__main__":
    # Test rapido
    gan_loss = GANLoss()

    # batch size 2, output patch 14x14
    fake_preds = torch.randn(2, 1, 14, 14)
    real_preds = torch.randn(2, 1, 14, 14)

    # Adversarial loss: discriminator real should be close to 1, fake close to 0
    loss_real = gan_loss(real_preds, True)
    loss_fake = gan_loss(fake_preds, False)
    print(f"GANLoss real: {loss_real.item():.4f}, fake: {loss_fake.item():.4f}")

    # L1 loss test: two random tensors
    pred_img = torch.randn(2, 3, 128, 128)
    target_img = torch.randn(2, 3, 128, 128)
    l1 = l1_loss(pred_img, target_img)
    print(f"L1 loss: {l1.item():.4f}")
