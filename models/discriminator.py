import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator for 128x128 images.
    Takes concatenated input [sketch, image] with 6 channels.
    Outputs a patch-level real/fake prediction map.
    """
    def __init__(self, in_channels=6, features=64):
        super(PatchGANDiscriminator, self).__init__()
        self.net = nn.Sequential(
            # First conv layer: reduce spatial size to 64x64
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Second conv: 64x64 -> 32x32
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Third conv: 32x32 -> 16x16
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Fourth conv: 16x16 -> 15x15 (stride 1)
            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Final conv: 15x15 -> 14x14 output patch map with 1 channel (real/fake)
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    # Quick test: create dummy input tensor (batch_size=1, 6 channels, 128x128)
    model = PatchGANDiscriminator()
    x = torch.randn(1, 6, 128, 128)
    with torch.no_grad():
        out = model(x)
    print(f"PatchGANDiscriminator output shape: {out.shape}")  # Expected: [1, 1, 14, 14]
