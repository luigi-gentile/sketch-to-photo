import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(UNetGenerator, self).__init__()
        # Encoder (downsampling) layers
        # The first down block doesn't apply normalization
        self.down1 = self.contract_block(in_channels, features, normalize=False)    # 128 -> 64
        self.down2 = self.contract_block(features, features * 2)                   # 64 -> 32
        self.down3 = self.contract_block(features * 2, features * 4)               # 32 -> 16
        self.down4 = self.contract_block(features * 4, features * 8)               # 16 -> 8
        self.down5 = self.contract_block(features * 8, features * 8)               # 8 -> 4
        self.down6 = self.contract_block(features * 8, features * 8)               # 4 -> 2

        # Bottleneck layer: compress 2x2 to 1x1 feature map
        self.bottleneck = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(features * 8, features * 8, kernel_size=4, stride=2, padding=1, bias=False),  # 2->1
            nn.ReLU(inplace=False),
        )

        # Decoder (upsampling) layers with skip connections
        # Some have dropout for regularization
        self.up1 = self.expand_block(features * 8, features * 8, dropout=True)  # 1 -> 2
        self.up2 = self.expand_block(features * 16, features * 8, dropout=True) # 2 -> 4
        self.up3 = self.expand_block(features * 16, features * 8, dropout=True) # 4 -> 8
        self.up4 = self.expand_block(features * 16, features * 4)               # 8 -> 16
        self.up5 = self.expand_block(features * 8, features * 2)                # 16 -> 32
        self.up6 = self.expand_block(features * 4, features)                    # 32 -> 64

        # Final layer upsamples to output size (128x128) and outputs RGB image with Tanh activation
        self.final_up = nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1)
        self.final_act = nn.Tanh()

    def contract_block(self, in_channels, out_channels, normalize=True):
        """
        Downsampling block: Conv2d -> BatchNorm (optional) -> LeakyReLU.
        Reduces spatial size by factor 2.
        """
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        # Changed inplace=False
        layers.append(nn.LeakyReLU(0.2, inplace=False))
        return nn.Sequential(*layers)

    def expand_block(self, in_channels, out_channels, dropout=False):
        """
        Upsampling block: ConvTranspose2d -> BatchNorm -> ReLU -> Dropout (optional).
        Increases spatial size by factor 2.
        """
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # Changed inplace=False
            nn.ReLU(inplace=False)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encode input through downsampling blocks
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        # Bottleneck representation
        bottleneck = self.bottleneck(d6)

        # Decode and upsample, concatenating skip connections from encoder layers
        up1 = self.up1(bottleneck)
        up1 = torch.cat([up1, d6], dim=1)

        up2 = self.up2(up1)
        up2 = torch.cat([up2, d5], dim=1)

        up3 = self.up3(up2)
        up3 = torch.cat([up3, d4], dim=1)

        up4 = self.up4(up3)
        up4 = torch.cat([up4, d3], dim=1)

        up5 = self.up5(up4)
        up5 = torch.cat([up5, d2], dim=1)

        up6 = self.up6(up5)
        up6 = torch.cat([up6, d1], dim=1)

        # Final output layer to produce RGB image with tanh activation [-1,1]
        out = self.final_up(up6)
        out = self.final_act(out)

        return out


if __name__ == "__main__":
    # Quick test: create dummy input tensor (batch_size=1, 3 channels, 128x128)
    model = UNetGenerator()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    print(f"UNetGenerator output shape: {out.shape}")  # Expected: [1, 3, 128, 128]
