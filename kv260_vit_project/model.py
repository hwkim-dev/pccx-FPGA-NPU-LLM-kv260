import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleViTEnhancer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, embed_dim=64, num_heads=4, num_layers=2):
        super(SimpleViTEnhancer, self).__init__()

        # Encoder
        # Downsample 1: (B, C, H, W) -> (B, embed_dim, H/2, W/2)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(embed_dim)
        )

        # Downsample 2: (B, embed_dim, H/2, W/2) -> (B, embed_dim*2, H/4, W/4)
        self.encoder2 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(embed_dim * 2)
        )

        self.bottleneck_dim = embed_dim * 2

        # Transformer Bottleneck
        # Input to transformer: (B, L, D) where L = (H/4 * W/4), D = bottleneck_dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.bottleneck_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder
        # Upsample 1: (B, bottleneck_dim, H/4, W/4) -> (B, embed_dim, H/2, W/2)
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(self.bottleneck_dim, embed_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(embed_dim)
        )

        # Upsample 2: (B, embed_dim, H/2, W/2) -> (B, out_channels, H, W)
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Output the image directly in [0, 1]
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)

        # Prepare for Transformer
        B, C2, H2, W2 = enc2.shape
        feat_flat = enc2.flatten(2).transpose(1, 2)

        # Transformer
        feat_trans = self.transformer_encoder(feat_flat)

        # Reshape back
        feat_reshaped = feat_trans.transpose(1, 2).view(B, C2, H2, W2)

        # Decoder
        dec1 = self.decoder1(feat_reshaped)
        # Skip connection from encoder1
        if dec1.shape == enc1.shape:
             dec1 = dec1 + enc1

        out = self.decoder2(dec1)

        return out

if __name__ == "__main__":
    # Simple test
    model = SimpleViTEnhancer()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
