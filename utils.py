import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_ch, out_ch):
    """Two convolutions with batch norm and ReLU"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),  # enc1.0
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),          # enc1.2
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(64)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),         # enc2.0
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),        # enc2.2
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(128)
        )
        
        self.conv_btlnk = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),        # conv_btlnk.0
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),        # conv_btlnk.2
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(256)
        )
        
        # Decoder with skip connections
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)  # upconv2
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),        # dec2.0 (256 input from concat)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),        # dec2.2
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(128)
        )
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)  # upconv1
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),         # dec1.0 (128 input from concat)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),          # dec1.2
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(64)
        )
        
        self.final_conv = nn.Conv2d(64, out_channels, 1)  # final_conv
        
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)           # 128x128x64
        x = self.pool(enc1)           # 64x64x64
        
        enc2 = self.enc2(x)           # 64x64x128
        x = self.pool(enc2)           # 32x32x128
        
        x = self.conv_btlnk(x)        # 32x32x256
        
        # Decoder with skip connections
        x = self.upconv2(x)           # 64x64x128
        x = torch.cat([x, enc2], dim=1)  # 64x64x256
        x = self.dec2(x)              # 64x64x128
        
        x = self.upconv1(x)           # 128x128x64
        x = torch.cat([x, enc1], dim=1)  # 128x128x128
        x = self.dec1(x)              # 128x128x64
        
        x = self.final_conv(x)        # 128x128x1
        
        return x
