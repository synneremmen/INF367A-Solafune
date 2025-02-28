import torch.nn as nn
import torch.nn.functional as F
import torch

class encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()

        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
    def forward(self, x, skip_features):
        x = self.conv_transpose(x)
        skip_features = F.interpolate(skip_features, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        x = torch.cat([x, skip_features], dim=1) # applying skip connection
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.enc1 = encoder(12, 32)
        self.enc2 = encoder(32, 64)
        self.enc3 = encoder(64, 128)
        self.enc4 = encoder(128, 256)
        
        self.middle = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU()
        )

        self.dec1 = decoder(256, 128)
        self.dec2 = decoder(128, 64)
        self.dec3 = decoder(64, 32)
        self.dec4 = decoder(32, 12)

        self.out = nn.Sequential(
            nn.Conv2d(12, 5, kernel_size=3, padding=1),
            nn.ReLU()
        )


    def forward(self, x):
        # Encoder 
        # [batch, 12, 1024, 1024]
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3) 

        # A few layers after encoding
        x = self.middle(e4)

        # Decoder 
        x = self.dec1(x, e4) 
        x = self.dec2(x, e3) 
        x = self.dec3(x, e2) 
        x = self.dec4(x, e1)
        
        x = self.out(x) # får se om vi trenger eller kaster denne
        return x # [batch, 5, 1024, 1024]
    
"""
To calculate the number of parameters in the UNet model, we need to sum up the parameters of each layer in the model. Here's a step-by-step breakdown:

Encoder Layers:

encoder(12, 32):
Conv2d: (12 * 32 * 3 * 3) + 32 = 3488
Conv2d: (32 * 32 * 3 * 3) + 32 = 9248
encoder(32, 64):
Conv2d: (32 * 64 * 3 * 3) + 64 = 18496
Conv2d: (64 * 64 * 3 * 3) + 64 = 36928
encoder(64, 128):
Conv2d: (64 * 128 * 3 * 3) + 128 = 73856
Conv2d: (128 * 128 * 3 * 3) + 128 = 147584
encoder(128, 256):
Conv2d: (128 * 256 * 3 * 3) + 256 = 295168
Conv2d: (256 * 256 * 3 * 3) + 256 = 590080
Middle Layers:

Conv2d: (256 * 512 * 3 * 3) + 512 = 1179648
Conv2d: (512 * 256 * 3 * 3) + 256 = 1179904

Decoder Layers:

decoder(256, 128):
ConvTranspose2d: (256 * 128 * 2 * 2) + 128 = 131200
Conv2d: (384 * 256 * 3 * 3) + 256 = 884992
Conv2d: (256 * 128 * 3 * 3) + 128 = 295040
decoder(128, 64):
ConvTranspose2d: (128 * 64 * 2 * 2) + 64 = 32832
Conv2d: (192 * 128 * 3 * 3) + 128 = 221312
Conv2d: (128 * 64 * 3 * 3) + 64 = 73792
decoder(64, 32):
ConvTranspose2d: (64 * 32 * 2 * 2) + 32 = 8224
Conv2d: (96 * 64 * 3 * 3) + 64 = 55360
Conv2d: (64 * 32 * 3 * 3) + 32 = 18464
decoder(32, 12):
ConvTranspose2d: (32 * 12 * 2 * 2) + 12 = 3084
Conv2d: (44 * 32 * 3 * 3) + 32 = 12736
Conv2d: (32 * 12 * 3 * 3) + 12 = 3468

Output Layer:
Conv2d: (12 * 5 * 3 * 3) + 5 = 650

This gives a total of 5,852,286 parameters in the UNet model.
"""