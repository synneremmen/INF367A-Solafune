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
    
# model = UNet()
# total_params = sum(p.numel() for p in model.parameters())
# print(f'Total number of parameters: {total_params}')

# Params: 5274393