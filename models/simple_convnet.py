import torch.nn as nn
import torch.nn.functional as F

class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        
        # Encoder: downsample from 1024x1024 to 256x256
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 1024 -> 512
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 512 -> 256
        )
        
        # Decoder: upsample from 256x256 back to 1024x1024
        self.up1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, 
                                      kernel_size=2, stride=2)  # 256 -> 512
        self.up2 = nn.ConvTranspose2d(in_channels=32, out_channels=5,  # output 5 channels for 4 classes
                                      kernel_size=2, stride=2)  # 512 -> 1024
        
    
    def forward(self, x):
        # [batch, 12, 1024, 1024]
        x = self.down1(x)    # [batch, 32, 512, 512]
        x = self.down2(x)    # [batch, 64, 256, 256]
        x = F.relu(self.up1(x))  # [batch, 32, 512, 512]
        x = self.up2(x)      # [batch, 5, 1024, 1024]
        return x
