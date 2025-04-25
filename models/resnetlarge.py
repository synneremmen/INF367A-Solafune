import torch.nn as nn
import torchvision.models as models

class UNetResNet18Large(nn.Module):
    def __init__(self, num_classes=5, in_channels=12):
        super(UNetResNet18Large, self).__init__()
        
        # Load ResNet18 model
        self.encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze early layers
        for param in self.encoder.conv1.parameters():
            param.requires_grad = False  # Freeze initial conv layer

        for param in self.encoder.bn1.parameters():
            param.requires_grad = False  # Freeze first batch norm

        for param in self.encoder.layer1.parameters():
            param.requires_grad = False  # Freeze first ResNet block
        
        # Modify first conv layer to accept 12 input channels
        self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        
        # Decoder (simple upsampling with convolutional layers)
        self.up1 = nn.ConvTranspose2d(512, 64, kernel_size=2, stride=2)  # Reduced channels
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # Reduced channels
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  # Reduced channels
        self.up4 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)   # Reduced channels
        
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        
        # Final segmentation layer
        self.final_conv = nn.Conv2d(8, num_classes, kernel_size=1)  # Reduced channels
        
    def forward(self, x):
        # Encoder
        x1 = self.encoder.conv1(x)
        x1 = self.encoder.bn1(x1)
        x1 = self.encoder.relu(x1)
        x1 = self.encoder.maxpool(x1)
        
        x2 = self.encoder.layer1(x1)
        x3 = self.encoder.layer2(x2)
        x4 = self.encoder.layer3(x3)
        x5 = self.encoder.layer4(x4)
        
        # Decoder with skip connections
        d4 = self.up1(x5)  # 512 -> 64
        d4 = self.conv1(d4)
        
        d3 = self.up2(d4)  # 64 -> 32
        d3 = self.conv2(d3)
        
        d2 = self.up3(d3)  # 32 -> 16
        d2 = self.conv3(d2)
        
        d1 = self.up4(d2)  # 16 -> 8
        d1 = self.conv4(d1)
        
        # Final classification layer
        out = self.final_conv(d1)
        
        return out

# model = UNetResNet18Small(num_classes=5)
# total_params = sum(p.numel() for p in model.parameters())
# print(f'Total number of parameters to train: {total_params}')

# Total number of parameters to train: 11 908 805