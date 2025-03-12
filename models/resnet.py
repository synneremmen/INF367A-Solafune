import torch.nn as nn
import torchvision.models as models

class UNetResNet18(nn.Module):
    def __init__(self, num_classes=5, in_channels=12):
        super(UNetResNet18, self).__init__()
        
        # Load ResNet18 model
        self.encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # if we want to freeze all layers
        # for param in self.encoder.parameters():
        #    param.requires_grad = False

        # Freeze early layers
        for param in self.encoder.conv1.parameters():
            param.requires_grad = False  # Freeze initial conv layer

        for param in self.encoder.bn1.parameters():
            param.requires_grad = False  # Freeze first batch norm

        for param in self.encoder.layer1.parameters():
            param.requires_grad = False  # Freeze first ResNet block
        
        # Modify first conv layer to accept 12 input channels
        self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        #self.encoder.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Decoder (simple upsampling with convolutional layers)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        #self.up5 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        #self.conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        
        # Final segmentation layer
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)     # 16 -> 32
        
    def forward(self, x):
        # Encoder
        x1 = self.encoder.conv1(x)  # Get input to decoder on the right shape: from [batch_size, 12, 1024, 1024] to [batch_size, 64, 1024, 1024]
        x1 = self.encoder.bn1(x1)
        x1 = self.encoder.relu(x1)
        x1 = self.encoder.maxpool(x1)
        
        x2 = self.encoder.layer1(x1)
        x3 = self.encoder.layer2(x2)
        x4 = self.encoder.layer3(x3)
        x5 = self.encoder.layer4(x4)
        
        # Decoder with skip connections
        d4 = self.up1(x5)  # 512 -> 256
        d4 = self.conv1(d4)
        
        d3 = self.up2(d4)  # 256 -> 128
        d3 = self.conv2(d3)
        
        d2 = self.up3(d3)  # 128 -> 64
        d2 = self.conv3(d2)
        
        d1 = self.up4(d2)  # 64 -> 32
        d1 = self.conv4(d1)
        
        # Final classification layer
        out = self.final_conv(d1)
        
        return out

# model = UNetResNet18(num_classes=5)
# total_params = sum(p.numel() for p in model.parameters())
# print(f'Total number of parameters to train: {total_params}')
# 