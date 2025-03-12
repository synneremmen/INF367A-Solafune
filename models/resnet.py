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
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x1 = self.encoder.conv1(x)  # Get input to decoder on the right shape: from [batch_size, 12, 1024, 1024] to [batch_size, 64, 1024, 1024]
        x1 = self.encoder.bn1(x1)
        x1 = self.encoder.relu(x1)
        x1 = self.encoder.maxpool(x1)
        print("x1",x1.shape)
        
        x2 = self.encoder.layer1(x1)
        print("x2",x2.shape)
        x3 = self.encoder.layer2(x2)
        print("x3",x3.shape)
        x4 = self.encoder.layer3(x3)
        print("x4",x4.shape)
        x5 = self.encoder.layer4(x4)
        print("x5",x5.shape)
        
        # Decoder with skip connections
        d4 = self.up1(x5)  # 512 -> 256
        print("d4",d4.shape)
        d4 = self.conv1(d4)
        print("d4",d4.shape)
        
        d3 = self.up2(d4)  # 256 -> 128
        print("d3",d3.shape)
        d3 = self.conv2(d3)
        print("d3",d3.shape)
        
        d2 = self.up3(d3)  # 128 -> 64
        print("d2",d2.shape)
        d2 = self.conv3(d2)
        print("d2",d2.shape)
        
        d1 = self.up4(d2)  # 64 -> 32
        print("d1",d1.shape)
        d1 = self.conv4(d1)
        print("d1",d1.shape)
        
        #d0 = self.up5(d1)  # 32 -> 16
        #print("d0",d0.shape)
        #d0 = self.conv5(d0)
        #print("d0",d0.shape)
        
        # Final classification layer
        out = self.final_conv(d1)
        print("out",out.shape)
        
        return out

# model = UNetResNet18(num_classes=5)
# total_params = sum(p.numel() for p in model.parameters())
# print(f'Total number of parameters to train: {total_params}')
# 