

import torch
import torch.nn as nn
import torchvision.models as models

class UNet_Light(nn.Module):
    def __init__(self, num_classes=5):
        super(UNet_Light, self).__init__()
        
        # Load a pretrained MobileNetV2 backbone
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features

        # Modify the first layer to accept 12-channel input
        self.encoder = backbone
        # in_channels = self.encoder[0][0].in_channels  # Default: 3
        out_channels = self.encoder[0][0].out_channels  # Default: 32
        
        self.encoder[0][0] = nn.Conv2d(12, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Freeze backbone layers
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Bottleneck (bridge between encoder and decoder)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder (Upsampling)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        self.final_upsample = nn.Upsample(size=(1024, 1024), mode="bilinear", align_corners=False)

    def forward(self, x):
        with torch.no_grad():  # Ensure frozen layers don't compute gradients
            x = self.encoder(x)  # Feature extraction
        
        x = self.bottleneck(x)  # Bottleneck processing
        x = self.decoder(x)  # Upsampling back to 1024x1024
        x = self.final_upsample(x)
        return x

# Test the model
# model = UNet_Light(num_classes=5)
# test_input = torch.randn(1, 12, 1024, 1024)  # Batch=1, 12 channels, 1024x1024 image
# output = model(test_input)
# print(model)
# total_params = sum(p.numel() for p in model.parameters())
# print(f'Total number of parameters: {total_params}')

# Params: 9693861