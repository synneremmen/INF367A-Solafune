import torch.nn as nn
import torch.nn.functional as F

class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()

        # Encoder
        self.down1 = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),     # 32*12*3*3 + 32 = 3,488
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),     # 32*32*3*3 + 32 = 9,248
            nn.ReLU(),
            nn.MaxPool2d(2)  # 1024 -> 512
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),     # 64*32*3*3 + 64 = 18,496
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),     # 64*64*3*3 + 64 = 36,928
            nn.ReLU(),
            nn.MaxPool2d(2)  # 512 -> 256
        )

        # Decoder
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 32*64*2*2 = 8,192
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),          # 32*32*3*3 + 32 = 9,248
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 16*32*2*2 = 2,048
            nn.ReLU(),
            nn.Conv2d(16, 5, kernel_size=3, padding=1)            # 5*16*3*3 + 5 = 725
        )

    def forward(self, x):
        x = self.down1(x)   # [B, 32, 512, 512]
        x = self.down2(x)   # [B, 64, 256, 256]
        x = self.up1(x)     # [B, 32, 512, 512]
        x = self.up2(x)     # [B, 5, 1024, 1024]
        return x
    
# model = SimpleConvNet()
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f'Total number of trainable parameters: {trainable_params}')

# Total number of trainable parameters: 88421