import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder1 = self.encoder(3, 64)
        self.encoder2 = self.encoder(64, 128)
        self.encoder3 = self.encoder(128, 256)

    def encoder(self, x, in_channels, num_filters):
        x = nn.Conv2d(in_channels, num_filters, 3)(x)
        x = self.relu(x)
        x = nn.Conv2d(num_filters, num_filters, 3)(x)
        x = self.relu(x)
        x = self.pool(x)
        return x
    
    def decoder(self, x, skip_feature, in_channels, num_filters):
        x = nn.ConvTranspose2d(in_channels, num_filters,kernel_size=(2,2), stride=2)(x)
        skip_feature = F.interpolate(skip_feature, size=(x.shape[1], x.shape[2]))
        x = torch.cat((x, skip_feature))
        x = nn.Conv2d(num_filters, kernel_size=3)(x)
        x = self.relu(x)
        x = nn.Conv2d(num_filters, kernel_size=3)(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        # [batch, 12, 1024, 1024]

        # [batch, 5, 1024, 1024]
        return x
