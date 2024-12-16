import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        
        # starting number of filters
        nf = 64
        
        self.layers = nn.Sequential(
            # initial layer without normalization
            nn.Conv2d(input_channels, nf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # increasing number of filters as we go deeper
            self._make_disc_block(nf, nf * 2),     # 256x256 -> 128x128
            self._make_disc_block(nf * 2, nf * 4),  # 128x128 -> 64x64
            self._make_disc_block(nf * 4, nf * 8),  # 64x64 -> 32x32
            
            nn.Conv2d(nf * 8, 1, kernel_size=4, stride=1, padding=1)  
        )

    def _make_disc_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.layers(x)
