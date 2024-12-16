import torch
import torch.nn as nn

class SimpleStrokeBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()


        # simplified stroke detection - just two directions instead of four
        self.vertical = nn.Conv2d(channels, channels // 2, kernel_size=(5, 1), padding=(2, 0))
        self.horizontal = nn.Conv2d(channels, channels // 2, kernel_size=(1, 5), padding=(0, 2))
        
        self.combine = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x):
        v_strokes = self.vertical(x)
        h_strokes = self.horizontal(x)
        combined = torch.cat([v_strokes, h_strokes], dim=1)
        return self.combine(combined)

class StrokeAwareResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            SimpleStrokeBlock(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class StrokeAwareGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, num_residual_blocks=6):  
        super().__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # downsampling
        self.down_blocks = nn.Sequential(
            self._make_down_block(64, 128),
            self._make_down_block(128, 256)
        )
        
        # residual blocks with stroke awareness
        self.residual_blocks = nn.Sequential(
            *[StrokeAwareResidualBlock(256) for _ in range(num_residual_blocks)]
        )
        
        # upsampling
        self.up_blocks = nn.Sequential(
            self._make_up_block(256, 128),
            self._make_up_block(128, 64)
        )
        
        # final convolution
        self.final = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def _make_down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 
                              kernel_size=3, stride=2, 
                              padding=1, output_padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.down_blocks(x)
        x = self.residual_blocks(x)
        x = self.up_blocks(x)
        x = self.final(x)
        return x
