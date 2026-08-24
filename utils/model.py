import torch
import torch.nn as nn
import math

class ConvBlock(nn.Module):
    # Conv -> BN -> leakyReLU
    def __init__(
            self,
            in_channels,
            out_channels,
            use_activation=True,
            use_BatchNorm=True,
            **kwargs
    ):

        super().__init__()
        self.use_activation = use_activation
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels) if use_BatchNorm else nn.Identity()
        self.ac = (
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x1 = self.cnn(x)
        x2 = self.bn(x1)
        x3 = self.ac(x2)
        return x3 if self.use_activation else x2

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels * (scale_factor ** 2),
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.ps = nn.PixelShuffle(scale_factor)
        self.ac = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        return self.ac(self.ps(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.b1 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.b2 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_activation=False
        )

    def forward(self, x):
        out = self.b1(x)
        out = self.b2(out)
        return out + x

class Generator(nn.Module):
    def __init__(self, scale=4, in_channels=3, num_channels=64, num_residual_blocks=16):
        super().__init__()

        if scale < 2 or (scale & (scale - 1)) != 0:
            raise ValueError("Scale must be a power of 2")
        num_upsamples = int(math.log2(scale))

        # Initial conv
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_BatchNorm=False)

        # Residual blocks
        self.res = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_residual_blocks)])

        # Conv after residual
        self.conv = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_activation=False)

        # Upsampling
        self.up = nn.Sequential(*[UpsampleBlock(num_channels, scale_factor=2) for _ in range(num_upsamples)])

        # Final conv
        self.final = nn.Conv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = self.initial(x)
        c = self.res(x)
        c = self.conv(c) + x
        c = self.up(c)
        return torch.sigmoid(self.final(c))