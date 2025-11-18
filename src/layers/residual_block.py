import torch
import torch.nn as nn
from src.layers.conv_layer import ConvLayer
import random

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, survival_prob=1.0):
        super(ResidualBlock, self).__init__()
        self.survival_prob = survival_prob
        self.active = True  

        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        if self.training:
            self.active = random.random() < self.survival_prob
        else:
            self.active = True  

        out = self.conv1(x)
        out = self.conv2(out)

        if not self.active:
            out = 0 

        out += self.shortcut(x) 
        out = nn.ReLU(inplace=True)(out)
        return out
