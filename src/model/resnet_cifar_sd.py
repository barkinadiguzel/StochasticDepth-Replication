import torch
import torch.nn as nn
from src.blocks.basic_stage import BasicStage
from src.layers.conv_layer import ConvLayer
from src.layers.residual_block import ResidualBlock

class ResNetCIFAR_SD(nn.Module):
    def __init__(self, num_classes=10, block_counts=[2,2,2,2], survival_prob_start=1.0, survival_prob_end=0.5):
        """
        block_counts: List indicating number of residual blocks per stage.
                      Example for ResNet-18 CIFAR: [2, 2, 2, 2]
        """
        super(ResNetCIFAR_SD, self).__init__()
        self.in_channels = 16

        self.conv1 = ConvLayer(3, self.in_channels, kernel_size=3, stride=1, padding=1)

      
        self.stage1 = BasicStage(self.in_channels, 16, num_blocks=block_counts[0],
                                 survival_prob_start=survival_prob_start,
                                 survival_prob_end=survival_prob_end, stride=1)
        self.stage2 = BasicStage(16, 32, num_blocks=block_counts[1],
                                 survival_prob_start=survival_prob_start,
                                 survival_prob_end=survival_prob_end, stride=2)
        self.stage3 = BasicStage(32, 64, num_blocks=block_counts[2],
                                 survival_prob_start=survival_prob_start,
                                 survival_prob_end=survival_prob_end, stride=2)
        self.stage4 = BasicStage(64, 64, num_blocks=block_counts[3],
                                 survival_prob_start=survival_prob_start,
                                 survival_prob_end=survival_prob_end, stride=2)

      
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
