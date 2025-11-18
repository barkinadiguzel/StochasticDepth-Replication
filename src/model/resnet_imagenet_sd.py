import torch
import torch.nn as nn
from src.blocks.bottleneck_stage import BottleneckStage
from src.layers.conv_layer import ConvLayer

class ResNetImageNet_SD(nn.Module):
    def __init__(self, num_classes=1000, block_counts=[3,4,6,3], survival_prob_start=1.0, survival_prob_end=0.5):
        """
        block_counts: number of bottleneck blocks per stage (ResNet-50 example: [3,4,6,3])
        """
        super(ResNetImageNet_SD, self).__init__()
        self.in_channels = 64

        # Initial conv + maxpool
        self.conv1 = ConvLayer(3, self.in_channels, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stages
        self.stage1 = BottleneckStage(self.in_channels, 256, num_blocks=block_counts[0],
                                      survival_prob_start=survival_prob_start,
                                      survival_prob_end=survival_prob_end, stride=1)
        self.stage2 = BottleneckStage(256, 512, num_blocks=block_counts[1],
                                      survival_prob_start=survival_prob_start,
                                      survival_prob_end=survival_prob_end, stride=2)
        self.stage3 = BottleneckStage(512, 1024, num_blocks=block_counts[2],
                                      survival_prob_start=survival_prob_start,
                                      survival_prob_end=survival_prob_end, stride=2)
        self.stage4 = BottleneckStage(1024, 2048, num_blocks=block_counts[3],
                                      survival_prob_start=survival_prob_start,
                                      survival_prob_end=survival_prob_end, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
