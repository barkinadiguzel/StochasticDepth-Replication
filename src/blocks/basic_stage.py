import torch
import torch.nn as nn
from src.layers.residual_block import ResidualBlock
from src.layers.stochastic_depth import StochasticDepth

class BasicStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, survival_prob_start=1.0, survival_prob_end=0.5, stride=1):
        super(BasicStage, self).__init__()
        self.blocks = nn.ModuleList()
        probs = [
            survival_prob_start - (survival_prob_start - survival_prob_end) * i / (num_blocks - 1)
            for i in range(num_blocks)
        ]
        for i in range(num_blocks):
            s = stride if i == 0 else 1  # only first block may have stride > 1
            block = ResidualBlock(in_channels if i == 0 else out_channels, out_channels, stride=s, survival_prob=probs[i])
            self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
