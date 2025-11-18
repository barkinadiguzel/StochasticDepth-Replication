import torch
import random

class StochasticDepth:
    def __init__(self, survival_prob=1.0):
        self.survival_prob = survival_prob

    def forward(self, x, block_function, training=True):
        if training:
            active = random.random() < self.survival_prob
            if active:
                return block_function(x)
            else:
                return x
        else:
            return self.survival_prob * block_function(x) + (1 - self.survival_prob) * x
