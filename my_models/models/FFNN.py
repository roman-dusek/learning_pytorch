import os
import torch
from torch import nn

def create_layer(inputs, outputs):
    return nn.Sequential(
        nn.Linear(inputs, outputs),
        nn.Tanh()
    )

class FFNN(nn.Module):

    def __init__(self, input_size, units: list, n_classes):
        super(FFNN, self).__init__()
        self.layer_sizes = [input_size, *units, n_classes]

        self.layers = [create_layer(ins, outs)
                       for ins, outs in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]

        self.ffnn = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.ffnn(x)
        return x





#%%
