import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision

class ChestNet(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3,3), padding=1)

        )