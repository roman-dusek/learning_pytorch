import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision

from types import SimpleNamespace


class InceptionBlockNaive(nn.Module):

    def __init__(self, c_in, c_out: dict, act_fn):
        super().__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(c_in, c_out["1x1"], kernel_size=1),
            nn.BatchNorm2d(c_out["1x1"]),
            act_fn()
        )

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(c_in, c_out["3x3"], kernel_size=3),
            nn.BatchNorm2d(c_out["3x3"]),
            act_fn()
        )

        self.conv5x5 = nn.Sequential(
            nn.Conv2d(c_in, c_out["5x5"], kernel_size=5),
            nn.BatchNorm2d(c_out["5x5"]),
            act_fn()
        )

        self.max_pool = nn.Sequential(
            nn.MaxPool2d(c_in, c_out["max"], kernel_size=3, padding=1, stride=1)
        )

    def forward(self, x):
        x_conv1x1 = self.conv1x1(x)
        x_conv3x3 = self.conv3x3(x)
        x_conv5x5 = self.conv5x5(x)
        x_max_pool = self.max_pool(x)
        x_out = torch.cat([x_conv1x1, x_conv3x3, x_conv5x5, x_max_pool], dim=1)
        return x_out


class InceptionReductionBlock(nn.Module):

    def __init__(self, c_in, c_red: dict, c_out: dict, act_fn):
        super().__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(c_in, c_out["1x1"], kernel_size=1),
            nn.BatchNorm2d(c_out["1x1"]),
            act_fn()
        )

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(c_in, c_red["3x3"], kernel_size=1),
            nn.BatchNorm2d(c_red["3x3"]),
            act_fn(),
            nn.Conv2d(c_red["3x3"], c_out["3x3"], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out["3x3"]),
            act_fn()
        )

        self.conv5x5 = nn.Sequential(
            nn.Conv2d(c_in, c_red["5x5"], kernel_size=1),
            nn.BatchNorm2d(c_red["5x5"]),
            act_fn(),
            nn.Conv2d(c_red["5x5"], c_out["5x5"], kernel_size=5, padding=2),
            nn.BatchNorm2d(c_out["5x5"]),
            act_fn()
        )

        self.max_pool = nn.Sequential(
            nn.MaxPool2d(c_in, c_out["max"], kernel_size=3, padding=1, stride=1),
            nn.Conv2d(c_in, c_out["1x1"], kernel_size=1),
            nn.BatchNorm2d(c_out["1x1"]),
            act_fn()
        )

    def forward(self, x):
        x_conv1x1 = self.conv1x1(x)
        x_conv3x3 = self.conv3x3(x)
        x_conv5x5 = self.conv5x5(x)
        x_max_pool = self.max_pool(x)
        x_out = torch.cat([x_conv1x1, x_conv3x3, x_conv5x5, x_max_pool], dim=1)
        return x_out


act_fns_dict = {"tanh": nn.Tanh,
                  "relu": nn.ReLU,
                  "leakyrelu": nn.LeakyReLU,
                  "gelu": nn.GELU}


class GoogLeNet(pl.LightningModule):

    def __init__(self, n_classes, act_fn_name="relu", ):
        super().__init__()
        self.hparams = SimpleNamespace(n_classes=n_classes,
                                         act_fn_name=act_fn_name,
                                         act_fn=act_fns_dict[act_fn_name])

        self._create_network()
        self._init_params()

    def _create_network(self):
        self.input_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            self.hparams.act_fn()
        )

        self.inception_blocks = nn.Sequential(
            InceptionReductionBlock(
                64,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 32, "5x5": 8, "max": 8},
                act_fn=self.hparams.act_fn
            ),
            InceptionReductionBlock(
                64,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 32, "5x5": 8, "max": 8},
                act_fn=self.hparams.act_fn
            ),
            InceptionReductionBlock(
                64,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 32, "5x5": 8, "max": 8},
                act_fn=self.hparams.act_fn
            ),
            InceptionReductionBlock(
                64,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 32, "5x5": 8, "max": 8},
                act_fn=self.hparams.act_fn
            ),
            InceptionReductionBlock(
                64,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 32, "5x5": 8, "max": 8},
                act_fn=self.hparams.act_fn
            ),

        )

        # Mapping to classification output
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, self.hparams.num_classes)
        )
    def forward(self, x):
        pass


    def configure_optimizers(self):
        pass


    def training_step(self):
        pass
