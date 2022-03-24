import torch
import torch.nn as nn
import pandas as pd
import numpy as np


class BasicMF(nn.Module):

    def __init__(self, n_users, n_items, embd_dim):
        super().__init__()
        self.user_embd = nn.Embedding(n_users, embd_dim)
        self.item_embd = nn.Embedding(n_items, embd_dim)


        self._init_weights()

    def forward(self, u: torch.LongTensor, i: torch.LongTensor):

        return (self.user_embd(u) * self.item_embd(i)).sum(dim=1)

    def _init_weights(self) -> None:
        self.user_embd.weight.data.uniform_(0, .5)
        self.item_embd.weight.data.uniform_(0, .5)

class BiasMF(nn.Module):

    def __init__(self, n_users, n_items, embd_dim, data_mean):
        super().__init__()
        self.user_embd = nn.Embedding(n_users, embd_dim)
        self.item_embd = nn.Embedding(n_items, embd_dim)

        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.mu = data_mean

        self._init_weights()

    def forward(self, u, v):
        pred = self.user_bias(u) + self.item_bias(v) + self.mu
        pred += (self.user_embd(u) * self.item_embd(v)).sum(dim=1,keepdim=True)
        return pred.squeeze()

    def _init_weights(self) -> None:
        self.user_embd.weight.data.uniform_(0, .5)
        self.item_embd.weight.data.uniform_(0, .5)

        self.user_bias.weight.data.uniform_(0, .2)
        self.item_bias.weight.data.uniform_(0, .2)