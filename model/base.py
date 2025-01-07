import math

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import repeat, rearrange


class Encoder(nn.Module):
    def __init__(self, name):
        super().__init__()

        self.name = name


class Decoder(nn.Module):
    def __init__(self, name):
        super().__init__()

        # self.denormalizer = denormalizer
        self.name = name


class Denoiser(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name


class GNN(nn.Module):
    def __init__(self, graph, name):
        super().__init__()

        self.graph = graph

        self.name = name
