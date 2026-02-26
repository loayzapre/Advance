import torch
import torch.nn as nn
import torch.distributions as td
from .flows import FlowSequential, AffineCoupling

class GaussianPrior(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
        self.register_buffer("mean", torch.zeros(M))
        self.register_buffer("std", torch.ones(M))

    def sample(self, shape):
        return torch.zeros(*shape, self.M)

    def log_prob(self, z):
        return torch.zeros(z.size(0), device=z.device)

class MixturePrior(nn.Module):
    def __init__(self, M, K=10):
        super().__init__()
        self.M = M
        self.K = K

    def sample(self, shape):
        return torch.zeros(*shape, self.M)

    def log_prob(self, z):
        return torch.zeros(z.size(0), device=z.device)

class FlowPrior(nn.Module):
    def __init__(self, M, n_layers=4, hidden=128):
        super().__init__()
        self.M = M
        self.flow = FlowSequential(*[
            AffineCoupling(M, hidden=hidden, swap=(i % 2 == 1))
            for i in range(n_layers)
        ])

    def sample(self, shape):
        return torch.zeros(*shape, self.M)

    def log_prob(self, z):
        return torch.zeros(z.size(0), device=z.device)