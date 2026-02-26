import torch
import torch.nn as nn

class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden=128, swap=False):
        super().__init__()
        self.dim = dim
        self.swap = swap

    def forward(self, z):
        log_det = torch.zeros(z.size(0), device=z.device)
        return z, log_det

    def inverse(self, z):
        log_det = torch.zeros(z.size(0), device=z.device)
        return z, log_det

class FlowSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, z):
        log_det = torch.zeros(z.size(0), device=z.device)
        return z, log_det

    def inverse(self, z):
        log_det = torch.zeros(z.size(0), device=z.device)
        return z, log_det