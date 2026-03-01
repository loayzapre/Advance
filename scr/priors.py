import torch
import torch.nn as nn
import torch.distributions as td
from .flows import FlowSequential, AffineCoupling

class GaussianPrior(nn.Module):
    """
    Standard Normal prior: p(z)=N(0, I)
    """
    def __init__(self, M):
        super().__init__()
        self.M = M
        self.register_buffer("_loc", torch.zeros(M))
        self.register_buffer("_scale", torch.ones(M))

    def _dist(self, device):
        loc = self._loc.to(device)
        scale = self._scale.to(device)
        return td.Independent(td.Normal(loc, scale), 1)

    def sample(self, shape):
        return self._dist(self._loc.device).sample(shape)

    def log_prob(self, z):
        return self._dist(z.device).log_prob(z)


class MixturePrior(nn.Module):
    """
    Diagonal-covariance GMM prior with K components.
    """
    def __init__(self, M, K=10):
        super().__init__()
        self.M = M
        self.K = K

        self.logits = nn.Parameter(torch.zeros(K))
        self.loc = nn.Parameter(torch.randn(K, M) * 0.05)
        self.log_scale = nn.Parameter(torch.zeros(K, M))

    def _dist(self):
        mix = td.Categorical(logits=self.logits)
        comp = td.Independent(td.Normal(self.loc, self.log_scale.exp()), 1)
        return td.MixtureSameFamily(mix, comp)

    def sample(self, shape):
        return self._dist().sample(shape)

    def log_prob(self, z):
        return self._dist().log_prob(z)


class FlowPrior(nn.Module):
    """
    Flow-based prior:
      u ~ N(0,I)
      z = f(u)
      log p(z) = log p(u) + log |det J_{f^{-1}}(z)|
    """
    def __init__(self, M, n_layers=4, hidden=128):
        super().__init__()
        self.M = M
        self.flow = FlowSequential(*[
            AffineCoupling(M, hidden=hidden, swap=(i % 2 == 1))
            for i in range(n_layers)
        ])
        self.register_buffer("_base_loc", torch.zeros(M))
        self.register_buffer("_base_scale", torch.ones(M))

    def _base(self, device):
        loc = self._base_loc.to(device)
        scale = self._base_scale.to(device)
        return td.Independent(td.Normal(loc, scale), 1)

    @torch.no_grad()
    def sample(self, shape):
        device = self._base_loc.device
        u = self._base(device).sample(shape)        # (..., M)
        z, _ = self.flow(u)                         # forward
        return z

    def log_prob(self, z):
        u, log_det_inv = self.flow.inverse(z)       # (..., M), (...,)
        return self._base(z.device).log_prob(u) + log_det_inv