import torch
import torch.nn as nn

class AffineCoupling(nn.Module):
    """
    RealNVP-style affine coupling.

    z split into (z1, z2).
    z1 stays fixed, z2 transformed using scale+shift predicted from z1.

    forward:  z -> y,  log_det = sum(log|dy/dz|)  over transformed dims
    inverse:  y -> z,  log_det = sum(log|dz/dy|)  (negative of forward log_det)
    """
    def __init__(self, dim: int, hidden: int = 128, swap: bool = False):
        super().__init__()
        if dim < 2:
            raise ValueError("AffineCoupling requires dim >= 2")

        self.dim = dim
        self.swap = swap

        self.d1 = dim // 2
        self.d2 = dim - self.d1

        self.net = nn.Sequential(
            nn.Linear(self.d1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * self.d2),
        )

        # bound log-scale for stability
        self.scale_factor = 2.0

    def _maybe_swap(self, z: torch.Tensor) -> torch.Tensor:
        if not self.swap:
            return z
        return torch.cat([z[..., self.d1:], z[..., :self.d1]], dim=-1)

    def _split(self, z: torch.Tensor):
        z1 = z[..., :self.d1]
        z2 = z[..., self.d1:]
        return z1, z2

    def forward(self, z: torch.Tensor):
        # z: (..., dim)
        z = self._maybe_swap(z)
        z1, z2 = self._split(z)

        h = self.net(z1)
        t, s = h[..., :self.d2], h[..., self.d2:]
        s = torch.tanh(s) * self.scale_factor  # bounded log-scale

        y2 = z2 * torch.exp(s) + t
        y = torch.cat([z1, y2], dim=-1)
        y = self._maybe_swap(y)

        log_det = s.sum(dim=-1)  # (...,)
        return y, log_det

    def inverse(self, y: torch.Tensor):
        # y: (..., dim)
        y = self._maybe_swap(y)
        y1, y2 = self._split(y)

        h = self.net(y1)
        t, s = h[..., :self.d2], h[..., self.d2:]
        s = torch.tanh(s) * self.scale_factor

        z2 = (y2 - t) * torch.exp(-s)
        z = torch.cat([y1, z2], dim=-1)
        z = self._maybe_swap(z)

        log_det = (-s).sum(dim=-1)  # (...,)
        return z, log_det


class FlowSequential(nn.Module):
    """
    Sequential container that also accumulates log determinants.
    """
    def __init__(self, *layers: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, z: torch.Tensor):
        # z: (..., dim)
        log_det_total = torch.zeros(z.shape[:-1], device=z.device)
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total = log_det_total + log_det
        return z, log_det_total

    def inverse(self, z: torch.Tensor):
        # z: (..., dim)
        log_det_total = torch.zeros(z.shape[:-1], device=z.device)
        for layer in reversed(self.layers):
            z, log_det = layer.inverse(z)
            log_det_total = log_det_total + log_det
        return z, log_det_total