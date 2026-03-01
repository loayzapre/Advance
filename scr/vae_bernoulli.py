import torch
import torch.nn as nn
import torch.distributions as td

import torch
import torch.nn as nn
import torch.distributions as td

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net: nn.Module, latent_dim: int):
        super().__init__()
        self.encoder_net = encoder_net
        self.latent_dim = latent_dim

    def forward(self, x):
        out = self.encoder_net(x)
        M = self.latent_dim
        if out.shape[-1] != 2 * M:
            raise ValueError(f"encoder_net must output (B, {2*M}), got {tuple(out.shape)}")

        mean, log_std = out[..., :M], out[..., M:]
        log_std = torch.clamp(log_std, min=-8.0, max=2.0)
        std = torch.exp(log_std)

        return td.Independent(td.Normal(mean, std), 1)

class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net: nn.Module):
        super().__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        logits = self.decoder_net(z)

        if logits.dim() == 2:
            if logits.shape[1] != 28 * 28:
                raise ValueError(f"decoder_net (2D) must output (B, 784), got {tuple(logits.shape)}")
            logits = logits.view(z.shape[0], 28, 28)
        elif logits.dim() == 4 and logits.shape[1] == 1:
            logits = logits[:, 0, :, :]  # (B,28,28)
        elif logits.dim() != 3:
            raise ValueError(f"decoder_net must output (B,784) or (B,28,28) or (B,1,28,28), got {tuple(logits.shape)}")

        return td.Independent(td.Bernoulli(logits=logits), 2)