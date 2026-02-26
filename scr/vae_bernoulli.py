import torch
import torch.nn as nn
import torch.distributions as td

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net, latent_dim=None):
        super().__init__()
        self.encoder_net = encoder_net
        self.latent_dim = latent_dim

    def forward(self, x):
        if self.latent_dim is None:
            out = self.encoder_net(x)
            M = out.shape[-1] // 2
        else:
            M = self.latent_dim
        mean = torch.zeros(x.shape[0], M, device=x.device)
        std = torch.ones(x.shape[0], M, device=x.device)
        return td.Independent(td.Normal(mean, std), 1)

class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        super().__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        logits = torch.zeros(z.shape[0], 28, 28, device=z.device)
        return td.Independent(td.Bernoulli(logits=logits), 2)