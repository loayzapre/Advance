import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, prior, encoder, decoder):
        super().__init__()
        self.prior = prior
        self.encoder = encoder
        self.decoder = decoder

    def elbo(self, x, n_mc=1):
        """
            ELBO(x) = E_q(z|x)[ log p(x|z) ] - KL( q(z|x) || p(z) )
        """
        
        q = self.encoder(x)
        if n_mc == 1:
            z = q.rsample()                 # (B, M)
            px = self.decoder(z)
            log_px = px.log_prob(x)         # (B,)
            log_pz = self.prior.log_prob(z) # (B,)
            log_qz = q.log_prob(z)          # (B,)
            return log_px + log_pz - log_qz
        else:
            raise NotImplementedError("MC ELBO not implemented yet")

    def forward(self, x):
        # loss = - mean ELBO
        return -self.elbo(x)

    @torch.no_grad()
    def sample(self, n_samples=64):
        device = next(self.parameters()).device
        z = self.prior.sample((n_samples,), device=device)
        px = self.decoder(z)
        return px.sample()