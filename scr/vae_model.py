import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, prior, encoder, decoder):
        super().__init__()
        self.prior = prior
        self.encoder = encoder
        self.decoder = decoder

    def _reduce(self, lp: torch.Tensor) -> torch.Tensor:
        """
        Reduce log_prob to per-sample:
        (B,) -> (B,)
        (B,D...) -> sum over last dims -> (B,)
        (S,B) -> (S,B)
        (S,B,D...) -> sum over last dims -> (S,B)
        """
        while lp.dim() > 2:
            lp = lp.sum(dim=-1)
        if lp.dim() == 2 and lp.shape[0] != lp.shape[-1]:
            # usually (S,B) or (B,D). If it's (B,D), sum over D.
            # safer rule: if last dim != batch dim, sum last dim.
            pass
        return lp if lp.dim() <= 2 else lp.sum(dim=-1)

    def elbo(self, x, n_mc=1):
        elbo, _, _ = self.elbo_terms(x, n_mc=n_mc)
        return elbo

    def forward(self, x, n_mc=1):
        return -self.elbo(x, n_mc=n_mc).mean()

    @torch.no_grad()
    def sample(self, n_samples=64):
        device = next(self.parameters()).device
        z = self.prior.sample((n_samples,))
        if isinstance(z, torch.Tensor):
            z = z.to(device)
        px = self.decoder(z)
        return px.sample()
    
    def elbo_terms(self, x, n_mc=1):
        """
        Returns (elbo, recon, kl) each of shape (B,)
        recon = E_q[log p(x|z)]
        kl    = E_q[log q(z|x) - log p(z)]
        """
        q = self.encoder(x)
        B = x.shape[0]

        if n_mc > 1:
            z = q.rsample((n_mc,))  # (S,B,M)
            S, B, M = z.shape

            z_flat = z.reshape(S * B, M)
            x_rep = x.unsqueeze(0).expand(S, *x.shape)
            x_flat = x_rep.reshape(S * B, *x.shape[1:])

            px = self.decoder(z_flat)
            log_px = px.log_prob(x_flat)          # (S*B,)
            recon = log_px.view(S, B).mean(0)     # (B,)

            logq = q.log_prob(z)
            if logq.dim() > 2:
                logq = logq.sum(dim=-1)
            logp = self.prior.log_prob(z)
            if logp.dim() > 2:
                logp = logp.sum(dim=-1)
            kl = (logq - logp).mean(0)            # (B,)

        else:
            z = q.rsample()                       # (B,M)
            px = self.decoder(z)
            recon = px.log_prob(x)                # (B,)

            logq = q.log_prob(z)
            if logq.dim() > 1:
                logq = logq.sum(-1)
            logp = self.prior.log_prob(z)
            if logp.dim() > 1:
                logp = logp.sum(-1)
            kl = logq - logp                      # (B,)

        elbo = recon - kl
        return elbo, recon, kl