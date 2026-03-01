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
        """
        ELBO(x) = E_q(z|x)[ log p(x|z) + log p(z) - log q(z|x) ]
        Returns: (B,)
        """
        q = self.encoder(x)

        if n_mc == 1:
            z = q.rsample()                 # (B, M)
            px = self.decoder(z)

            log_px = px.log_prob(x)         # (B,) or (B,D...)
            log_pz = self.prior.log_prob(z) # (B,) or (B, ...)
            log_qz = q.log_prob(z)          # (B,) or (B, ...)

            # reduce to (B,)
            log_px = log_px.view(log_px.shape[0], -1).sum(dim=1) if log_px.dim() > 1 else log_px
            log_pz = log_pz.view(log_pz.shape[0], -1).sum(dim=1) if log_pz.dim() > 1 else log_pz
            log_qz = log_qz.view(log_qz.shape[0], -1).sum(dim=1) if log_qz.dim() > 1 else log_qz

            return log_px + log_pz - log_qz

        # n_mc > 1
        z = q.rsample((n_mc,))              # (S, B, M)
        px = self.decoder(z)

        log_px = px.log_prob(x)             # (S,B) or (S,B,D...)
        log_pz = self.prior.log_prob(z)     # (S,B) or (S,B,...)
        log_qz = q.log_prob(z)              # (S,B) or (S,B,...)

        # reduce to (S,B)
        if log_px.dim() > 2:
            log_px = log_px.view(n_mc, log_px.shape[1], -1).sum(dim=2)
        if log_pz.dim() > 2:
            log_pz = log_pz.view(n_mc, log_pz.shape[1], -1).sum(dim=2)
        if log_qz.dim() > 2:
            log_qz = log_qz.view(n_mc, log_qz.shape[1], -1).sum(dim=2)

        return (log_px + log_pz - log_qz).mean(dim=0)  # (B,)

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