import torch

@torch.no_grad()
def evaluate_test_elbo(model, data_loader, device, n_mc=1):
    """
    Returns average ELBO per datapoint over the whole loader.
    Robust to last batch smaller than batch_size.
    """
    model.eval()
    total = 0.0
    count = 0

    for x, _ in data_loader:
        x = x.to(device)
        elbo = model.elbo(x, n_mc=n_mc)   # (B,)
        total += elbo.sum().item()
        count += elbo.numel()

    return total / max(count, 1)


@torch.no_grad()
def collect_aggregate_posterior(model, data_loader, device, max_batches=100):
    """
    Collects latent samples from q(z|x) over (up to) max_batches and returns them.
    This approximates the aggregate posterior q(z) = E_data[q(z|x)] by samples.

    Returns: (N, M) tensor of z samples.
    """
    model.eval()
    zs = []

    for i, (x, _) in enumerate(data_loader):
        if i >= max_batches:
            break
        x = x.to(device)
        q = model.encoder(x)          # distribution
        z = q.sample()                # (B, M)  (no need rsample since no grads)
        zs.append(z.detach().cpu())

    if not zs:
        M = getattr(model.prior, "M", None) or 2
        return torch.empty(0, M)

    return torch.cat(zs, dim=0)       # (N, M)


@torch.no_grad()
def sample_prior(model, n_samples=1000, device=None):
    """
    Draw samples from the model prior.
    Returns: (n_samples, M)
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    z = model.prior.sample((n_samples,))
    if isinstance(z, torch.Tensor):
        z = z.to(device)
    return z.detach().cpu()