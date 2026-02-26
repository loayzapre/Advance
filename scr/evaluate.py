import torch

@torch.no_grad()
def evaluate_test_elbo(model, data_loader, device):
    return 0.0

@torch.no_grad()
def collect_aggregate_posterior(model, data_loader, device, max_batches=100):
    M = getattr(model.prior, "M", 2)
    return torch.zeros(10, M)

@torch.no_grad()
def sample_prior(model, n_samples=1000):
    M = getattr(model.prior, "M", 2)
    return torch.zeros(n_samples, M)