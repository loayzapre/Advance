import os, json, random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from scr.priors import GaussianPrior, MixturePrior, FlowPrior
from scr.vae_bernoulli import GaussianEncoder, BernoulliDecoder
from scr.vae_model import VAE
from scr.train import train
from scr.evaluate import (
    collect_aggregate_posterior,
    sample_prior,
    evaluate_test_elbo_breakdown
)
from scr.plots import save_loss_curve

from scr.plots import save_sample_grid, save_recon_grid

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_loaders(batch_size=128):
    th = 0.5

    def binarize(img):
        x = transforms.functional.to_tensor(img)  # (1,28,28)
        x = (x > th).float()
        return x.squeeze(0)                       # (28,28)

    train_ds = datasets.MNIST("data/", train=True, download=True, transform=binarize)
    test_ds  = datasets.MNIST("data/", train=False, download=True, transform=binarize)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    return train_loader, test_loader

def build_model(prior_name, M, device):
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )
    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )

    if prior_name == "gaussian":
        prior = GaussianPrior(M)
    elif prior_name == "mog":
        prior = MixturePrior(M, K=10)
    elif prior_name == "flow":  
        prior = FlowPrior(M, n_layers=4, hidden=256)
    else:
        raise ValueError(f"Unknown prior name: {prior_name}")

    encoder = GaussianEncoder(encoder_net, latent_dim=M)
    decoder = BernoulliDecoder(decoder_net)
    return VAE(prior, encoder, decoder).to(device)

def sample_grid(model, n=8, device=None, title="Samples"):
    """
    Generates an n x n grid of samples from the model prior and decoder.
    Assumes decoder outputs Bernoulli distribution over (28,28).
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    x = model.sample(n_samples=n*n)   # should return (n*n, 28, 28) or (n*n, ...)
    if x.dim() == 4 and x.shape[1] == 1:
        x = x[:, 0]  # (N,28,28)

    x = x.detach().cpu()

    fig, ax = plt.subplots()
    big = torch.zeros(n*28, n*28)

    idx = 0
    for i in range(n):
        for j in range(n):
            big[i*28:(i+1)*28, j*28:(j+1)*28] = x[idx]
            idx += 1

    ax.imshow(big.numpy(), cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

def run_name(prior_name, M, prior_obj=None):
    if prior_name == "gaussian":
        return f"gaussian_M{M}"
    if prior_name == "mog":
        K = getattr(prior_obj, "K", 10)
        return f"mog_M{M}_K{K}"
    if prior_name == "flow":
        # si guardas n_layers/hidden como attrs en FlowPrior, aquí los lees
        L = getattr(prior_obj, "n_layers", 4)
        H = getattr(prior_obj, "hidden", 128)
        return f"flow_M{M}_L{L}_H{H}"
    return f"{prior_name}_M{M}"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_s = 64

    train_loader, test_loader = make_loaders(batch_size=batch_s)

    base_dir = "runs"
    os.makedirs(base_dir, exist_ok=True)

    

    priors = ["gaussian", "mog", "flow"] # gaussian, mog, flow
    seeds = [0, 1, 2, 3, 4] # Multiple seeds for better estimates of performance
    M = 2 # Latent dimension, M = 2 for visualization
    epochs = 50

    summary = {}

    for p in priors:
        vals = []
        for s in seeds:
            set_seed(s)

            model = build_model(p, M, device)
            print(p, type(model.prior), getattr(model.prior, "n_layers", None))
            run = run_name(p, M, model.prior)
            out_dir = os.path.join(base_dir, run, f"seed{s}")
            os.makedirs(out_dir, exist_ok=True)

            opt = torch.optim.Adam(model.parameters(), lr=1e-3)

            # ---- TRAIN ONCE (and keep losses) ----
            epoch_losses = train(model, opt, train_loader, epochs=epochs, device=device)
            save_loss_curve(epoch_losses, out_dir)

            # ---- EVALUATE TEST ELBO (approx log-likelihood) ----
            # Use more MC samples for a less noisy estimate
            metrics_eval = evaluate_test_elbo_breakdown(model, test_loader, device, n_mc=10)
            vals.append(metrics_eval["test_elbo"])

            # ---- SAVE MODEL ----
            torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

            # ---- SAVE CONFIG / METRICS ----
            config = {"prior": p, "M": M, "seed": s, "epochs": epochs, "lr": 1e-3, "batch": batch_s, "test_n_mc": 10}
            with open(os.path.join(out_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)

            metrics = {
                "test_elbo": float(metrics_eval["test_elbo"]),
                "test_recon": float(metrics_eval["test_recon"]),
                "test_kl": float(metrics_eval["test_kl"])
            }

            with open(os.path.join(out_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

            # ---- SAVE SAMPLES / RECONS ----
            save_sample_grid(model, os.path.join(out_dir, "samples.png"), n=8)
            save_recon_grid(model, test_loader, os.path.join(out_dir, "recon.png"), n=8, device=device)

            # ---- SAVE LATENT SAMPLES ----
            agg_z = collect_aggregate_posterior(model, test_loader, device, max_batches=30)  # (N,M)
            prior_z = sample_prior(model, n_samples=5000, device=device)                      # (5000,M)

            np.save(os.path.join(out_dir, "agg_posterior_z.npy"), agg_z.numpy())
            np.save(os.path.join(out_dir, "prior_samples_z.npy"), prior_z.numpy())

        summary[p] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "all": list(map(float, vals))}
        print(p, summary[p])

    with open(os.path.join(base_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":  
    main()