import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt

from scr.plots import collect_latents_with_labels
from experiments import build_model, make_loaders


@torch.no_grad()
def plot_prior_contours_with_data(
    model,
    data_loader,
    device,
    max_batches=100,
    grid_lim=5.0,
    grid_n=200,
    title="Prior contours + encoded data",
):
    model.eval()

    # 1) Collect latents (M must be 2)
    Z, y = collect_latents_with_labels(
        model, data_loader, device, max_batches=max_batches, use_mean=True
    )
    if Z.shape[1] != 2:
        raise ValueError(f"Need M=2 to plot contours, got M={Z.shape[1]}")

    # Ensure cpu numpy for plotting
    Z_np = Z.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    # 2) Make grid
    xs = np.linspace(-grid_lim, grid_lim, grid_n)
    ys = np.linspace(-grid_lim, grid_lim, grid_n)
    X, Y = np.meshgrid(xs, ys)
    grid = np.stack([X.ravel(), Y.ravel()], axis=1)  # (grid_n^2, 2)
    grid_t = torch.tensor(grid, dtype=torch.float32, device=device)

    # 3) Evaluate log p(z) on grid
    prior = model.prior
    logp = prior.log_prob(grid_t)

    # If log_prob returns (N, M), reduce to (N,)
    if logp.dim() == 2:
        logp = logp.sum(dim=-1)

    logp = logp.detach().cpu().numpy().reshape(grid_n, grid_n)

    # 4) Plot
    plt.figure()
    cs = plt.contour(X, Y, logp, levels=15)
    plt.clabel(cs, inline=1, fontsize=8)

    sc = plt.scatter(Z_np[:, 0], Z_np[:, 1], c=y_np, s=12, alpha=0.6)
    plt.colorbar(sc, label="Digit label")

    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def load_first_checkpoint(model, ckpt_dir, device):
    # Typical structure: runs/{prior}_M{M}/seed*/model.pt
    pattern = os.path.join(ckpt_dir, "seed*", "model.pt")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found matching: {pattern}")

    ckpt_path = candidates[0]  # pick first (or choose best if you track metrics)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return ckpt_path


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = make_loaders(batch_size=128)

    prior_name = "mog"  # "mog", "flow", "gaussian"
    M = 2
    model = build_model(prior_name, M, device).to(device)

    ckpt_dir = f"runs/{prior_name}_M{M}_K10"  # adjust if you changed K or run_name format
    ckpt_path = load_first_checkpoint(model, ckpt_dir, device)

    freeze(model)

    plot_prior_contours_with_data(
        model,
        test_loader,
        device,
        max_batches=100,
        grid_lim=5.0,
        grid_n=200,
        title=f"{prior_name.upper()} Prior Contours (log p(z)) + Encoded Data\n{ckpt_path}",
    )


if __name__ == "__main__":
    main()