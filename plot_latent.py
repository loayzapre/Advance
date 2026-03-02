import os
import re
import glob
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from experiments import build_model, make_loaders
from scr.plots import collect_latents_with_labels


@torch.no_grad()
def plot_prior_contours_with_data(
    model,
    data_loader,
    device,
    out_path,
    max_batches=30,
    grid_n=300,
    title="Prior contours + aggregate posterior",
    use_mean=True,
    zoom_quantiles=(0.01, 0.99),
):
    """
    Saves a figure showing:
      - scatter of aggregate posterior samples/means (colored by label)
      - contour lines of log p(z) (prior), restricted to high-density levels
    Works for M=2 only.
    """
    model.eval()

    # 1) Collect latents
    Z, y = collect_latents_with_labels(
        model, data_loader, device, max_batches=max_batches, use_mean=use_mean
    )
    if Z.shape[1] != 2:
        raise ValueError(f"Need M=2 to plot contours, got M={Z.shape[1]}")

    Z_np = Z.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    # 2) Robust zoom bounds from aggregate posterior (avoid outliers dominating)
    qlo, qhi = zoom_quantiles
    prior_samp = model.prior.sample((5000,)).detach().cpu().numpy()
    Z_all = np.vstack([Z_np, prior_samp])

    x_min, x_max = np.quantile(Z_all[:, 0], [qlo, qhi])
    y_min, y_max = np.quantile(Z_all[:, 1], [qlo, qhi])
    # Pad a bit
    pad_x = 0.15 * (x_max - x_min + 1e-6)
    pad_y = 0.15 * (y_max - y_min + 1e-6)

    gx_min, gx_max = x_min - pad_x, x_max + pad_x
    gy_min, gy_max = y_min - pad_y, y_max + pad_y

    # 3) Grid over zoomed region
    xs = np.linspace(gx_min, gx_max, grid_n)
    ys = np.linspace(gy_min, gy_max, grid_n)
    X, Y = np.meshgrid(xs, ys)
    grid = np.stack([X.ravel(), Y.ravel()], axis=1)  # (grid_n^2, 2)
    grid_t = torch.tensor(grid, dtype=torch.float32, device=device)

    # 4) Prior log-density on grid
    logp = model.prior.log_prob(grid_t)
    if logp.dim() == 2:
        logp = logp.sum(dim=-1)
    logp = logp.detach().cpu().numpy().reshape(grid_n, grid_n)

    # 5) Levels: only high-density region so Flow doesn't paint the whole plot
    flat = logp.reshape(-1)
    m = flat.max()
    levels = np.linspace(m - 12, m - 0.2, 20)

    # 6) Plot
    plt.figure(figsize=(7, 6))

    # --- aggregate posterior ---
    sc = plt.scatter(Z_np[:, 0], Z_np[:, 1], c=y_np, s=10, alpha=0.6, label="agg posterior")
    plt.colorbar(sc, label="Digit label")

    # --- show MoG component centers (+ optional ellipses) ---
    if hasattr(model.prior, "loc") and hasattr(model.prior, "log_scale"):
        mus = model.prior.loc.detach().cpu().numpy()          # (K,2)
        sig = np.exp(model.prior.log_scale.detach().cpu().numpy())  # (K,2)

        # centers
        plt.scatter(mus[:, 0], mus[:, 1], s=80, marker="x", label="MoG means")

        # optional: axis-aligned 1-sigma ellipses (diag cov)
        t = np.linspace(0, 2*np.pi, 200)
        for k in range(mus.shape[0]):
            ex = mus[k, 0] + sig[k, 0] * np.cos(t)
            ey = mus[k, 1] + sig[k, 1] * np.sin(t)
            plt.plot(ex, ey, linewidth=1.0, alpha=0.6)

    plt.legend(loc="best")

    cs = plt.contour(X, Y, logp, levels=levels, linewidths=1.2, alpha=0.9)
    plt.clabel(cs, inline=1, fontsize=8)

    plt.xlim(gx_min, gx_max)
    plt.ylim(gy_min, gy_max)
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)


def parse_run_dir(run_dir):
    """
    Parses prior and M from folder names like:
      runs/gaussian_M2
      runs/mog_M2_K10
      runs/flow_M2_L4_H128
    """
    base = os.path.basename(run_dir)
    prior = base.split("_")[0]
    m = re.search(r"_M(\d+)", base)
    M = int(m.group(1)) if m else None
    return prior, M


def load_config(seed_dir):
    cfg_path = os.path.join(seed_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            return json.load(f)
    return None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use TRAIN for aggregate posterior unless your assignment says otherwise
    train_loader, test_loader = make_loaders(batch_size=128)
    agg_loader = test_loader

    ckpts = sorted(glob.glob("runs/*/seed*/model.pt"))
    if not ckpts:
        raise FileNotFoundError("No checkpoints found under runs/*/seed*/model.pt")

    for ckpt_path in ckpts:
        seed_dir = os.path.dirname(ckpt_path)
        run_dir = os.path.dirname(seed_dir)

        cfg = load_config(seed_dir)
        if cfg is not None:
            prior_name = cfg["prior"]
            M = int(cfg["M"])
            seed = cfg.get("seed", os.path.basename(seed_dir).replace("seed", ""))
        else:
            prior_name, M = parse_run_dir(run_dir)
            seed = os.path.basename(seed_dir).replace("seed", "")

        if M != 2:
            print(f"[skip] {ckpt_path} (M={M} not 2)")
            continue

        model = build_model(prior_name, M, device).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        freeze(model)

        out_path = os.path.join(seed_dir, "latent_prior_vs_agg.png")
        title = f"{prior_name.upper()} | M={M} | seed={seed}"

        try:
            plot_prior_contours_with_data(
                model=model,
                data_loader=agg_loader,      # aggregate posterior source
                device=device,
                out_path=out_path,
                max_batches=100,
                grid_n=300,
                title=title,
                use_mean=True,               # set False if you want to see variance “arms”
                zoom_quantiles=(0.01, 0.99), # robust zoom
            )
            print(f"[ok] saved {out_path}")
        except Exception as e:
            print(f"[fail] {ckpt_path}: {e}")


if __name__ == "__main__":
    main()