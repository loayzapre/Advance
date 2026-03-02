import torch
import os, json
import matplotlib.pyplot as plt

@torch.no_grad()
def collect_latents_with_labels(model, data_loader, device, max_batches=30, use_mean=True):
    """
    Returns:
      Z: (N, M)
      y: (N,)
    use_mean=True -> use q.mean (cleaner scatter)
    use_mean=False -> sample from q(z|x)
    """
    model.eval()
    Zs, ys = [], []

    for i, (x, y) in enumerate(data_loader):
        if i >= max_batches:
            break
        x = x.to(device)

        q = model.encoder(x)

        if use_mean:
            # For Independent(Normal), mean is available as q.mean
            z = q.mean
        else:
            z = q.sample()

        Zs.append(z.detach().cpu()) 
        ys.append(y.detach().cpu())

    Z = torch.cat(Zs, dim=0)
    y = torch.cat(ys, dim=0)
    return Z, y

@torch.no_grad()
def save_sample_grid(model, path, n=8):
    model.eval()
    device = next(model.parameters()).device
    x = model.sample(n_samples=n*n)
    if x.dim() == 4 and x.shape[1] == 1:
        x = x[:, 0]
    x = x.detach().cpu()

    big = torch.zeros(n*28, n*28)
    idx = 0
    for i in range(n):
        for j in range(n):
            big[i*28:(i+1)*28, j*28:(j+1)*28] = x[idx]
            idx += 1

    plt.figure()
    plt.imshow(big.numpy(), cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

@torch.no_grad()
def save_recon_grid(model, data_loader, path, n=8, device=None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    x, _ = next(iter(data_loader))
    x = x.to(device)[:n*n]

    q = model.encoder(x)
    z = q.mean
    px = model.decoder(z)
    xr = px.sample()

    if x.dim() == 4 and x.shape[1] == 1:
        x = x[:, 0]
    if xr.dim() == 4 and xr.shape[1] == 1:
        xr = xr[:, 0]

    x = x.detach().cpu()
    xr = xr.detach().cpu()

    big = torch.zeros(n*28, 2*n*28)
    for idx in range(n*n):
        i, j = divmod(idx, n)
        big[i*28:(i+1)*28, j*56:(j*56)+28] = x[idx]
        big[i*28:(i+1)*28, j*56+28:(j*56)+56] = xr[idx]

    plt.figure()
    plt.imshow(big.numpy(), cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def save_loss_curve(epoch_losses, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "loss_values.json"), "w") as f:
        json.dump({"epoch_loss": epoch_losses}, f, indent=2)

    plt.figure()
    plt.plot(epoch_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (-ELBO)")
    plt.title("Training loss per epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150)
    plt.close()