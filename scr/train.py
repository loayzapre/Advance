# train.py
import torch
from tqdm import tqdm

def train(model, optimizer, data_loader, epochs, device, n_mc=1, grad_clip=None):
    model.train()
    total_steps = len(data_loader) * epochs
    pbar = tqdm(total=total_steps, desc="Training")

    for epoch in range(epochs):
        for x, _ in data_loader:
            x = x.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = model(x, n_mc=n_mc)  # assume VAE.forward(x, n_mc) -> scalar
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            pbar.set_postfix(loss=float(loss.item()), epoch=f"{epoch+1}/{epochs}")
            pbar.update(1)

    pbar.close()


@torch.no_grad()
def eval_elbo(model, data_loader, device, n_mc=1):
    model.eval()
    total = 0.0
    count = 0

    for x, _ in data_loader:
        x = x.to(device)
        elbo = model.elbo(x, n_mc=n_mc)   # (B,)
        total += elbo.sum().item()
        count += elbo.numel()

    return total / max(count, 1)