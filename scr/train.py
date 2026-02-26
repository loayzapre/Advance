# train.py
import torch
from tqdm import tqdm

def train(model, optimizer, data_loader, epochs, device):
    model.train()
    total_steps = len(data_loader) * epochs
    pbar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        for x, _ in data_loader:
            x = x.to(device)

            optimizer.zero_grad()
            loss = model(x)          # negative ELBO
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=float(loss.item()), epoch=f"{epoch+1}/{epochs}")
            pbar.update()

@torch.no_grad()
def eval_elbo(model, data_loader, device):
    model.eval()
    vals = []
    for x, _ in data_loader:
        x = x.to(device)
        vals.append(model.elbo(x).detach().cpu())
    return torch.stack(vals).mean().item()