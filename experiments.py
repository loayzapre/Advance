import os, json, random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from scr.priors import GaussianPrior, MixturePrior, FlowPrior
from scr.vae_bernoulli import GaussianEncoder, BernoulliDecoder
from scr.vae_model import VAE
from scr.train import train
from scr.evaluate import evaluate_test_elbo

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_loaders(batch_size=128):
    th = 0.5
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > th).float().squeeze(0))
    ])
    train_ds = datasets.MNIST("data/", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST("data/", train=False, download=True, transform=tfm)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def build_model(prior_name, M, device):
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 2*M),
    )
    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28)),
    )

    if prior_name == "gaussian":
        prior = GaussianPrior(M)
    elif prior_name == "mog":
        prior = MixturePrior(M, K=10)
    elif prior_name == "flow":
        prior = FlowPrior(M, n_layers=4, hidden=128)
    else:
        raise ValueError(f"Unknown prior name: {prior_name}")

    encoder = GaussianEncoder(encoder_net)
    decoder = BernoulliDecoder(decoder_net)
    return VAE(prior, encoder, decoder).to(device)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = make_loaders(batch_size=128)

    os.makedirs("models", exist_ok=True)

    priors = ["gaussian", "mog", "flow"]
    seeds = [0, 1]
    M = 2

    results = {}
    for p in priors:
        vals = []
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = build_model("gaussian", M=2, device=device)

        x = torch.rand(4, 28, 28).to(device)
        x = (x > 0.5).float()

        for s in seeds:
            set_seed(s)
            model = build_model(p, M, device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            train(model, opt, train_loader, epochs=1, device=device)
            vals.append(evaluate_test_elbo(model, test_loader, device))
            torch.save(model.state_dict(), f"models/{p}_seed{s}.pt")

        results[p] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "all": list(map(float, vals))}
        print(p, results[p])

    with open("models/partA_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()