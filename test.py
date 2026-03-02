import torch
import argparse

from experiments import build_model, make_loaders


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="path to model.pt")
    parser.add_argument("--prior", type=str, required=True,
                        choices=["gaussian", "mog", "flow"])
    parser.add_argument("--M", type=int, default=2)
    parser.add_argument("--batch", type=int, default=128)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loaders
    train_loader, test_loader = make_loaders(batch_size=args.batch)

    # build model
    model = build_model(args.prior, args.M, device).to(device)

    # load checkpoint
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)

    model.eval()

    # take one batch
    x, _ = next(iter(test_loader))
    x = x.to(device)

    # compute ELBO terms
    elbo, recon, kl = model.elbo_terms(x, n_mc=1)

    print("\n==== ELBO diagnostics ====")
    print("batch size:", x.shape[0])

    print("\n--- means ---")
    print("recon mean:", recon.mean().item())
    print("kl mean:", kl.mean().item())
    print("elbo mean:", elbo.mean().item())

    print("\n--- ranges ---")
    print("recon range:", recon.min().item(), recon.max().item())
    print("kl range:", kl.min().item(), kl.max().item())

    # encoder statistics
    q = model.encoder(x)

    mu = q.mean
    print("\n==== encoder statistics ====")
    print("mu min:", mu.min().item())
    print("mu max:", mu.max().item())
    print("mu abs mean:", mu.abs().mean().item())

    if hasattr(q, "variance"):
        var = q.variance
        print("var min:", var.min().item())
        print("var max:", var.max().item())

    print("\nDone.\n")


if __name__ == "__main__":
    main()