#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import torch

from scr.fid import compute_fid
from experiments import make_loaders, build_model


def to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    """
    fid.py needs values in [-1,1].
    We convert: x in [0,1] -> 2x - 1 in [-1,1].
    """
    x = x.float()
    # Heuristics to detect if x is already in [-1,1]
    if x.min().item() >= -0.01 and x.max().item() <= 1.01:
        x = 2.0 * x - 1.0
    return x.clamp(-1.0, 1.0)


@torch.no_grad()
def collect_real_images(test_loader, n: int) -> torch.Tensor:
    """
    Collects n real images from the test loader, returning them as a tensor in the range [-1,1].
    """
    chunks = []
    got = 0
    for x, _ in test_loader:
        # the binary returns (B,28,28)
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (B,1,28,28)
        take = min(x.shape[0], n - got)
        chunks.append(x[:take].cpu())
        got += take
        if got >= n:
            break
    if got < n:
        raise RuntimeError(f"the test loader only delivered {got} images, but you requested n={n}")
    x_real = torch.cat(chunks, dim=0)
    return to_minus1_1(x_real)


@torch.no_grad()
def generate_images_with_timing(model, n: int, batch_size: int, device: str) -> tuple[torch.Tensor, dict]:
    """
    Generates (n,1,28,28) images on CPU, in the range [-1,1].
    Measures wall-clock time ONLY for sampling (model.sample + post-proc).
    """
    model.eval()
    outs = []
    generated = 0

    t0 = time.perf_counter()
    while generated < n:
        b = min(batch_size, n - generated)

        # The API: model.sample(n_samples=...)
        x = model.sample(n_samples=b)  # waiting: (b,28,28) o (b,1,28,28)

        # Normaliza shape
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (b,1,28,28)
        elif x.ndim == 4 and x.shape[1] != 1:
            raise ValueError(f"Expected 1 channel, but sample returned shape {tuple(x.shape)}")

        outs.append(x.detach().to("cpu"))
        generated += b
    t1 = time.perf_counter()

    x_gen = torch.cat(outs, dim=0)
    x_gen = to_minus1_1(x_gen)

    secs = t1 - t0
    stats = {
        "n": n,
        "batch_size": batch_size,
        "sampling_seconds_total": secs,
        "sampling_ms_per_image": 1000.0 * secs / n,
        "sampling_images_per_second": (n / secs) if secs > 0 else float("inf"),
    }
    return x_gen, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path a runs/.../model.pt")
    ap.add_argument("--prior", required=True, choices=["gaussian", "mog", "flow"])
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--device", default="cpu", help="need to be cpu for the classifier, but the model can be on cpu or cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-samples", type=int, default=None, help="Default: complete size of the test set")
    ap.add_argument("--classifier-ckpt", default="mnist_classifier.pth")
    args = ap.parse_args()

    if args.device != "cpu":
        # No te lo prohíbo, pero dijiste gbar CPU. Mejor que falle si te equivocas.
        raise ValueError("Este script está pensado para CPU. Usa --device cpu.")

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"No existe checkpoint: {model_path}")

    clf_path = Path(args.classifier_ckpt)
    if not clf_path.exists():
        raise FileNotFoundError(
            f"No existe mnist_classifier.pth: {clf_path}\n"
            f"Colócalo en la raíz del proyecto o pasa --classifier-ckpt <ruta>."
        )

    # Loaders: from experiments.py, but just the test loader is needed for the real images
    _, test_loader = make_loaders(batch_size=args.batch_size)

    # N real samples
    if args.num_samples is None:
        n = len(test_loader.dataset)
    else:
        n = args.num_samples

    # Build & load model
    model = build_model(args.prior, args.M, device="cpu")
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Real
    x_real = collect_real_images(test_loader, n=n)

    # Generated + timing
    x_gen, timing = generate_images_with_timing(model, n=n, batch_size=args.batch_size, device="cpu")

    # FID (fid.py will move the classifier to device="cpu")
    fid = compute_fid(
        x_real, x_gen,
        device="cpu",
        classifier_ckpt=str(clf_path),
    )

    print("\n=== Results ===")
    print(f"FID: {fid:.4f}")
    print("Sampling wall-clock (just generation):")
    print(f"  total seconds   : {timing['sampling_seconds_total']:.6f}")
    print(f"  ms / image      : {timing['sampling_ms_per_image']:.6f}")
    print(f"  images / second : {timing['sampling_images_per_second']:.3f}")


if __name__ == "__main__":
    main()