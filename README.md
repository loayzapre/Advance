# Variational Autoencoder Experiments

This project implements a Variational Autoencoder (VAE) with different priors and flow-based extensions for latent variable modeling.
It includes scripts for training, evaluation, and visualization of the latent space.

## Project Structure
Project Structure
project/
│
├── src/
│   ├── __init__.py
│   ├── vae_model.py
│   ├── vae_bernoulli.py
│   ├── priors.py
│   ├── flows.py
│   ├── plots.py
│   ├── train.py
│   └── evaluate.py
│
├── experiments.py
├── plot_latent.py
├── data/
├── runs/
└── README.md

src/: Core implementation (model, priors, flows, training utilities).

experiments.py: Main script for running training experiments.

plot_latent.py: Visualization of latent space and priors.

data/: Dataset storage.

runs/: Training outputs and checkpoints.

## Environment Setup
Create a Python environment (recommended: conda or micromamba).

Example with conda:

    conda create -n vae python=3.11
    conda activate vae

Install dependencies:

    pip install torch torchvision matplotlib numpy tqdm

Optional (for experiments):

    pip install scikit-learn

## Running Training
Train the model using:

    python experiments.py

This will:

1. Load the dataset (e.g. MNIST)
2. Train the VAE
3. Save logs and outputs inside:

runs/

## Generating Latent Space Plots
To visualize the latent space and prior contours:

    python plot_latent.py

This script will:

- Encode dataset samples
- Project them into the latent space
- Plot:
    - latent samples
    - prior density contours

The resulting figures will be saved in:

    runs/plots/

## Notes

The latent dimension M must be 2 for contour visualization.

Different priors can be implemented in src/priors.py.

Flow transformations are implemented in src/flows.py.