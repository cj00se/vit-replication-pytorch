# ViT Replication — PyTorch

A from-scratch implementation of the Vision Transformer (ViT) architecture in PyTorch,
based on the paper "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2021).

## What This Is

This notebook walks through building a ViT from the ground up — manually constructing
each component of the architecture and verifying tensor shapes at each step before
assembling the full model and training it on a small image dataset.

## What I Built

- Patch embedding layer using `nn.Conv2d`
- Class token and positional embeddings
- Multi-head self-attention blocks
- Transformer encoder stack
- MLP classification head
- Full ViT assembled from scratch and trained end-to-end

## Architecture Details

- Image size: 224x224
- Patch size: 16x16 → 196 patches + 1 class token = 197 sequence length
- Hidden dim: 768
- Reused modular `engine` and `utils` from prior projects for training loop and logging

## Dataset

Small custom image dataset loaded via `torchvision.datasets.ImageFolder`.
Trained for 10 epochs, results tracked with TensorBoard and matplotlib.

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/cj00se/vit-replication-pytorch.git
cd vit-replication-pytorch
```

### 2. Install dependencies
```bash
pip install torch torchvision torchinfo tensorboard matplotlib
```

### 3. Run the notebook
Open `ViT_replication.ipynb` in Jupyter or VS Code and run all cells.

## Guided By
[Zero to Mastery Learn PyTorch for Deep Learning](https://www.learnpytorch.io/) by Daniel Bourke

## Reference

Dosovitskiy et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image
Recognition at Scale.* arXiv:2010.11929. https://doi.org/10.48550/arXiv.2010.11929

## Hardware
Trained on an NVIDIA RTX 5070 Ti (CUDA).
