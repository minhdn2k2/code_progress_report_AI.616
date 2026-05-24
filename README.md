# m-SAM Experiments on CIFAR-10

This repository contains the implementation of SGD, full-batch n-SAM, and mini-batch m-SAM experiments on CIFAR-10 using ResNet-18 with Group Normalization.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Experiments

To run the main experiment:

```bash
python run_all.py \
    --train-subset 0 \
    --epochs 1000 \
    --total-updates 1000 \
    --eval-every-updates 1 \
    --micro-batch-size 256 \
    --eval-batch-size 256 \
    --lr 0.001 \
    --rho 0.05 \
    --seed 0
```

## Arguments

| Argument | Description |
|---|---|
| `--train-subset` | Number of training samples to use (`0` means full dataset) |
| `--epochs` | Number of training epochs |
| `--total-updates` | Total optimization updates |
| `--eval-every-updates` | Evaluation frequency |
| `--micro-batch-size` | Micro-batch size for gradient accumulation |
| `--eval-batch-size` | Batch size for evaluation |
| `--lr` | Learning rate |
| `--rho` | SAM perturbation radius |
| `--seed` | Random seed |

## Dataset

The experiments use the CIFAR-10 dataset with the standard train/test split.

## Model

- ResNet-18
- Group Normalization instead of Batch Normalization

## Methods

The repository supports:

- SGD
- Full-batch n-SAM
- Mini-batch m-SAM

## Hardware

Experiments were conducted on NVIDIA RTX Pro 6000 Blackwell GPUs.
