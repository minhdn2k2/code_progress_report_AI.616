import argparse
import csv
import json
import math
import os
import random
import time
from itertools import cycle

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import get_cifar10_loaders
from models import resnet18_gn_cifar10
from sam import SAM, perturb_weights, restore_weights


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def accuracy(logits, targets):
    return (logits.argmax(dim=1) == targets).float().sum().item()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_seen = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct += accuracy(logits, y)
        total_seen += bs

    return {
        "loss": total_loss / max(total_seen, 1),
        "acc": total_correct / max(total_seen, 1),
    }


def compute_full_gradient(model, loader, criterion, device, train_size, desc):
    """
    Computes the gradient of the average training loss over the whole training set.
    Gradients are accumulated over micro-batches to avoid memory issues.
    """
    model.train()
    model.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_correct = 0.0
    total_seen = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        scaled_loss = loss * (bs / train_size)
        scaled_loss.backward()

        total_loss += loss.item() * bs
        total_correct += accuracy(logits, y)
        total_seen += bs

        pbar.set_postfix(loss=total_loss / max(total_seen, 1))

    return {
        "loss": total_loss / max(total_seen, 1),
        "acc": total_correct / max(total_seen, 1),
    }


def sgd_step(model, optimizer, criterion, batch, device):
    model.train()
    x, y = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    return loss.item(), accuracy(logits, y) / x.size(0)


def msam_step(model, optimizer, criterion, batch, device):
    """
    Practical m-SAM update: same mini-batch is used for ascent and descent.
    """
    model.train()
    x, y = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.first_step(zero_grad=True)

    logits_second = model(x)
    loss_second = criterion(logits_second, y)
    loss_second.backward()
    optimizer.second_step(zero_grad=True)

    return loss.item(), accuracy(logits, y) / x.size(0)


def nsam_fullbatch_step(
    model,
    optimizer,
    criterion,
    full_train_loader,
    device,
    train_size,
    rho,
):
    """
    Full-batch n-SAM:
      1) compute full-dataset gradient at w
      2) perturb w by normalized full gradient
      3) compute full-dataset gradient at w + epsilon
      4) restore w and take one optimizer step

    This performs exactly one parameter update using the full training set.
    """
    first_stats = compute_full_gradient(
        model,
        full_train_loader,
        criterion,
        device,
        train_size,
        desc="n-SAM first full-gradient",
    )

    perturbations = perturb_weights(model, rho=rho, adaptive=False)

    second_stats = compute_full_gradient(
        model,
        full_train_loader,
        criterion,
        device,
        train_size,
        desc="n-SAM second full-gradient",
    )

    restore_weights(perturbations)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return {
        "first_loss": first_stats["loss"],
        "first_acc": first_stats["acc"],
        "second_loss": second_stats["loss"],
        "second_acc": second_stats["acc"],
    }


def write_row(csv_path, row):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_summary(output_dir, summary):
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def train_same_epochs(args, model, optimizer, criterion, loaders, device, output_dir):
    train_loader, full_train_loader, train_eval_loader, test_loader, train_size = loaders
    metrics_path = os.path.join(output_dir, "metrics.csv")

    global_update = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_seen = 0

        if args.method == "n_sam":
            stats = nsam_fullbatch_step(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                full_train_loader=full_train_loader,
                device=device,
                train_size=train_size,
                rho=args.rho,
            )
            global_update += 1
            epoch_loss = stats["first_loss"]
            epoch_acc = stats["first_acc"]
            epoch_seen = train_size

        else:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
            for batch in pbar:
                if args.method == "sgd":
                    loss, acc = sgd_step(model, optimizer, criterion, batch, device)
                elif args.method == "m_sam":
                    loss, acc = msam_step(model, optimizer, criterion, batch, device)
                else:
                    raise ValueError(f"Unknown method: {args.method}")

                bs = batch[0].size(0)
                global_update += 1
                epoch_loss += loss * bs
                epoch_acc += acc * bs
                epoch_seen += bs

                pbar.set_postfix(
                    loss=epoch_loss / max(epoch_seen, 1),
                    acc=epoch_acc / max(epoch_seen, 1),
                    updates=global_update,
                )

            epoch_loss /= max(epoch_seen, 1)
            epoch_acc /= max(epoch_seen, 1)

        train_eval = evaluate(model, train_eval_loader, criterion, device)
        test_eval = evaluate(model, test_loader, criterion, device)

        row = {
            "comparison": "same_epochs",
            "method": args.method,
            "m": args.batch_size if args.method == "m_sam" else "",
            "epoch": epoch,
            "update": global_update,
            "train_loss_step": epoch_loss,
            "train_acc_step": epoch_acc,
            "train_eval_loss": train_eval["loss"],
            "train_eval_acc": train_eval["acc"],
            "test_loss": test_eval["loss"],
            "test_acc": test_eval["acc"],
            "generalization_gap": train_eval["acc"] - test_eval["acc"],
            "elapsed_sec": time.time() - start_time,
        }
        write_row(metrics_path, row)
        print(json.dumps(row, indent=2))

    return {
        "final_train_acc": train_eval["acc"],
        "final_test_acc": test_eval["acc"],
        "final_gap": train_eval["acc"] - test_eval["acc"],
        "updates": global_update,
        "epochs": args.epochs,
    }


def train_same_updates(args, model, optimizer, criterion, loaders, device, output_dir):
    train_loader, full_train_loader, train_eval_loader, test_loader, train_size = loaders
    metrics_path = os.path.join(output_dir, "metrics.csv")

    train_iter = cycle(train_loader)
    start_time = time.time()

    for update in range(1, args.total_updates + 1):
        if args.method == "n_sam":
            stats = nsam_fullbatch_step(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                full_train_loader=full_train_loader,
                device=device,
                train_size=train_size,
                rho=args.rho,
            )
            step_loss = stats["first_loss"]
            step_acc = stats["first_acc"]

        else:
            batch = next(train_iter)
            if args.method == "sgd":
                step_loss, step_acc = sgd_step(model, optimizer, criterion, batch, device)
            elif args.method == "m_sam":
                step_loss, step_acc = msam_step(model, optimizer, criterion, batch, device)
            else:
                raise ValueError(f"Unknown method: {args.method}")

        should_eval = (
            update == 1
            or update == args.total_updates
            or update % args.eval_every_updates == 0
        )

        if should_eval:
            train_eval = evaluate(model, train_eval_loader, criterion, device)
            test_eval = evaluate(model, test_loader, criterion, device)

            row = {
                "comparison": "same_updates",
                "method": args.method,
                "m": args.batch_size if args.method == "m_sam" else "",
                "epoch": "",
                "update": update,
                "train_loss_step": step_loss,
                "train_acc_step": step_acc,
                "train_eval_loss": train_eval["loss"],
                "train_eval_acc": train_eval["acc"],
                "test_loss": test_eval["loss"],
                "test_acc": test_eval["acc"],
                "generalization_gap": train_eval["acc"] - test_eval["acc"],
                "elapsed_sec": time.time() - start_time,
            }
            write_row(metrics_path, row)
            print(json.dumps(row, indent=2))

    return {
        "final_train_acc": train_eval["acc"],
        "final_test_acc": test_eval["acc"],
        "final_gap": train_eval["acc"] - test_eval["acc"],
        "updates": args.total_updates,
        "epochs": None,
    }


def build_optimizer(args, model):
    if args.method == "m_sam":
        return SAM(
            model.parameters(),
            optim.SGD,
            rho=args.rho,
            adaptive=False,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    return optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, required=True, choices=["sgd", "n_sam", "m_sam"])
    parser.add_argument("--comparison", type=str, required=True, choices=["same_epochs", "same_updates"])

    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs/debug")
    parser.add_argument("--train-subset", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--total-updates", type=int, default=50)
    parser.add_argument("--eval-every-updates", type=int, default=10)

    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--rho", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    loaders = get_cifar10_loaders(
        data_dir=args.data_dir,
        train_batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        micro_batch_size=args.micro_batch_size,
        train_subset=args.train_subset,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = resnet18_gn_cifar10(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args, model)

    config = vars(args).copy()
    config["device_used"] = str(device)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    if args.comparison == "same_epochs":
        summary = train_same_epochs(args, model, optimizer, criterion, loaders, device, args.output_dir)
    else:
        summary = train_same_updates(args, model, optimizer, criterion, loaders, device, args.output_dir)

    summary.update(config)
    save_summary(args.output_dir, summary)
    print("Final summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()