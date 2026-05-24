import argparse
import os
import subprocess
import sys
from datetime import datetime


def run(cmd, dry_run=False):
    print("\n" + "=" * 100)
    print(" ".join(cmd))
    print("=" * 100 + "\n")

    if not dry_run:
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--root-output-dir", type=str, default="./outputs")
    parser.add_argument("--train-subset", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--total-updates", type=int, default=50)
    parser.add_argument("--eval-every-updates", type=int, default=10)
    parser.add_argument("--micro-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_root = os.path.join(args.root_output_dir, f"sam_cifar10_{timestamp}")
    os.makedirs(exp_root, exist_ok=True)

    methods = [
        ("sgd", 128),
        ("n_sam", args.micro_batch_size),
        ("m_sam", 32),
        ("m_sam", 128),
        ("m_sam", 512),
    ]

    comparisons = ["same_epochs", "same_updates"]

    for comparison in comparisons:
        for method, batch_size in methods:
            name = method if method != "m_sam" else f"m_sam_m{batch_size}"
            out_dir = os.path.join(exp_root, comparison, name)

            cmd = [
                sys.executable,
                "train.py",
                "--method", method,
                "--comparison", comparison,
                "--data-dir", args.data_dir,
                "--output-dir", out_dir,
                "--train-subset", str(args.train_subset),
                "--batch-size", str(batch_size),
                "--micro-batch-size", str(args.micro_batch_size),
                "--eval-batch-size", str(args.eval_batch_size),
                "--num-workers", str(args.num_workers),
                "--epochs", str(args.epochs),
                "--total-updates", str(args.total_updates),
                "--eval-every-updates", str(args.eval_every_updates),
                "--lr", str(args.lr),
                "--rho", str(args.rho),
                "--seed", str(args.seed),
                "--device", args.device,
            ]

            run(cmd, dry_run=args.dry_run)

    print(f"\nAll runs finished. Results are under:\n{exp_root}")


if __name__ == "__main__":
    main()