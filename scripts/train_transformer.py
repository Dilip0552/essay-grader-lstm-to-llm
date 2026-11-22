"""Fine-tune a transformer on the nea folds. Needs a GPU (Kaggle/Colab: use notebooks/gpu_runner.ipynb).

python scripts/train_transformer.py --config configs/deberta_v3_base.yaml
python scripts/train_transformer.py --config configs/deberta_v3_base.yaml --folds 0 --seed 43   # extra seed
python scripts/train_transformer.py --config configs/deberta_v3_base.yaml --limit 64 --epochs 1  # quick test
"""
import argparse

from aes.models.transformer import train_fold
from aes.utils import load_config, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--seed", type=int, help="overrides config seed; results go to <name>_s<seed>")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--limit", type=int, help="smoke test: use only this many essays per split")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # Command-line overrides change the output folder name, so they never overwrite the main results.
    if args.seed is not None and args.seed != cfg["seed"]:
        cfg["seed"], cfg["name"] = args.seed, f"{cfg['name']}_s{args.seed}"
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.limit:
        cfg["name"] = f"smoke_{cfg['name']}"  # smoke_* folders are gitignored and left out of the tables
    for fold in args.folds:
        set_seed(cfg["seed"])  # re-seed per fold, so each fold is reproducible on its own
        train_fold(cfg, fold, limit=args.limit)


if __name__ == "__main__":
    main()
