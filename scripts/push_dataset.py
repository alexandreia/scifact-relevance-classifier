"""scripts/push_dataset.py — publish the custom SciFact pairs as a HF dataset.

What this script does
---------------------
For each field variant (`title`, `abstract`, `title_abstract`):
  1. Loads `data_scifact/scifact_pairs_<variant>.csv` (built by scifact_dataset.py)
  2. Splits it into `train` / `test` Datasets using the existing `split` column
  3. Pushes those splits as a *config* of the dataset repo on HF Hub

Result on the Hub: ONE dataset repo with THREE configs. Anyone can do:

    from datasets import load_dataset
    ds = load_dataset("alexandreia/scifact-relevance-pairs", "title_abstract")
    ds["train"]   # the same 2537 rows we trained on, deterministically

Then the script uploads `scripts/dataset_card.md` as the dataset README.

How to run
----------
    cd "Assignment 1"
    source venv/bin/activate
    python scripts/push_dataset.py

Requires:
    - HF_TOKEN set in .env (already configured)
    - The `datasets` and `huggingface_hub` packages (in requirements.txt)
    - data_scifact/*.csv built (run scifact_dataset.py first if missing)
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

# Read HF_TOKEN from .env (no extra dependency needed — minimal parser).
ENV_PATH = Path(__file__).parent.parent / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit(
        "HF_TOKEN not found. Add it to .env (e.g. HF_TOKEN=hf_...) "
        "or export it in your shell before running this script."
    )

# Edit this if you'd prefer a different repo name.
REPO_ID = "andreiaalexa/scifact-relevance-pairs"

VARIANTS = ("title", "abstract", "title_abstract")
DATA_DIR = Path(__file__).parent.parent / "data_scifact"


def build_dataset_dict(csv_path: Path) -> DatasetDict:
    """Read one variant CSV and split it into a HF DatasetDict by `split` column."""
    df = pd.read_csv(csv_path)
    print(f"  loaded {len(df)} rows from {csv_path.name}")
    print(f"    split counts: {df['split'].value_counts().to_dict()}")
    print(f"    label counts: {df['label'].value_counts().to_dict()}")

    splits = {}
    for split_name in ("train", "test"):
        sub = df[df["split"] == split_name].reset_index(drop=True)
        splits[split_name] = Dataset.from_pandas(sub, preserve_index=False)
    return DatasetDict(splits)


def main():
    print(f"Target dataset repo: https://huggingface.co/datasets/{REPO_ID}")

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
    print("Repo ready.\n")

    for variant in VARIANTS:
        csv_path = DATA_DIR / f"scifact_pairs_{variant}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} missing. Run `python scifact_dataset.py` first."
            )
        print(f"Variant: {variant}")
        dsd = build_dataset_dict(csv_path)
        dsd.push_to_hub(REPO_ID, config_name=variant, token=HF_TOKEN)
        print(f"  pushed config '{variant}' to {REPO_ID}\n")

    # Upload the dataset card (README on the dataset's HF page).
    card_path = Path(__file__).parent / "dataset_card.md"
    if card_path.exists():
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN,
            commit_message="docs: dataset card",
        )
        print(f"Uploaded dataset card from {card_path.name}")
    else:
        print(f"NOTE: no dataset card found at {card_path}; skipping.")

    print(f"\nDone. View at: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
