"""scripts/push_model.py — publish the trained classifier to HF Hub.

Design decision (worth defending in the report):
  We push ONLY the trained classifier + metadata + the helper module needed
  to build inference features. We do NOT push the local copy of e5-small-v2
  (130MB) — it's a frozen public dependency. Users can fetch it directly from
  `intfloat/e5-small-v2`. This keeps our model repo small (~5MB) and focused
  on what's novel: the trained sklearn pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi

# --- Load HF_TOKEN from .env ---
ENV_PATH = Path(__file__).parent.parent / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("HF_TOKEN not found in .env")

REPO_ID = "andreiaalexa/scifact-relevance-classifier"
ROOT = Path(__file__).parent.parent
ARTEFACTS = ROOT / "artefacts_scifact"


def main():
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    print(f"Repo ready: https://huggingface.co/{REPO_ID}\n")

    # (path_in_repo, local_path) pairs
    STANCE = ROOT / "artefacts_stance"
    uploads = [
        # Binary relevance classifier (original task)
        ("classifier.pkl",            ARTEFACTS / "classifier.pkl"),
        ("metadata.json",             ARTEFACTS / "metadata.json"),
        ("experiment_results.json",   ARTEFACTS / "experiment_results.json"),
        # 3-class stance classifier (NEI / REFUTES / SUPPORTS)
        ("classifier_stance.pkl",          STANCE / "classifier_stance.pkl"),
        ("metadata_stance.json",           STANCE / "metadata_stance.json"),
        ("experiment_results_stance.json", STANCE / "experiment_results_stance.json"),
        # Feature builder used by both classifiers
        ("scifact_features.py",       ROOT / "scifact_features.py"),
        # Pre-encoded SciFact corpus so the Space can retrieve without re-encoding
        ("corpus_embeddings.npy",     ARTEFACTS / "corpus_embeddings.npy"),
        ("corpus_meta.csv",           ARTEFACTS / "corpus_meta.csv"),
    ]

    for repo_path, local_path in uploads:
        if not local_path.exists():
            raise FileNotFoundError(f"Missing: {local_path}")
        size_kb = local_path.stat().st_size / 1024
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type="model",
            token=HF_TOKEN,
            commit_message=f"upload {repo_path}",
        )
        print(f"  uploaded {repo_path:30s}  ({size_kb:8.1f} KB)")

    # Upload model card as the repo README
    card = Path(__file__).parent / "model_card.md"
    if card.exists():
        api.upload_file(
            path_or_fileobj=str(card),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="model",
            token=HF_TOKEN,
            commit_message="docs: model card",
        )
        print(f"  uploaded model card")
    else:
        print(f"  WARN: no model_card.md found at {card}")

    print(f"\nDone. View at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
