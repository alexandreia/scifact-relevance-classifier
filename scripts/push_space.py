"""scripts/push_space.py — create and populate the Gradio Space.

A HF Space is a separate repo (different namespace from datasets/ and models/).
We create it with sdk='gradio' and upload three files:

  scripts/space_app.py         -> app.py
  scripts/space_requirements.txt -> requirements.txt
  scripts/space_card.md        -> README.md (with YAML frontmatter for SDK config)

After this script runs, HF will install the requirements and start the app.
First boot takes ~3-5 minutes (it has to download torch + e5-small-v2).
Subsequent visits are instant once the Space is warm.
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

REPO_ID = "andreiaalexa/scifact-relevance-classifier"  # under spaces/, not models/
SCRIPTS = Path(__file__).parent


def main():
    api = HfApi(token=HF_TOKEN)

    # Create the Space repo. sdk="gradio" tells HF this is a Gradio Space.
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
    )
    print(f"Space repo ready: https://huggingface.co/spaces/{REPO_ID}\n")

    uploads = [
        ("app.py",            SCRIPTS / "space_app.py"),
        ("requirements.txt",  SCRIPTS / "space_requirements.txt"),
        ("README.md",         SCRIPTS / "space_card.md"),
    ]
    for repo_path, local_path in uploads:
        if not local_path.exists():
            raise FileNotFoundError(f"Missing source file: {local_path}")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type="space",
            token=HF_TOKEN,
            commit_message=f"upload {repo_path}",
        )
        print(f"  uploaded {repo_path}")

    print(f"\nDone. The Space will start building automatically.")
    print(f"View at: https://huggingface.co/spaces/{REPO_ID}")
    print("\nFirst build will take ~3-5 min (downloading torch + e5-small-v2).")
    print("Watch the build logs in the Space UI under the 'Logs' tab.")


if __name__ == "__main__":
    main()
