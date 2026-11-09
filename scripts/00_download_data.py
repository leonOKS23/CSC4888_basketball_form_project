#!/usr/bin/env python3
"""Download the SPL Open Data free-throw dataset.

Clones (a sparse subset of) https://github.com/Sport-Performance-Lab/SPL-Open-Data
into ``data/raw/SPL-Open-Data`` using git's partial-clone + sparse-checkout
features, so only the ``basketball/freethrow`` folder's blobs are fetched
(the full repo includes other large binary assets and is ~1.5 GB; the
basketball/freethrow data folder alone is ~1.2 GB, mostly many individually
large per-trial JSON files).

Usage:
    python scripts/00_download_data.py
    python scripts/00_download_data.py --full   # full clone, no sparse-checkout

If you already have a local clone of SPL-Open-Data, you can instead just
point spl_ft/config.py's SPL_REPO_DIR at it, or symlink it into
data/raw/SPL-Open-Data.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spl_ft import config

REPO_URL = "https://github.com/Sport-Performance-Lab/SPL-Open-Data.git"


def run(cmd, **kwargs):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Full clone instead of sparse-checkout")
    parser.add_argument("--force", action="store_true", help="Remove existing clone first")
    args = parser.parse_args()

    dest = config.SPL_REPO_DIR
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if args.force:
            print(f"Removing existing {dest}")
            shutil.rmtree(dest)
        else:
            print(f"{dest} already exists. Use --force to re-download. Nothing to do.")
            return

    if shutil.which("git") is None:
        print("ERROR: git is not available on PATH. Cannot download the dataset.")
        print("Next step: install git, or manually download/extract the repo ZIP from")
        print(f"  {REPO_URL.replace('.git', '')}")
        print(f"into {dest}")
        sys.exit(1)

    if args.full:
        run(["git", "clone", "--depth", "1", REPO_URL, str(dest)])
    else:
        run([
            "git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout",
            REPO_URL, str(dest),
        ])
        run(["git", "sparse-checkout", "init", "--cone"], cwd=str(dest))
        run(["git", "sparse-checkout", "set", "basketball/freethrow"], cwd=str(dest))
        run(["git", "checkout"], cwd=str(dest))

    n_trials = len(list(config.FREETHROW_DATA_DIR.glob("**/*.json")))
    print(f"\nDownloaded. Found {n_trials} trial JSON files under {config.FREETHROW_DATA_DIR}")
    if n_trials == 0:
        print("WARNING: no trial files found — the dataset layout may have changed. "
              "Inspect data/raw/SPL-Open-Data manually.")


if __name__ == "__main__":
    main()
