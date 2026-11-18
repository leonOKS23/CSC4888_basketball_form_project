# Basketball free-throw form project

Predict make/miss from body mechanics before release using SPL motion capture.
Current status: reproducible setup and data audit complete; release detection is
still an exploratory heuristic. Feature extraction and modeling are next.

## Setup and reproduce

Python 3.12 was used for verification. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
python scripts/inspect_data.py
```

The downloader retrieves 583 trial files (about 1.2 GiB) into ignored `data/raw/`.
It pins upstream commit `d510cc472f80efa867def667f6b3d806085cc94d`, verifies
each Git blob SHA, and skips already verified files when rerun. It needs internet
access; inspection and the notebook run offline afterward.

Open `basketball_shot_analyzer.ipynb` with the `.venv` Python interpreter in your
notebook editor. It loads the 458 trials from the 2025 session. The audit script
checks both sessions and writes small, tracked audit outputs to `data/processed/`.

See [data notes](DATA_NOTES.md) for findings and [tickets](TICKETS.md) for remaining
work. Dependencies are pinned in `requirements.txt`, including the notebook kernel.

Validation: every notebook code cell was executed sequentially with the project
interpreter and Matplotlib's noninteractive Agg backend. This checks computation
and plot generation, but does not check an interactive notebook editor.
