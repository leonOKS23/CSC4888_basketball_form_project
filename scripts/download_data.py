"""Download pinned trial JSON files; verify Git blob hashes and resume safely."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen
import time

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / 'scripts/dataset_manifest.json').read_text())


def valid(data, entry):
    header = f'blob {len(data)}\0'.encode()
    return hashlib.sha1(header + data).hexdigest() == entry['sha']


def download(entry):
    relative = entry['path'].split('basketball/freethrow/data/', 1)[1]
    target = ROOT / 'data/raw' / relative
    if target.exists() and valid(target.read_bytes(), entry):
        return
    url = f"https://raw.githubusercontent.com/Sport-Performance-Lab/SPL-Open-Data/{MANIFEST['commit']}/{entry['path']}"
    for attempt in range(4):
        try:
            with urlopen(url, timeout=90) as response:
                data = response.read()
            if not valid(data, entry):
                raise ValueError(f'Hash mismatch: {relative}')
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.part')
            temporary.write_bytes(data)
            temporary.replace(target)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, _ in enumerate(pool.map(download, MANIFEST['files']), 1):
            if i % 25 == 0 or i == len(MANIFEST['files']):
                print(f"Verified {i}/{len(MANIFEST['files'])} trials", flush=True)
