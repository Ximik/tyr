"""Weight files from `manifest.json`, downloaded to `<root>/<key>/` and checked
against their sha256. A file verified for the same hash is not read again; the
record of that is one file per weight file under `.verified/`, so concurrent
runs writing a shared directory do not overwrite each other's records. Gated
repositories need `HF_TOKEN`."""

import hashlib
import json
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "manifest.json"
VERIFIED = ".verified"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(key: str, root: Path) -> Path:
    """The directory holding the verified files of manifest entry `key`;
    raises if a file cannot be downloaded or its hash differs."""
    from huggingface_hub import hf_hub_download

    entry = json.loads(MANIFEST.read_text())["weights"][key]
    target = root / key
    for name, expected in entry["files"].items():
        path = target / name
        record = target / VERIFIED / name
        if path.exists() and record.exists() and record.read_text() == expected:
            continue
        print(f"{key}: {name}", flush=True)
        target.mkdir(parents=True, exist_ok=True)
        hf_hub_download(entry["repo"], name, revision=entry["revision"], local_dir=target)
        actual = sha256(path)
        if actual != expected:
            path.unlink()
            raise ValueError(f"{key}/{name}: sha256 {actual}, manifest {expected}")
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(expected)
    return target
