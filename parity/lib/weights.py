"""Weight files from `manifest.json`, downloaded to `<root>/<key>/` and checked
against their sha256. A file verified for the same hash is not read again.
Gated repositories need `HF_TOKEN`."""

import hashlib
import json
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "manifest.json"
VERIFIED = ".verified.json"


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
    verified_path = target / VERIFIED
    verified = json.loads(verified_path.read_text()) if verified_path.exists() else {}
    for name, expected in entry["files"].items():
        path = target / name
        if verified.get(name) == expected and path.exists():
            continue
        print(f"{key}: {name}", flush=True)
        target.mkdir(parents=True, exist_ok=True)
        hf_hub_download(entry["repo"], name, revision=entry["revision"], local_dir=target)
        actual = sha256(path)
        if actual != expected:
            path.unlink()
            raise ValueError(f"{key}/{name}: sha256 {actual}, manifest {expected}")
        verified[name] = expected
        verified_path.write_text(json.dumps(verified, indent=2))
    return target
