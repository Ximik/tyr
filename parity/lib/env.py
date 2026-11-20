"""`KEY=VALUE` lines of the repository's `.env` (gitignored) into the
environment. Blank lines and `#` comments are skipped, quotes around a value
are removed, empty values are skipped, and variables already set are kept."""

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def load(path: Path = ENV_FILE) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError(f"{path}: expected KEY=VALUE, got {line!r}")
        value = value.strip().strip("'\"")
        if value:
            os.environ.setdefault(key.strip(), value)
