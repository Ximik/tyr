"""Venvs created on first use and reinstalled when their requirements change."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def prepare(venv: Path, requirements: Path) -> Path:
    """The Python of `venv`, created if missing and given `requirements` if they
    changed since the last install (without pip's download cache)."""
    python = venv / "bin" / "python"
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    installed = venv / requirements.name
    if not installed.exists() or installed.read_text() != requirements.read_text():
        # Wheels unpack next to the venv: a small tmpfs `/tmp` cannot hold the CUDA ones.
        scratch = venv / ".pip-tmp"
        scratch.mkdir(exist_ok=True)
        try:
            subprocess.run([str(python), "-m", "pip", "install", "--quiet", "--no-cache-dir",
                            "--disable-pip-version-check", "-r", str(requirements)],
                           check=True, env=os.environ | {"TMPDIR": str(scratch)})
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        shutil.copyfile(requirements, installed)
    return python


def ensure(venv: Path, requirements: Path) -> None:
    """Return if this process runs in `venv`; otherwise prepare it and re-run
    this script in it."""
    if Path(sys.prefix).resolve() == venv.resolve():
        return
    python = prepare(venv, requirements)
    os.execv(python, [str(python), sys.argv[0], *sys.argv[1:]])
