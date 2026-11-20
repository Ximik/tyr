"""Venvs created on first use and reinstalled when their requirements change."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# The locks in `parity/envs` are resolved for this Python.
PYTHON = "3.12"


def prepare(venv: Path, requirements: Path) -> Path:
    """The Python of `venv`, a uv venv on Python `PYTHON` synced to
    `requirements` if they changed since the last sync. uv downloads that Python
    when the system has none; it and uv's cache stay next to `venv`."""
    from uv import find_uv_bin

    uv = find_uv_bin()
    env = os.environ | {"UV_CACHE_DIR": str(venv.parent / ".uv-cache"),
                        "UV_PYTHON_INSTALL_DIR": str(venv.parent / ".python")}
    python = venv / "bin" / "python"
    installed = venv / requirements.name
    if not python.exists() or not _runs(python, PYTHON):
        shutil.rmtree(venv, ignore_errors=True)
        subprocess.run([uv, "venv", "--quiet", "--python", PYTHON, str(venv)], check=True, env=env)
    if not installed.exists() or installed.read_text() != requirements.read_text():
        subprocess.run([uv, "pip", "sync", "--quiet", "--python", str(python),
                        "--index-strategy", "unsafe-best-match", str(requirements)], check=True, env=env)
        shutil.copyfile(requirements, installed)
    return python


def _runs(python: Path, version: str) -> bool:
    found = subprocess.run([str(python), "-c", "import sys; print(*sys.version_info[:2], sep='.')"],
                           capture_output=True, text=True)
    return found.returncode == 0 and found.stdout.strip() == version


def ensure(venv: Path, requirements: Path) -> None:
    """Return if this process runs in `venv`; otherwise create it with this
    Python if missing, give it `requirements` if they changed since the last
    install, and re-run this script in it."""
    if Path(sys.prefix).resolve() == venv.resolve():
        return
    python = venv / "bin" / "python"
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    installed = venv / requirements.name
    if not installed.exists() or installed.read_text() != requirements.read_text():
        subprocess.run([str(python), "-m", "pip", "install", "--quiet", "--no-cache-dir",
                        "--disable-pip-version-check", "-r", str(requirements)], check=True)
        shutil.copyfile(requirements, installed)
    os.execv(python, [str(python), sys.argv[0], *sys.argv[1:]])
