"""Running a script inside its own venv, created on first use."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def ensure(venv: Path, requirements: Path) -> None:
    """Return if this process runs in `venv`; otherwise create `venv` if
    missing, install `requirements` if they changed since the last install
    (without pip's download cache), and re-run this script in it."""
    if Path(sys.prefix).resolve() == venv.resolve():
        return
    python = venv / "bin" / "python"
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    installed = venv / requirements.name
    if not installed.exists() or installed.read_text() != requirements.read_text():
        subprocess.run([str(python), "-m", "pip", "install", "--quiet", "--no-cache-dir", "--disable-pip-version-check",
                        "-r", str(requirements)], check=True)
        shutil.copyfile(requirements, installed)
    os.execv(python, [str(python), sys.argv[0], *sys.argv[1:]])
