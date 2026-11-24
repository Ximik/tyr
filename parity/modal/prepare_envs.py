"""Create the orchestrator venv and the venv of every `parity/envs/*.txt` under
the directory given as the only argument, as `run.py` lays them out. The Modal
image runs this before checking out the commit, so the layer holding the venvs
is reused while their requirements are unchanged."""

import sys
from pathlib import Path

PARITY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PARITY))
from lib import venv  # noqa: E402

root = Path(sys.argv[1])
venv.ensure(root / "orchestrator", PARITY / "requirements.txt")
for requirements in sorted((PARITY / "envs").glob("*.txt")):
    venv.prepare(root / requirements.stem, requirements)
