"""Run the parity tests of the current commit on Modal.

    python parity/modal/run_modal.py [--tests a,b]

The image fetches HEAD from the remote that has it and runs
`parity/modal/build.sh`. `run.py --weights-only` fills the volume
`tyr-parity-weights`; then `run.py` runs on a CPU machine and on an H100 at
once, and their results are copied from the volume `tyr-parity-out` to
`output/parity/<sha>/{cpu,gpu}/`. Exits nonzero if a step fails or any case is
FAIL or ERROR.

Runs itself in `parity/modal/.venv`, created on first use with
`parity/modal/requirements.txt`. `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` (a
Modal API token) and `HF_TOKEN` (gated Hugging Face repositories) are read from
the environment or the repository's `.env`."""

import argparse
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import env, venv  # noqa: E402

venv.ensure(Path(__file__).resolve().parent / ".venv", Path(__file__).resolve().parent / "requirements.txt")
env.load()

import modal  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
# Read only here, never in the image, so they may differ from the commit.
LOCAL_ONLY = ("parity/modal/run_modal.py", "parity/modal/requirements.txt")
BASE_IMAGE = "nvidia/cuda:13.0.2-devel-ubuntu24.04"
CHECKOUT = "/opt/tyr"
ENVS = "/opt/parity-envs"
WEIGHTS = "/weights"
OUT = "/out"
TIMEOUT = 6 * 3600


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def https_url(url: str) -> str:
    if url.startswith("git@"):
        host, path = url[len("git@"):].split(":", 1)
        return f"https://{host}/{path}"
    if url.startswith("ssh://"):
        return "https://" + url[len("ssh://"):].split("@", 1)[-1]
    return url


def checked_commit() -> tuple[str, str]:
    """HEAD and an https URL of a remote with a branch containing it; exits if
    the tree has other changes or no remote branch contains HEAD."""
    dirty = [line[3:] for line in git("status", "--porcelain").splitlines() if line[3:] not in LOCAL_ONLY]
    if dirty:
        sys.exit("the working tree has changes besides " + " and ".join(LOCAL_ONLY) + ":\n  " + "\n  ".join(dirty))
    sha = git("rev-parse", "HEAD").strip()
    branches = [line.strip().split(" -> ")[0] for line in git("branch", "-r", "--contains", sha).splitlines()]
    if not branches:
        sys.exit(f"{sha} is on no remote-tracking branch; push it first")
    remote = branches[0].split("/", 1)[0]
    return sha, https_url(git("remote", "get-url", remote).strip())


def build_image(sha: str, url: str) -> modal.Image:
    toolchain = (REPO / "lean-toolchain").read_text().strip()
    return (
        modal.Image.from_registry(BASE_IMAGE, add_python="3.12")
        # The base image's entrypoint prints a banner and a missing-driver warning on CPU machines.
        .entrypoint([])
        .apt_install("ca-certificates", "git", "curl", "unzip", "make", "g++")
        .run_commands("curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh"
                      " | sh -s -- -y --default-toolchain none")
        .env({"PATH": "/root/.elan/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
        .run_commands(f"elan toolchain install {toolchain}")
        .run_commands(
            f"git init -q {CHECKOUT} && cd {CHECKOUT} && git fetch -q --depth 1 {url} {sha}"
            " && git checkout -q --detach FETCH_HEAD && parity/modal/build.sh"
        )
    )


def start(label: str, command: str, mounts: list[str], **options) -> modal.Sandbox:
    """Start `command` in the checkout; its output is printed prefixed with
    `label`, and the mounted volumes are synced before it exits."""
    script = f"{command} 2>&1; status=$?; sync {' '.join(mounts)}; exit $status"
    sandbox = modal.Sandbox.create("bash", "-c", script, workdir=CHECKOUT, timeout=TIMEOUT, **options)
    threading.Thread(target=lambda: [print(f"[{label}] {line}", end="", flush=True) for line in sandbox.stdout],
                     daemon=True).start()
    return sandbox


def finish(sandbox: modal.Sandbox) -> int:
    sandbox.wait(raise_on_termination=False)
    return sandbox.returncode if sandbox.returncode is not None else 1


def download(volume: modal.Volume, source: str, target: Path) -> None:
    shutil.rmtree(target, ignore_errors=True)
    for entry in volume.listdir(source, recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        path = target / Path(entry.path).relative_to(source)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            for chunk in volume.read_file(entry.path):
                f.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", help="comma-separated test names; default: all")
    args = parser.parse_args()
    sha, url = checked_commit()

    app = modal.App.lookup("tyr-parity", create_if_missing=True)
    weights = modal.Volume.from_name("tyr-parity-weights", create_if_missing=True)
    out = modal.Volume.from_name("tyr-parity-out", create_if_missing=True)
    common = {"app": app, "image": build_image(sha, url)}

    hf_secret = [modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})] if "HF_TOKEN" in os.environ else []

    with modal.enable_output():
        selection = f" --tests {args.tests}" if args.tests else ""
        fetch = start("weights", f"python3 parity/run.py --envs {ENVS} --weights-only --weights {WEIGHTS}{selection}", [WEIGHTS],
                        volumes={WEIGHTS: weights}, secrets=hf_secret,
                        cpu=4.0, **common)
        if finish(fetch) != 0:
            print("fetching weights failed", file=sys.stderr)
            return 1

        volumes = {WEIGHTS: weights.read_only(), OUT: out}
        runs = {
            device: start(device, f"python3 parity/run.py --envs {ENVS} --device {kind} --weights {WEIGHTS} --out {OUT}{selection}",
                          [OUT], volumes=volumes, **resources, **common)
            for device, kind, resources in (
                ("cpu", "cpu", {"cpu": 16.0, "memory": 65536}),
                ("gpu", "cuda", {"gpu": "H100", "cpu": 8.0, "memory": 65536}),
            )
        }
        codes = {device: finish(sandbox) for device, sandbox in runs.items()}

    for device in runs:
        try:
            download(out, f"{sha}/{device}", REPO / "output" / "parity" / sha / device)
        except Exception as error:
            print(f"{device}: no results downloaded: {error}", file=sys.stderr)
            codes[device] = codes[device] or 1
    for device, code in codes.items():
        print(f"{device}: exit {code}")
    return 0 if all(code == 0 for code in codes.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
