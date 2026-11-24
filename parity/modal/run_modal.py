"""Run the parity tests of the current commit on Modal.

    python parity/modal/run_modal.py start [--tests a,b]
    python parity/modal/run_modal.py status <sandbox id>
    python parity/modal/run_modal.py download <sandbox id> [--tests a,b] [--tensors]

`start` builds the image, which holds the toolchain, the native dependencies
and the venvs but no commit. A build sandbox then fetches HEAD from the remote
that has it into the tree of the last build in the volume `tyr-parity-build`,
runs `parity/modal/build.sh`, so only what changed is rebuilt, and stores the
tree there under the commit. `start` then fills the volume `tyr-parity-weights`
with `run.py --weights-only`, starts `run.py` on a CPU machine and on an H100,
each on a copy of that tree, and prints the two sandboxes' ids; it does not
wait for them. Each writes its results under its id in the volume
`tyr-parity-out`, as `run.py` lays them out (`<id>/<sha>/<device>/`), committed
every minute. `status` shows whether a sandbox is running or its exit code, and
the verdicts written so far; `download` copies its results, without the tensor
files unless `--tensors`, to `output/parity/<id>/`.

Runs itself in `parity/modal/.venv`, created on first use with
`parity/modal/requirements.txt`. `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` (a
Modal API token) and `HF_TOKEN` (gated Hugging Face repositories) are read from
the environment or the repository's `.env`."""

import argparse
import os
import json
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
BUILD = "/build"
# Builds the build volume keeps; the runs of a build copy it right after it is stored.
KEEP_BUILDS = 3
TIMEOUT = 6 * 3600
RUNS = (("cpu", "cpu", {"cpu": 16.0, "memory": 65536}), ("gpu", "cuda", {"gpu": "H100", "cpu": 8.0, "memory": 65536}))


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
    the tree has other changes or no remote branch contains HEAD. Markdown
    changes are allowed: nothing the image builds or runs reads them."""
    dirty = [path for path in (line[3:] for line in git("status", "--porcelain").splitlines())
             if path not in LOCAL_ONLY and not path.endswith(".md")]
    if dirty:
        sys.exit("the working tree has changes besides " + ", ".join(LOCAL_ONLY) + " and Markdown files:\n  "
                 + "\n  ".join(dirty))
    sha = git("rev-parse", "HEAD").strip()
    branches = [line.strip().split(" -> ")[0] for line in git("branch", "-r", "--contains", sha).splitlines()]
    if not branches:
        sys.exit(f"{sha} is on no remote-tracking branch; push it first")
    remote = branches[0].split("/", 1)[0]
    return sha, https_url(git("remote", "get-url", remote).strip())


def build_image() -> modal.Image:
    toolchain = (REPO / "lean-toolchain").read_text().strip()
    return (
        modal.Image.from_registry(BASE_IMAGE, add_python="3.12")
        # The base image's entrypoint prints a banner and a missing-driver warning on CPU machines.
        .entrypoint([])
        .apt_install("ca-certificates", "git", "curl", "unzip", "make", "g++", "rsync")
        .run_commands("curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh"
                      " | sh -s -- -y --default-toolchain none")
        .env({"PATH": "/root/.elan/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
        .run_commands(f"elan toolchain install {toolchain}")
        .add_local_dir(REPO / "deps", f"{CHECKOUT}/deps", copy=True)
        .run_commands(f"cd {CHECKOUT} && TYR_DEPS_CACHE=/tmp/wheels deps/fetch.sh cuda && rm -rf /tmp/wheels deps")
        .add_local_file(REPO / "parity/requirements.txt", "/opt/parity-src/requirements.txt", copy=True)
        .add_local_dir(REPO / "parity/envs", "/opt/parity-src/envs", copy=True)
        .add_local_file(REPO / "parity/lib/__init__.py", "/opt/parity-src/lib/__init__.py", copy=True)
        .add_local_file(REPO / "parity/lib/venv.py", "/opt/parity-src/lib/venv.py", copy=True)
        .add_local_file(REPO / "parity/modal/prepare_envs.py", "/opt/parity-src/modal/prepare_envs.py", copy=True)
        .run_commands(f"python3 /opt/parity-src/modal/prepare_envs.py {ENVS} && rm -rf /opt/parity-src")
    )


def sandbox(command: str, mount: str, **options) -> modal.Sandbox:
    """Start `command` in the checkout; the volume at `mount` is committed every
    minute and before it exits."""
    script = (f"(while sleep 60; do sync {mount}; done) & syncer=$!; {command} 2>&1; status=$?; "
              f"kill $syncer; sync {mount}; exit $status")
    return modal.Sandbox.create("bash", "-c", script, workdir=CHECKOUT, timeout=TIMEOUT, **options)


def checkout(sha: str) -> str:
    """The command that copies the tree built for `sha` into the checkout. The
    image's `external/` is not in the build volume."""
    return f"rsync -a {BUILD}/builds/{sha}/ {CHECKOUT}/"


def wait(run: modal.Sandbox, label: str) -> int:
    """Print the sandbox's output and return its exit code."""
    reader = threading.Thread(target=lambda: [print(f"[{label}] {line}", end="", flush=True) for line in run.stdout])
    reader.start()
    run.wait(raise_on_termination=False)
    reader.join()
    return run.returncode if run.returncode is not None else 1


def build(sha: str, url: str, cache: modal.Volume, **options) -> int:
    """Build `sha` from the tree of the last stored build and store the result
    under `builds/<sha>`, then point `latest` at it. Probe build failures do
    not fail this step: `run.py` records them per test."""
    builds = f"{BUILD}/builds"
    script = " && ".join([
        f"latest=$(cat {BUILD}/latest 2>/dev/null || true)",
        f'if [ -n "$latest" ] && [ -d {builds}/$latest ]; then rsync -a {builds}/$latest/ ./; fi',
        "if [ ! -d .git ]; then git init -q; fi",
        f"git fetch -q --depth 1 {url} {sha}",
        "git checkout -q --force --detach FETCH_HEAD",
        "{ parity/modal/build.sh || echo 'some probes failed to build'; }",
        f"rm -rf {builds}/{sha}.partial",
        f"mkdir -p {builds}",
        f"rsync -a --exclude=/external ./ {builds}/{sha}.partial/",
        f"rm -rf {builds}/{sha}",
        f"mv {builds}/{sha}.partial {builds}/{sha}",
        f"echo {sha} > {BUILD}/latest.partial",
        f"mv {BUILD}/latest.partial {BUILD}/latest",
        f"(cd {builds} && ls -1t | grep -v '\\.partial$' | tail -n +{KEEP_BUILDS + 1} | xargs -r rm -rf)",
    ])
    return wait(sandbox(script, BUILD, volumes={BUILD: cache}, cpu=16.0, memory=65536, **options), "build")


def fetch_weights(sha: str, selection: str, weights: modal.Volume, cache: modal.Volume, **options) -> int:
    """Run `run.py --weights-only`, printing its output, and wait for it."""
    fetch = sandbox(f"{checkout(sha)} && python3 parity/run.py --envs {ENVS} --weights-only --weights {WEIGHTS}"
                    f"{selection}", WEIGHTS, volumes={WEIGHTS: weights, BUILD: cache.read_only()}, cpu=4.0,
                    **options)
    return wait(fetch, "weights")


def start(args) -> int:
    sha, url = checked_commit()
    app = modal.App.lookup("tyr-parity", create_if_missing=True)
    weights = modal.Volume.from_name("tyr-parity-weights", create_if_missing=True)
    out = modal.Volume.from_name("tyr-parity-out", create_if_missing=True)
    cache = modal.Volume.from_name("tyr-parity-build", create_if_missing=True)
    common = {"app": app, "image": build_image()}
    hf_secret = [modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})] if "HF_TOKEN" in os.environ else []
    selection = f" --tests {args.tests}" if args.tests else ""

    with modal.enable_output():
        if build(sha, url, cache, **common) != 0:
            print("building failed", file=sys.stderr)
            return 1
        if fetch_weights(sha, selection, weights, cache, secrets=hf_secret, **common) != 0:
            print("fetching weights failed", file=sys.stderr)
            return 1
    for device, kind, resources in RUNS:
        run = sandbox(f"{checkout(sha)} && python3 parity/run.py --envs {ENVS} --device {kind} --weights {WEIGHTS}"
                      f" --out {OUT}/$MODAL_SANDBOX_ID{selection}",
                      OUT, volumes={WEIGHTS: weights.read_only(), OUT: out, BUILD: cache.read_only()},
                      **resources, **common)
        run.set_tags({"sha": sha, "device": device, "tests": args.tests or ""})
        print(f"{device}: {run.object_id}")
    return 0


def tagged(sandbox_id: str) -> tuple[modal.Sandbox, str, str]:
    """The sandbox `start` created, its commit and its device."""
    run = modal.Sandbox.from_id(sandbox_id)
    tags = run.get_tags()
    if "sha" not in tags or "device" not in tags:
        sys.exit(f"{sandbox_id} is not a sandbox `start` created")
    return run, tags["sha"], tags["device"]


def results(out: modal.Volume, source: str) -> list[str]:
    """The files under `source` in the out volume, or none yet."""
    try:
        return [e.path for e in out.listdir(source, recursive=True) if e.type == modal.volume.FileEntryType.FILE]
    except Exception:
        return []


def status(args) -> int:
    run, sha, device = tagged(args.id)
    code = run.poll()
    print(f"{sha} {device}: " + ("running" if code is None else f"exited {code}"))
    out = modal.Volume.from_name("tyr-parity-out")
    source = f"{args.id}/{sha}/{device}"
    cases = (p for p in results(out, source)
             if p.endswith("/result.json") and Path(p).parent.name not in ("reference", "tyr"))
    for path in sorted(cases):
        case = str(Path(path).parent.relative_to(source))
        verdict = json.loads(b"".join(out.read_file(path))).get("verdict")
        print(f"  {case}: {verdict}")
    return 0


def download(args) -> int:
    _, sha, device = tagged(args.id)
    out = modal.Volume.from_name("tyr-parity-out")
    source = f"{args.id}/{sha}/{device}"
    target = REPO / "output" / "parity" / args.id / sha / device
    tests = set(args.tests.split(",")) if args.tests else None
    for test in tests or [""]:
        shutil.rmtree(target / test, ignore_errors=True)
    copied = 0
    for path in results(out, source):
        relative = Path(path).relative_to(source)
        if tests is not None and relative.parts[0] not in tests:
            continue
        if path.endswith(".safetensors") and not args.tensors:
            continue
        local = target / relative
        local.parent.mkdir(parents=True, exist_ok=True)
        with open(local, "wb") as f:
            for chunk in out.read_file(path):
                f.write(chunk)
        copied += 1
    print(f"{copied} files to {target}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    start_parser = commands.add_parser("start", help="build, fetch weights and start the CPU and GPU runs")
    start_parser.add_argument("--tests", help="comma-separated test names; default: all")
    status_parser = commands.add_parser("status", help="a run's state and its verdicts so far")
    status_parser.add_argument("id", help="a sandbox id `start` printed")
    download_parser = commands.add_parser("download", help="copy a run's results")
    download_parser.add_argument("id", help="a sandbox id `start` printed")
    download_parser.add_argument("--tests", help="comma-separated test names; default: all")
    download_parser.add_argument("--tensors", action="store_true", help="also the tensor files")
    args = parser.parse_args()
    return {"start": start, "status": status, "download": download}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
