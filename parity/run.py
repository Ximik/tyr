"""Run the parity tests here, the same way locally and in the Modal image.

    python parity/run.py [--tests a,b] [--device cpu|cuda ...]

Runs itself in the venv `<envs>/orchestrator` from `requirements.txt`. Each
test names an environment, `envs/<name>.txt`; its stages run in the venv
`<envs>/<name>`. Venvs are created on first use and reinstalled when their lock
changes. `HF_TOKEN`, for gated repositories, is read from the environment or the
repository's `.env`. Steps:
1. the venvs of the selected tests;
2. their weights, downloaded and checked (`lib/weights.py`);
3. `lake build` of their probes, with the checkout's Lake configuration;
4. for each device: each case's reference, then tyr, each in its own process,
   then the test's checks.
Devices default to CPU, plus CUDA when the build is a CUDA build and CUDA is
available. Results go to `<out>/<sha>/<cpu|gpu>/` (`-dirty` after the sha if
the tree has changes): `<test>/<case>/{reference,tyr}/`, their logs,
`<test>/<case>/result.json` (with each stage's `seconds`), `build.log`,
`run.json`. Exits 1 if any case is
FAIL or ERROR. `--envs-only`, `--weights-only` and `--build-only` do only steps
1, 2 and 3.

A test module in `tests/` defines `TEST` (`lib.test.Test`), `CHECKS` (callables
`(case_dir, device) -> {"verdict", ...}`) and
`reference(case, weights, device, dtype, out)` and `tyr(...)` with the same
arguments; `tyr` may read the reference output in `out.parent / "reference"`."""

import argparse
import importlib
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

PARITY = Path(__file__).resolve().parent
REPO = PARITY.parent
ENVS = PARITY / "envs"
DEFAULT_VENVS = REPO / "output" / "parity" / "envs"
sys.path.insert(0, str(PARITY))

from lib import venv  # noqa: E402


def venvs_root(argv: list[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--envs", type=Path, default=DEFAULT_VENVS)
    return parser.parse_known_args(argv)[0].envs


# Stage processes already run in their test's venv.
if "--stage" not in sys.argv[1:]:
    venv.ensure(venvs_root(sys.argv[1:]) / "orchestrator", PARITY / "requirements.txt")

from lib import env  # noqa: E402
from lib import weights as weight_files  # noqa: E402
from lib.compare import worst  # noqa: E402

STAGES = ("reference", "tyr")


def load_tests() -> dict:
    return {
        path.stem: importlib.import_module(f"tests.{path.stem}")
        for path in sorted((PARITY / "tests").glob("*.py"))
        if path.stem != "__init__"
    }


def run_stage(args) -> None:
    module = importlib.import_module(f"tests.{args.test}")
    getattr(module, args.stage)(
        PARITY / "cases" / f"{args.case}.json", Path(args.weights), args.device[0], args.dtype, Path(args.out),
    )


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def detect_devices() -> list[str]:
    devices = ["cpu"]
    cuda_build = (REPO / "external" / "wheels" / "torch" / "lib" / "libtorch_cuda.so").exists()
    gpu = shutil.which("nvidia-smi") is not None and subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True, text=True).stdout.startswith("GPU")
    if cuda_build and gpu:
        devices.append("cuda")
    return devices


def prepare_envs(tests: dict, root: Path) -> tuple[dict[str, Path], dict[str, str]]:
    """The Python of each environment the tests name, and errors by environment."""
    pythons, errors = {}, {}
    for name in sorted({m.TEST.env for m in tests.values()}):
        try:
            pythons[name] = venv.prepare(root / name, ENVS / f"{name}.txt")
        except Exception as error:
            errors[name] = f"environment {name}: {error}"
            print(errors[name], file=sys.stderr, flush=True)
    return pythons, errors


def fetch_weights(tests: dict, root: Path) -> dict[str, str]:
    """Errors by weights key; an empty map means every test's weights are in place."""
    errors = {}
    keys = {m.TEST.weights_for(stage) for m in tests.values() for stage in STAGES} - {None}
    for key in sorted(keys):
        try:
            weight_files.fetch(key, root)
        except Exception as error:
            errors[key] = f"weights {key}: {error}"
            print(errors[key], file=sys.stderr, flush=True)
    return errors


def build_probes(tests: dict) -> tuple[dict[str, str], str]:
    """Errors by probe name, and the build output."""
    errors, log = {}, ""
    for probe in sorted({m.TEST.probe for m in tests.values()}):
        result = subprocess.run(["lake", "build", f"Parity{probe}"], cwd=REPO,
                                capture_output=True, text=True)
        log += f"$ lake build Parity{probe}\n{result.stdout}{result.stderr}"
        if result.returncode != 0:
            errors[probe] = f"lake build Parity{probe} failed; see build.log"
            print(errors[probe], file=sys.stderr, flush=True)
    return errors, log


def run_case(module, python: Path, case: str, device: str, dtype: str, weights: Path, case_dir: Path) -> dict:
    """The case's result; `seconds` holds each stage's wall time, also on ERROR.
    `python` is the test's venv; `weights` the directory holding every manifest key."""
    seconds: dict[str, float] = {}
    try:
        return run_stages(module, python, case, device, dtype, weights, case_dir, seconds) | {"seconds": seconds}
    except Exception:
        return {"verdict": "ERROR", "error": traceback.format_exc(), "seconds": seconds}


def run_stages(module, python: Path, case: str, device: str, dtype: str, weights: Path, case_dir: Path,
               seconds: dict[str, float]) -> dict:
    for stage in STAGES:
        key = module.TEST.weights_for(stage)
        start = time.monotonic()
        with open(case_dir / f"{stage}.log", "w") as log:
            code = subprocess.run(
                [str(python), __file__, "--stage", stage, "--test", module.TEST.name, "--case", case,
                 "--device", device, "--dtype", dtype, "--weights", str(weights / key) if key else "",
                 "--out", str(case_dir / stage)],
                stdout=log, stderr=subprocess.STDOUT,
            ).returncode
        seconds[stage] = round(time.monotonic() - start, 1)
        if code != 0:
            raise RuntimeError(f"{stage} exited with {code}; see {stage}.log")
    start = time.monotonic()
    checks = {check.name: check(case_dir, device) for check in module.CHECKS}
    seconds["checks"] = round(time.monotonic() - start, 1)
    return {"verdict": worst(c["verdict"] for c in checks.values()), "checks": checks}


def run_device(tests: dict, pythons: dict[str, Path], device: str, failures: dict[str, str], build_log: str,
               weights_root: Path, out: Path, sha: str) -> bool:
    """Run every selected test on `device`; whether all cases passed or tied."""
    out.mkdir(parents=True)
    (out / "build.log").write_text(build_log)
    verdicts: dict[str, dict[str, str]] = {}
    for name, module in tests.items():
        if device not in module.TEST.devices:
            continue
        dtype = "float32" if device == "cpu" else module.TEST.gpu_dtype
        failure = failures.get(name)
        verdicts[name] = {}
        for case in module.TEST.cases:
            case_dir = out / name / case
            case_dir.mkdir(parents=True)
            if failure:
                result = {"verdict": "ERROR", "error": failure}
            else:
                result = run_case(module, pythons[module.TEST.env], case, device, dtype, weights_root, case_dir)
            (case_dir / "result.json").write_text(json.dumps(result, indent=2))
            verdicts[name][case] = result["verdict"]
            timing = " ".join(f"{k} {v}s" for k, v in result.get("seconds", {}).items())
            print(f"{device} {name} {case}: {result['verdict']} ({timing})", flush=True)

    run = {
        "sha": sha,
        "device": device,
        "envs": {name: (ENVS / f"{name}.txt").read_text().splitlines()
                 for name in sorted({tests[t].TEST.env for t in verdicts})},
        "tests": verdicts,
    }
    (out / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True))
    return all(v in ("PASS", "TIE") for t in verdicts.values() for v in t.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", help="comma-separated test names; default: all")
    parser.add_argument("--device", choices=("cpu", "cuda"), action="append",
                        help="repeatable; default: cpu, and cuda if built for and available")
    parser.add_argument("--weights", type=Path, default=REPO / "output" / "parity" / "weights")
    parser.add_argument("--out", type=Path, default=REPO / "output" / "parity")
    parser.add_argument("--envs", type=Path, default=DEFAULT_VENVS, help="where the venvs live")
    parser.add_argument("--envs-only", action="store_true")
    parser.add_argument("--weights-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--stage", choices=STAGES, help=argparse.SUPPRESS)
    parser.add_argument("--test", help=argparse.SUPPRESS)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument("--dtype", help=argparse.SUPPRESS)
    args = parser.parse_args()
    env.load()
    if args.stage:
        run_stage(args)
        return 0

    tests = load_tests()
    if args.tests:
        names = args.tests.split(",")
        unknown = [n for n in names if n not in tests]
        if unknown:
            print(f"unknown tests: {', '.join(unknown)}", file=sys.stderr)
            return 2
        tests = {n: tests[n] for n in names}

    if args.build_only:
        errors, log = build_probes(tests)
        print(log)
        return 1 if errors else 0
    pythons, env_errors = prepare_envs(tests, args.envs)
    if args.envs_only:
        return 1 if env_errors else 0
    weight_errors = fetch_weights(tests, args.weights)
    if args.weights_only:
        return 1 if weight_errors else 0
    build_errors, build_log = build_probes(tests)
    failures = {}
    for name, module in tests.items():
        failure = env_errors.get(module.TEST.env) or next(
            (weight_errors[k] for s in STAGES if (k := module.TEST.weights_for(s)) in weight_errors),
            None) or build_errors.get(module.TEST.probe)
        if failure:
            failures[name] = failure

    sha = git("rev-parse", "HEAD").strip() + ("-dirty" if git("status", "--porcelain").strip() else "")
    ok = True
    for device in args.device or detect_devices():
        out = args.out / sha / ("cpu" if device == "cpu" else "gpu")
        shutil.rmtree(out, ignore_errors=True)
        ok = run_device(tests, pythons, device, failures, build_log, args.weights, out, sha) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
