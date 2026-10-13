#!/usr/bin/env python3
"""Check test routing and build/run the required CPU suites from one manifest."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
IMPORT = re.compile(r"^import\s+(\S+)", re.MULTILINE)
TEST = re.compile(r"^\s*@\[test(?:_ignore|_should_error)?(?:\s|,|\])", re.MULTILINE)
MAIN = re.compile(r"^\s*(?:unsafe\s+)?def main\b", re.MULTILINE)
EXECUTABLE = re.compile(r"^lean_exe\s+(\w+)\s+where\s*\n\s*root\s*:=\s*`(Tests[\w.]*)", re.MULTILINE)


def closure(root: Path, modules: set[str]) -> set[str]:
    seen: set[str] = set()
    pending = list(modules)
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        seen.add(module)
        path = root / (module.replace(".", "/") + ".lean")
        if path.exists():
            pending.extend(IMPORT.findall(path.read_text()))
    return seen


def validate(root: Path, manifest: dict) -> list[str]:
    executables = dict(EXECUTABLE.findall((root / "lakefile.lean").read_text()))
    required = set(manifest["required"])
    optional = set(manifest["optional"])
    errors = []
    unknown = (required | optional) - executables.keys()
    if unknown:
        errors.append("Unknown test executable(s): " + ", ".join(sorted(unknown)))
    if required & optional:
        errors.append("Required and optional suites overlap")
    for target, reason in manifest["optional"].items():
        if not reason.strip():
            errors.append(f"Optional suite {target} needs a reason")

    required_modules = closure(root, {executables[t] for t in required if t in executables})
    optional_modules = closure(root, {executables[t] for t in optional if t in executables})
    test_modules = {
        str(path.relative_to(root).with_suffix("")).replace("/", ".")
        for path in (root / "Tests").rglob("*.lean") if TEST.search(path.read_text())
    }
    for module in sorted(test_modules - required_modules - optional_modules):
        errors.append(f"Unrouted @[test] module: {module}")
    for target, module in executables.items():
        path = root / (module.replace(".", "/") + ".lean")
        if not path.exists() or not MAIN.search(path.read_text()):
            errors.append(f"Test executable {target} has no main in {module}")
        if target in required | optional:
            continue
        # Focused runners are covered by their imported tests. Standalone
        # assertions (e.g. Laguna's mains) need an explicit suite assignment.
        imported_tests = closure(root, {module}) & test_modules
        if not imported_tests or not imported_tests <= required_modules | optional_modules:
            errors.append(f"Standalone test executable needs a suite assignment: {target}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--report", type=Path, help="Write per-suite status, timing and logs (with --run)")
    parser.add_argument("--wfail", action="store_true", help="Fail the build on any warning (with --build)")
    args = parser.parse_args()
    manifest = json.loads((REPO / "scripts/test_suites.json").read_text())
    errors = validate(REPO, manifest)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    targets = manifest["required"]
    print(f"Test inventory: {len(targets)} required, {len(manifest['optional'])} optional suites", flush=True)
    if args.build:
        return subprocess.call(["lake", "build", *(["--wfail"] if args.wfail else []), *targets], cwd=REPO)
    if args.run:
        report = {"suites": {target: {"status": "not_run"} for target in targets}}
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        for target, arguments in targets.items():
            print(f"Running required suite: {target}", flush=True)
            started = time.monotonic()
            command = [str(REPO / ".lake/build/bin" / target), *arguments]
            if args.report:
                log_path = args.report.parent / (target + ".log")
                with log_path.open("w") as log:
                    with subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, text=True) as process:
                        for line in process.stdout:
                            print(line, end="", flush=True)
                            log.write(line)
                        result = process.wait()
            else:
                result = subprocess.call(command, cwd=REPO)
            report["suites"][target] = {"status": "passed" if result == 0 else "failed",
                                       "exit_code": result, "seconds": time.monotonic() - started}
            if args.report:
                args.report.write_text(json.dumps(report, indent=2) + "\n")
            if result:
                return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
