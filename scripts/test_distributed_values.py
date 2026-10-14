#!/usr/bin/env python3
"""Run the two-rank Gloo value regressions with bounded lifetime and rank logs.

Build first: lake build --wfail RunDistributedValues
Run: lake env python3 scripts/test_distributed_values.py
"""

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]


def run_two_ranks(executable: Path, log_dir: Path, timeout: float, port: int | None = None) -> int:
    executable = executable.resolve()
    if not executable.is_file():
        raise ValueError(f"Missing {executable}; run lake build --wfail RunDistributedValues")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be finite and positive")
    if port is None:
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
    if not 0 < port < 65536:
        raise ValueError("Port must be in [1, 65535]")

    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    started = time.monotonic()
    failure = None
    with ExitStack() as files:
        try:
            for rank in range(2):
                log = files.enter_context((log_dir / f"rank-{rank}.log").open("w"))
                processes.append(subprocess.Popen(
                    [str(executable), str(rank), str(port)], cwd=REPO,
                    stdout=log, stderr=subprocess.STDOUT))
            while True:
                codes = [process.poll() for process in processes]
                if any(code is not None and code != 0 for code in codes):
                    failure = "A rank exited unsuccessfully"
                    break
                if all(code == 0 for code in codes):
                    break
                if time.monotonic() - started >= timeout:
                    failure = f"Two-rank regression exceeded {timeout:g} seconds"
                    break
                time.sleep(0.05)
        except BaseException as error:
            failure = str(error) or type(error).__name__
            raise
        finally:
            # A failed rank must not leave its peer blocked in rendezvous or a collective.
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            report = {
                "status": "failed" if failure else "passed",
                "failure": failure,
                "executable": str(executable),
                "port": port,
                "seconds": time.monotonic() - started,
                "ranks": [{"rank": rank, "exit_code": process.returncode,
                           "log": f"rank-{rank}.log"}
                          for rank, process in enumerate(processes)],
            }
            (log_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    for rank in range(len(processes)):
        print(f"--- rank {rank} ---", flush=True)
        print((log_dir / f"rank-{rank}.log").read_text(), end="", flush=True)
    if failure:
        print(failure, file=sys.stderr)
    print(f"Distributed value regressions: {report['status']} ({log_dir / 'report.json'})")
    return 1 if failure else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=REPO / ".lake/build/bin/RunDistributedValues")
    parser.add_argument("--log-dir", type=Path, default=REPO / "output/ci/distributed")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--port", type=int, help="Rendezvous port; otherwise select an available localhost port")
    args = parser.parse_args()
    try:
        return run_two_ranks(args.executable, args.log_dir, args.timeout, args.port)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    sys.exit(main())
