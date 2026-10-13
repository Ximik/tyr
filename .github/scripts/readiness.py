#!/usr/bin/env python3
"""Hosted preflight: distinguish an unconfigured repo from executed qualification."""
import json
import os
from pathlib import Path


def configuration(repository, labels):
    if not labels:
        if repository != "cpehle/tyr":
            raise ValueError("Set TYR_QUALIFICATION_RUNNER_LABELS for this repository; "
                             "Spark's default labels apply only to cpehle/tyr.")
        labels = '["self-hosted", "Linux", "ARM64", "gpu", "gb10"]'
    parsed = json.loads(labels)
    if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed) or \
            "self-hosted" not in parsed or not {"gpu", "tyr-qualification"}.intersection(parsed):
        raise ValueError("Runner labels must include self-hosted and gpu or tyr-qualification")
    return json.dumps(parsed)


def main():
    report = {"status": "blocked", "executed": 0, "skipped": 0,
              "repository": os.environ.get("GITHUB_REPOSITORY", "")}
    try:
        labels = configuration(report["repository"],
                               os.environ.get("TYR_QUALIFICATION_RUNNER_LABELS", ""))
        report.update(status="configured", runner_labels=json.loads(labels))
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as out:
                out.write(f"runner_labels={labels}\n")
    except (ValueError, TypeError) as error:
        report["reason"] = str(error)
    path = Path("output/qualification/readiness.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    message = "Qualification readiness: " + json.dumps(report)
    print(message)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as out:
            out.write(message + "\n\nConfigured is not qualified: only the GPU job records executed tests.\n")
    return 0 if report["status"] == "configured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
