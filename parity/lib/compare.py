"""Checks that compare a case's `reference/` and `tyr/` outputs and return a
verdict with the measured numbers."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lib import probe

VERDICTS = ("PASS", "TIE", "FAIL", "ERROR")


def worst(verdicts) -> str:
    return max(verdicts, key=VERDICTS.index, default="PASS")


@dataclass(frozen=True)
class LmTokens:
    """Greedy tokens must match. Where they first differ, the case is a TIE if
    the reference's top-1/top-2 logit gap there is below `margin` and the
    forced run's logits are within `logits_atol` of the reference at every
    step; on CPU there are no ties.

    Reads `tokens` from both `result.json` files, `logits` (`[batch, steps,
    vocab]`) from both tensor files and `margin` (`[batch, steps]`) from the
    reference's."""

    margin: float
    logits_atol: float
    name: str = "lm_tokens"

    def __call__(self, case_dir: Path, device: str) -> dict:
        ref, ref_tensors = probe.read(case_dir / "reference")
        tyr, tyr_tensors = probe.read(case_dir / "tyr")
        ref_tokens = np.asarray(ref["tokens"])
        tyr_tokens = np.asarray(tyr["tokens"])
        ref_logits = ref_tensors["logits"]
        tyr_logits = tyr_tensors["logits"]
        if ref_tokens.shape != tyr_tokens.shape:
            raise ValueError(f"tokens: reference {ref_tokens.shape}, tyr {tyr_tokens.shape}")
        if ref_logits.shape != tyr_logits.shape:
            raise ValueError(f"logits: reference {ref_logits.shape}, tyr {tyr_logits.shape}")

        step_diff = np.abs(ref_logits.astype(np.float64) - tyr_logits).max(axis=-1)
        max_diff = float(step_diff.max())
        rows = []
        for row in range(ref_tokens.shape[0]):
            differ = np.flatnonzero(ref_tokens[row] != tyr_tokens[row])
            entry = {"max_logit_diff": float(step_diff[row].max())}
            if differ.size == 0:
                entry["verdict"] = "PASS"
            else:
                step = int(differ[0])
                margin = float(ref_tensors["margin"][row, step])
                tie = device != "cpu" and margin < self.margin and max_diff <= self.logits_atol
                entry.update(
                    verdict="TIE" if tie else "FAIL",
                    step=step,
                    reference_token=int(ref_tokens[row, step]),
                    tyr_token=int(tyr_tokens[row, step]),
                    margin=margin,
                )
            rows.append(entry)
        return {"verdict": worst(r["verdict"] for r in rows), "max_logit_diff": max_diff, "rows": rows}
