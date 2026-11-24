"""Checks that compare a case's `reference/` and `tyr/` outputs and return a
verdict with the measured numbers."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lib import probe

VERDICTS = ("PASS", "TIE", "FAIL", "ERROR")
CPU_LOGITS_ATOL = 1e-3


def worst(verdicts) -> str:
    return max(verdicts, key=VERDICTS.index, default="PASS")


@dataclass(frozen=True)
class LmTokens:
    """Greedy tokens must match and the forced run's logits must be within
    `logits_atol` of the reference at every step; on CPU, which runs in fp32,
    within `CPU_LOGITS_ATOL` instead. Where the tokens first differ, the case is
    a TIE if the reference's logit of tyr's token there is less than `margin`
    below its top logit and the logits are within `logits_atol`; on CPU there
    are no ties.

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
        limit = CPU_LOGITS_ATOL if device == "cpu" else self.logits_atol
        rows = []
        for row in range(ref_tokens.shape[0]):
            differ = np.flatnonzero(ref_tokens[row] != tyr_tokens[row])
            entry = {"max_logit_diff": float(step_diff[row].max())}
            if differ.size == 0:
                entry["verdict"] = "PASS" if entry["max_logit_diff"] <= limit else "FAIL"
            else:
                step = int(differ[0])
                reference_token, tyr_token = int(ref_tokens[row, step]), int(tyr_tokens[row, step])
                at_step = ref_logits[row, step].astype(np.float64)
                gap = float(at_step[reference_token] - at_step[tyr_token])
                tie = device != "cpu" and gap < self.margin and max_diff <= self.logits_atol
                entry.update(
                    verdict="TIE" if tie else "FAIL",
                    step=step,
                    reference_token=reference_token,
                    tyr_token=tyr_token,
                    margin=float(ref_tensors["margin"][row, step]),
                    tyr_token_gap=gap,
                )
            rows.append(entry)
        return {"verdict": worst(r["verdict"] for r in rows), "max_logit_diff": max_diff, "rows": rows}


@dataclass(frozen=True)
class TensorsClose:
    """Each named tensor in both `tensors.safetensors` files has the same shape
    and differs from the reference by at most its tolerance in `atol`."""

    atol: dict[str, float]
    name: str = "tensors"

    def __call__(self, case_dir: Path, device: str) -> dict:
        _, ref_tensors = probe.read(case_dir / "reference")
        _, tyr_tensors = probe.read(case_dir / "tyr")
        tensors = {}
        for name, atol in self.atol.items():
            ref, tyr = ref_tensors[name], tyr_tensors[name]
            if ref.shape != tyr.shape:
                raise ValueError(f"{name}: reference {ref.shape}, tyr {tyr.shape}")
            diff = float(np.abs(ref.astype(np.float64) - tyr).max())
            tensors[name] = {"verdict": "PASS" if diff <= atol else "FAIL", "max_abs_diff": diff, "atol": atol}
        return {"verdict": worst(t["verdict"] for t in tensors.values()), "tensors": tensors}


@dataclass(frozen=True)
class IntsEqual:
    """Each named integer list in both `result.json` files is the same and not
    empty; on a difference, the first differing flat index and both values."""

    keys: tuple[str, ...]
    name: str = "ints"

    def __call__(self, case_dir: Path, device: str) -> dict:
        ref, _ = probe.read(case_dir / "reference")
        tyr, _ = probe.read(case_dir / "tyr")
        fields = {}
        for key in self.keys:
            a = np.asarray(ref[key]).ravel()
            b = np.asarray(tyr[key]).ravel()
            if a.size == 0 and b.size == 0:
                raise ValueError(f"{key}: empty in the reference and in tyr")
            entry = {"verdict": "PASS", "count": int(a.size)}
            if a.shape != b.shape:
                entry.update(verdict="FAIL", reference_count=int(a.size), tyr_count=int(b.size))
            elif (differ := np.flatnonzero(a != b)).size:
                i = int(differ[0])
                entry.update(verdict="FAIL", differing=int(differ.size), index=i,
                             reference=int(a[i]), tyr=int(b[i]))
            fields[key] = entry
        return {"verdict": worst(f["verdict"] for f in fields.values()), "fields": fields}


@dataclass(frozen=True)
class ListsEqual:
    """Each named list of integer lists in both `result.json` files is the same,
    item by item; on a difference, how many items differ and, for the first, its
    index, the first differing position and both lengths."""

    keys: tuple[str, ...]
    name: str = "lists"

    def __call__(self, case_dir: Path, device: str) -> dict:
        ref, _ = probe.read(case_dir / "reference")
        tyr, _ = probe.read(case_dir / "tyr")
        fields = {}
        for key in self.keys:
            a, b = ref[key], tyr[key]
            entry = {"verdict": "PASS", "count": len(a)}
            if len(a) != len(b):
                entry.update(verdict="FAIL", reference_count=len(a), tyr_count=len(b))
            elif differ := [i for i, (x, y) in enumerate(zip(a, b)) if x != y]:
                i = differ[0]
                x, y = a[i], b[i]
                at = next((j for j, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
                entry.update(verdict="FAIL", differing=len(differ), item=i, position=at,
                             reference_length=len(x), tyr_length=len(y),
                             reference=x[at:at + 8], tyr=y[at:at + 8])
            fields[key] = entry
        return {"verdict": worst(f["verdict"] for f in fields.values()), "fields": fields}


@dataclass(frozen=True)
class IntsTiedWhereUnstable:
    """The integer list `key` in both `result.json` files is the same and not
    empty, except that off the CPU a difference where the reference's own
    `unstable` list (the same computation on the CPU) differs from its `key` is
    a tie."""

    key: str
    unstable: str
    name: str = "ints_tied"

    def __call__(self, case_dir: Path, device: str) -> dict:
        ref, _ = probe.read(case_dir / "reference")
        tyr, _ = probe.read(case_dir / "tyr")
        a = np.asarray(ref[self.key]).ravel()
        b = np.asarray(tyr[self.key]).ravel()
        if a.size == 0 and b.size == 0:
            raise ValueError(f"{self.key}: empty in the reference and in tyr")
        if a.shape != b.shape:
            return {"verdict": "FAIL", "reference_count": int(a.size), "tyr_count": int(b.size)}
        differ = np.flatnonzero(a != b)
        if differ.size == 0:
            return {"verdict": "PASS", "count": int(a.size)}
        unstable = np.zeros(a.shape, dtype=bool)
        if device != "cpu":
            unstable = a != np.asarray(ref[self.unstable]).ravel()
        failing = differ[~unstable[differ]]
        entry = {"count": int(a.size), "differing": int(differ.size), "unstable": int(unstable.sum())}
        if failing.size:
            i = int(failing[0])
            return {"verdict": "FAIL", **entry, "index": i, "reference": int(a[i]), "tyr": int(b[i])}
        return {"verdict": "TIE", **entry}
