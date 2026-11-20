from dataclasses import dataclass


@dataclass(frozen=True)
class Test:
    """One parity test. `weights` is a key of `manifest.json`'s `weights`;
    `probe` names the executable `Parity<probe>`; `cases` are paths under
    `parity/cases/` without the `.json` suffix."""

    name: str
    weights: str
    probe: str
    devices: tuple[str, ...]
    gpu_dtype: str
    cases: list[str]
