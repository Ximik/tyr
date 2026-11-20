from dataclasses import dataclass


@dataclass(frozen=True)
class Test:
    """One parity test. `weights` is a key of `manifest.json`'s `weights`, read by
    tyr and, unless `reference_weights` names another key, by the reference; a
    test without weights in the manifest gets an empty weights path;
    `probe` names the executable `Parity<probe>`; `cases` are paths under
    `parity/cases/` without the `.json` suffix; `env` names the environment
    `parity/envs/<env>.txt` the test's stages run in."""

    name: str
    weights: str | None
    probe: str
    devices: tuple[str, ...]
    gpu_dtype: str
    cases: list[str]
    reference_weights: str | None = None
    env: str = "transformers"

    def weights_for(self, stage: str) -> str | None:
        return self.reference_weights or self.weights if stage == "reference" else self.weights
