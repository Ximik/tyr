"""Reading the cases' PCM16 WAV files."""

import json
import wave
from pathlib import Path

CASES = Path(__file__).resolve().parents[1] / "cases"


def case_audio(case: Path) -> Path:
    """The WAV file a case names under `"audio"`."""
    return CASES / json.loads(case.read_text())["audio"]


def read(path: Path):
    """Sampling rate and mono float32 samples, each 16-bit value divided by 32768."""
    import numpy as np

    with wave.open(str(path)) as f:
        if f.getnchannels() != 1 or f.getsampwidth() != 2:
            raise ValueError(f"{path}: expected mono 16-bit PCM")
        samples = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        return f.getframerate(), samples
