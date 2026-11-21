"""Kokoro-82M from phonemes. A case's phonemes are its text's words looked up in
misaki 0.9.4's US dictionaries."""

import json
import sys
from pathlib import Path

from lib import probe
from lib.compare import IntsEqual, TensorsClose
from lib.test import Test

TEST = Test(
    name="kittentts",
    weights="kokoro-82m",
    probe="KittenTTS",
    devices=("cpu", "cuda"),
    gpu_dtype="float32",
    cases=["kittentts/short", "kittentts/long"],
    env="kokoro",
)

SEED = 0
SPEED = 1.0
VOICE = "voices/af_heart.pt"

CHECKS = [IntsEqual(keys=("durations",)), TensorsClose(atol={
    "f0": 1e-2,
    "n": 1e-3,
    # Not reproducible yet: the vocoder's noise comes from `torch.rand`/`randn_like`
    # on a non-contiguous view, which tyr's contiguous CPU draws do not reproduce.
    # "wave": 1e-3,
})]


def reference(case, weights, device, dtype, out):
    """The `kokoro` package's `KModel.forward_with_tokens`, with the voice row
    `KPipeline` picks; F0 and N are taken from inside it."""
    import torch
    from safetensors.torch import save_file
    from kokoro.model import KModel

    phonemes = json.loads(case.read_text())["phonemes"]
    model = KModel(repo_id="hexgrad/Kokoro-82M", config=str(weights / "config.json"),
                   model=str(weights / "kokoro-v1_0.pth")).to(device).eval()
    ids = [0, *(model.vocab[p] for p in phonemes if p in model.vocab), 0]
    style = torch.load(weights / VOICE, weights_only=True)[len(phonemes) - 1].to(device)
    captured = {}
    f0n = model.predictor.F0Ntrain

    def capture(x, s):
        captured["f0"], captured["n"] = f0n(x, s)
        return captured["f0"], captured["n"]

    model.predictor.F0Ntrain = capture
    torch.manual_seed(SEED)
    with torch.no_grad():
        wave, durations = model.forward_with_tokens(torch.tensor([ids], device=device), style, SPEED)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps({"durations": durations.tolist()}))
    save_file({name: t.reshape(1, 1, -1).float().cpu().contiguous()
               for name, t in (("f0", captured["f0"]), ("n", captured["n"]), ("wave", wave))},
              str(out / "tensors.safetensors"))


def tyr(case, weights, device, dtype, out):
    """The checkpoint and voice converted to the safetensors tyr reads."""
    import shutil

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from kokoro_to_safetensors import convert_checkpoint, convert_voice

    model = out / "model"
    model.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(weights / "config.json", model / "config.json")
    convert_checkpoint(weights / "kokoro-v1_0.pth", model / "model.safetensors")
    convert_voice(weights / VOICE, model / "voice.safetensors")
    phonemes = json.loads(case.read_text())["phonemes"]
    request = {"phonemes": phonemes, "voice": str((model / "voice.safetensors").resolve()), "speed": SPEED,
               "seed": SEED}
    probe.run(TEST.probe, request, model, device, out)
