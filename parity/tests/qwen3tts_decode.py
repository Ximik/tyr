import json

from lib import audio, probe
from lib.compare import TensorsClose
from lib.test import Test

TEST = Test(
    name="qwen3tts_decode",
    weights="qwen3-tts-12hz-0.6b-base",
    probe="Qwen3TTSDecode",
    devices=("cpu", "cuda"),
    gpu_dtype="float32",
    cases=["qwen3tts/short", "qwen3tts/long"],
    env="qwen-tts",
)

CHECKS = [TensorsClose(atol={"wave": 1e-3})]


def reference(case, weights, device, dtype, out):
    """The `qwen-tts` package's 12 Hz speech tokenizer: the case's audio encoded,
    and those codes decoded."""
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from qwen_tts import Qwen3TTSTokenizer

    rate, samples = audio.read(audio.case_audio(case))
    tokenizer = Qwen3TTSTokenizer.from_pretrained(str(weights / "speech_tokenizer"), dtype=torch.float32,
                                                  device_map=device)
    codes = tokenizer.encode(samples, sr=rate).audio_codes[0]
    waves, _ = tokenizer.decode({"audio_codes": [codes]})
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps({"codes": codes.tolist()}))
    save_file({"wave": torch.from_numpy(np.asarray(waves[0], dtype=np.float32))}, str(out / "tensors.safetensors"))


def tyr(case, weights, device, dtype, out):
    codes = json.loads((out.parent / "reference" / "result.json").read_text())["codes"]
    probe.run(TEST.probe, {"codes": codes}, weights, device, out)
