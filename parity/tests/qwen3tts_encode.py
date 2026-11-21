import json

from lib import audio, probe
from lib.compare import IntsTiedWhereUnstable
from lib.test import Test

TEST = Test(
    name="qwen3tts_encode",
    weights="qwen3-tts-12hz-0.6b-base",
    probe="Qwen3TTSEncode",
    devices=("cpu", "cuda"),
    gpu_dtype="float32",
    cases=["qwen3tts/short", "qwen3tts/long"],
    env="qwen-tts",
)

CHECKS = [IntsTiedWhereUnstable(key="codes", unstable="cpu_codes")]


def reference(case, weights, device, dtype, out):
    """The `qwen-tts` package's 12 Hz speech tokenizer, with the encoder's sliding-window mask."""
    import torch
    import transformers.models.mimi.modeling_mimi as mimi
    from transformers.masking_utils import create_sliding_window_causal_mask
    from qwen_tts import Qwen3TTSTokenizer

    # Mimi's sdpa/eager path drops the config's `sliding_window`; only flash attention applies it.
    mimi.create_causal_mask = create_sliding_window_causal_mask
    rate, samples = audio.read(audio.case_audio(case))
    def encode(on):
        tokenizer = Qwen3TTSTokenizer.from_pretrained(str(weights / "speech_tokenizer"), dtype=torch.float32,
                                                      device_map=on)
        return tokenizer.encode(samples, sr=rate).audio_codes[0].tolist()

    result = {"codes": encode(device)}
    if device != "cpu":
        # About 5% of the codes sit so close to a quantizer tie that the reference's own GPU
        # run picks differently from its CPU run; there a differing tyr code is a tie.
        result["cpu_codes"] = encode("cpu")
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result))


def tyr(case, weights, device, dtype, out):
    probe.run(TEST.probe, {"audio": str(audio.case_audio(case))}, weights, device, out)
