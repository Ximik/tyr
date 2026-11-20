from lib import lm
from lib.compare import LmTokens
from lib.test import Test

TEST = Test(
    name="qwen25omni",
    weights="qwen2.5-omni-3b",
    probe="Qwen25Omni",
    devices=("cpu", "cuda"),
    gpu_dtype="float32",
    cases=["qwen25omni/short", "qwen25omni/long", "qwen25omni/batch"],
)

CHECKS = [LmTokens(margin=1e-3, logits_atol=5e-4)]


def reference(case, weights, device, dtype, out):
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        weights, dtype=getattr(torch, dtype), enable_audio_output=False)
    lm.reference(model.thinker.to(device).eval(), case, device, out)


def tyr(case, weights, device, dtype, out):
    lm.tyr(TEST.probe, case, weights, device, dtype, out)
