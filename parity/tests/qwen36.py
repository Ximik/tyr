from lib import lm
from lib.compare import LmTokens
from lib.test import Test

TEST = Test(
    name="qwen36",
    weights="qwen3.6-35b-a3b",
    probe="Qwen35",
    devices=("cuda",),
    gpu_dtype="bfloat16",
    cases=["qwen35/short", "qwen35/long", "qwen35/batch"],
)

CHECKS = [LmTokens(margin=0.25, logits_atol=2.6)]


def reference(case, weights, device, dtype, out):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(weights, dtype=getattr(torch, dtype)).to(device).eval()
    lm.reference(model, case, device, out)


def tyr(case, weights, device, dtype, out):
    lm.tyr(TEST.probe, case, weights, device, dtype, out)
