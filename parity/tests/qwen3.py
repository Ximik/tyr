from lib import lm
from lib.compare import LmTokens
from lib.test import Test

TEST = Test(
    name="qwen3",
    weights="qwen3-0.6b",
    probe="Qwen3",
    devices=("cpu", "cuda"),
    gpu_dtype="float32",
    cases=["qwen3/short", "qwen3/long", "qwen3/batch"],
)

CHECKS = [LmTokens(margin=1e-3, logits_atol=5e-4)]


def reference(case, weights, device, dtype, out):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(weights, dtype=getattr(torch, dtype)).to(device).eval()
    lm.reference(model, case, device, out)


def tyr(case, weights, device, dtype, out):
    lm.tyr(TEST.probe, case, weights, device, dtype, out)
