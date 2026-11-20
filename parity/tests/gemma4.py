from lib import lm
from lib.compare import LmTokens
from lib.test import Test

TEST = Test(
    name="gemma4",
    weights="gemma-4-e2b-it",
    probe="Gemma4",
    devices=("cpu", "cuda"),
    gpu_dtype="bfloat16",
    cases=["gemma4/short", "gemma4/long", "gemma4/batch"],
)

CHECKS = [LmTokens(margin=0.5, logits_atol=1.0)]


def reference(case, weights, device, dtype, out):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(weights, dtype=getattr(torch, dtype)).to(device).eval()
    lm.reference(model, case, device, out)


def tyr(case, weights, device, dtype, out):
    lm.tyr(TEST.probe, case, weights, device, dtype, out)
