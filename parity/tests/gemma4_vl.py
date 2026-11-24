from lib import vlm
from lib.compare import LmTokens
from lib.test import Test

TEST = Test(
    name="gemma4_vl",
    weights="gemma-4-e2b-it",
    probe="Gemma4VL",
    devices=("cpu", "cuda"),
    gpu_dtype="bfloat16",
    cases=["vision/describe", "vision/wide", "vision/tall"],
    env="vision",
)

CHECKS = [LmTokens(margin=0.25, logits_atol=1.0)]


def reference(case, weights, device, dtype, out):
    vlm.reference(case, weights, device, dtype, out, chat=None)


def tyr(case, weights, device, dtype, out):
    vlm.tyr(TEST.probe, weights, device, dtype, out)
