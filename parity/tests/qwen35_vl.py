from lib import vlm
from lib.compare import LmTokens
from lib.test import Test

TEST = Test(
    name="qwen35_vl",
    weights="qwen3.5-0.8b",
    probe="Qwen35VL",
    devices=("cpu", "cuda"),
    gpu_dtype="float32",
    cases=["vision/describe", "vision/wide", "vision/tall"],
    env="vision",
)

CHECKS = [LmTokens(margin=1e-3, logits_atol=2e-3)]


def reference(case, weights, device, dtype, out):
    vlm.reference(case, weights, device, dtype, out, chat={"enable_thinking": False})


def tyr(case, weights, device, dtype, out):
    vlm.tyr(TEST.probe, weights, device, dtype, out)
