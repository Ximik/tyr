from lib import tokenizer
from lib.test import Test

FAMILY = "qwen35"
TEST = Test(
    name="tokenizer_qwen35",
    weights="qwen3.5-0.8b-tokenizer",
    probe="Tokenizer",
    devices=("cpu",),
    gpu_dtype="float32",
    cases=["tokenizer/basic", "tokenizer/fuzz"],
)

CHECKS = tokenizer.CHECKS


def reference(case, weights, device, dtype, out):
    tokenizer.reference(FAMILY, case, weights, out)


def tyr(case, weights, device, dtype, out):
    tokenizer.tyr(TEST.probe, FAMILY, case, weights, device, out)
