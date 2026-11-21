from lib import tokenizer
from lib.test import Test

FAMILY = "gemma4"
TEST = Test(
    name="tokenizer_gemma4",
    weights="gemma-4-e2b-it-tokenizer",
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
