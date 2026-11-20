from lib import lm
from lib.compare import LmTokens
from lib.test import Test

TEST = Test(
    name="laguna",
    weights="laguna-xs-2.1-nvfp4",
    probe="Laguna",
    devices=("cuda",),
    gpu_dtype="bfloat16",
    cases=["laguna/short", "laguna/long", "laguna/batch"],
)

CHECKS = [LmTokens(margin=0.25, logits_atol=1.0)]


def reference(case, weights, device, dtype, out):
    """The model code of the checkpoint's own repository, at the manifest's revision."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        weights, dtype=getattr(torch, dtype), trust_remote_code=True, device_map=device).eval()
    lm.reference(model, case, device, out)


def tyr(case, weights, device, dtype, out):
    lm.tyr(TEST.probe, case, weights, device, dtype, out)
