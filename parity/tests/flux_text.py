import json

from lib import flux, probe
from lib.compare import IntsEqual, TensorsClose
from lib.test import Test

TEST = Test(
    name="flux_text",
    weights="flux2-klein-4b",
    probe="FluxText",
    devices=("cpu", "cuda"),
    gpu_dtype="float32",
    cases=["flux/short", "flux/long"],
    env="diffusers",
)

CHECKS = [IntsEqual(keys=("input_ids", "attention_mask")), TensorsClose(atol={"embeddings": 1e-2})]


def reference(case, weights, device, dtype, out):
    import torch
    from safetensors.torch import save_file

    prompt = json.loads(case.read_text())["prompt"]
    embeds, ids, mask = flux.prompt_embeds(weights, prompt, device, getattr(torch, dtype))
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps({"input_ids": ids, "attention_mask": mask}))
    save_file({"embeddings": embeds.float().cpu().contiguous()}, str(out / "tensors.safetensors"))


def tyr(case, weights, device, dtype, out):
    """The prompt tokenized by tyr; the embedding from the reference's ids."""
    fed = json.loads((out.parent / "reference" / "result.json").read_text())
    request = {"prompt": json.loads(case.read_text())["prompt"], **fed}
    probe.run(TEST.probe, request, weights, device, out)
