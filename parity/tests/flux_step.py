"""One transformer prediction on seeded random latents and text features. A case
gives the seed, the packed latent grid, the text length and the timestep in
`[0, 1]`."""

import json

from lib import flux, probe
from lib.compare import TensorsClose
from lib.test import Test

TEST = Test(
    name="flux_step",
    weights="flux2-klein-4b",
    probe="FluxStep",
    devices=("cpu", "cuda"),
    gpu_dtype="float32",
    cases=["flux_step/early", "flux_step/late"],
    env="diffusers",
)

CHECKS = [TensorsClose(atol={"velocity": 1e-4})]


def reference(case, weights, device, dtype, out):
    import torch
    from diffusers import Flux2KleinPipeline
    from safetensors.torch import save_file

    spec = json.loads(case.read_text())
    generator = torch.Generator().manual_seed(spec["seed"])
    grid = torch.randn(1, 128, spec["height"], spec["width"], generator=generator)
    text = torch.randn(1, spec["tokens"], 7680, generator=generator)
    timestep = torch.tensor([spec["timestep"]])
    model = flux.transformer(weights, device, getattr(torch, dtype))
    latents = Flux2KleinPipeline._pack_latents(grid)
    with torch.inference_mode():
        velocity = model(
            hidden_states=latents.to(device, model.dtype), encoder_hidden_states=text.to(device, model.dtype),
            timestep=timestep.to(device, model.dtype), guidance=None,
            img_ids=Flux2KleinPipeline._prepare_latent_ids(grid).to(device),
            txt_ids=Flux2KleinPipeline._prepare_text_ids(text).to(device), return_dict=False)[0]
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps({}))
    save_file({"latents": latents.contiguous(), "text": text, "timestep": timestep,
               "velocity": velocity.float().cpu().contiguous()}, str(out / "tensors.safetensors"))


def tyr(case, weights, device, dtype, out):
    spec = json.loads(case.read_text())
    request = {"dtype": dtype, "height": spec["height"], "width": spec["width"],
               "inputs": str(out.parent / "reference" / "tensors.safetensors")}
    probe.run(TEST.probe, request, weights, device, out)
