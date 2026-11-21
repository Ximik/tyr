"""A 256x256 image in the pipeline's default number of steps from seeded noise:
the denoised latents and the VAE decoder's output."""

import json

from lib import flux, probe
from lib.compare import TensorsClose
from lib.test import Test

TEST = Test(
    name="flux_sample",
    weights="flux2-klein-4b",
    probe="FluxSample",
    devices=("cpu", "cuda"),
    gpu_dtype="float32",
    cases=["flux/short", "flux/long"],
    env="diffusers",
)

SEED = 0
SIDE = 256
STEPS = 4

CHECKS = [TensorsClose(atol={"latents": 2e-3, "image": 2e-3})]


def reference(case, weights, device, dtype, out):
    """`Flux2KleinPipeline.__call__` on the prompt embedding of `lib.flux`, the
    text encoder released before the transformer loads."""
    import torch
    from diffusers import Flux2KleinPipeline
    from safetensors.torch import save_file

    dtype = getattr(torch, dtype)
    prompt = json.loads(case.read_text())["prompt"]
    embeds, ids, mask = flux.prompt_embeds(weights, prompt, device, dtype)
    pipe = Flux2KleinPipeline.from_pretrained(
        weights, transformer=flux.transformer(weights, device, dtype), text_encoder=None, tokenizer=None,
        torch_dtype=dtype).to(device)
    noise = torch.randn(1, 128, SIDE // 16, SIDE // 16, generator=torch.Generator().manual_seed(SEED))
    captured = {}
    decode = pipe.vae.decode

    def capture_image(latents, *args, **kwargs):
        result = decode(latents, *args, **kwargs)
        captured["image"] = result[0]
        return result

    def capture_latents(pipe, step, timestep, tensors):
        captured["latents"] = tensors["latents"]
        return tensors

    pipe.vae.decode = capture_image
    with torch.inference_mode():
        pipe(prompt_embeds=embeds, height=SIDE, width=SIDE, num_inference_steps=STEPS, latents=noise.to(device),
             output_type="pt", callback_on_step_end=capture_latents, callback_on_step_end_tensor_inputs=["latents"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps({"input_ids": ids, "attention_mask": mask}))
    save_file({"noise": Flux2KleinPipeline._pack_latents(noise).contiguous(),
               "latents": captured["latents"].float().cpu().contiguous(),
               "image": captured["image"].float().cpu().contiguous()}, str(out / "tensors.safetensors"))


def tyr(case, weights, device, dtype, out):
    reference_out = out.parent / "reference"
    fed = json.loads((reference_out / "result.json").read_text())
    request = {"dtype": dtype, "steps": STEPS, "noise": str(reference_out / "tensors.safetensors"), **fed}
    probe.run(TEST.probe, request, weights, device, out)
