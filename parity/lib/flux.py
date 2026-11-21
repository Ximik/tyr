"""FLUX.2-klein references built from diffusers' `Flux2KleinPipeline`. The
transformer is read from the single-file checkpoint tyr reads."""

from pathlib import Path

TRANSFORMER = "flux-2-klein-4b.safetensors"


def prompt_embeds(weights: Path, prompt: str, device: str, dtype):
    """The pipeline's prompt embedding and the `input_ids` and `attention_mask`
    it feeds the text encoder."""
    import torch
    from diffusers import Flux2KleinPipeline
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(weights / "tokenizer")
    encoder = Qwen3ForCausalLM.from_pretrained(weights / "text_encoder", dtype=dtype, device_map=device).eval()
    fed = {}

    def capture(module, args, kwargs):
        fed["input_ids"], fed["attention_mask"] = kwargs["input_ids"], kwargs["attention_mask"]

    encoder.register_forward_pre_hook(capture, with_kwargs=True)
    with torch.inference_mode():
        embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(encoder, tokenizer, prompt, dtype=dtype, device=device)
    return embeds, fed["input_ids"][0].tolist(), fed["attention_mask"][0].tolist()


def transformer(weights: Path, device: str, dtype):
    from diffusers import Flux2Transformer2DModel

    return Flux2Transformer2DModel.from_single_file(
        str(weights / TRANSFORMER), config=str(weights), subfolder="transformer", torch_dtype=dtype).to(device).eval()
