"""The protocol for models that read an image and emit tokens, as in `lib.lm`:
the reference decodes greedily for `lm.STEPS` tokens and keeps each step's
logits and top-1/top-2 gap; tyr decodes greedily from the same inputs, then is
fed the reference's tokens.

A case file holds `{"image": <path under parity/cases>, "prompt"}`, one user
turn of the image followed by the prompt. The inputs are the model's own
processor output: `input_ids` and the processor's integer tensors go to
`result.json` under `inputs`, its real-valued tensors (`pixel_values`) to the
tensor file."""

import json
from pathlib import Path

from lib import lm, probe

CASES = Path(__file__).resolve().parents[1] / "cases"


def reference(case: Path, weights: Path, device: str, dtype: str, out: Path, chat: dict | None = None) -> None:
    import torch
    from PIL import Image
    from safetensors.torch import save_file
    from transformers import AutoModelForImageTextToText, AutoProcessor

    spec = json.loads(case.read_text())
    processor = AutoProcessor.from_pretrained(weights)
    model = AutoModelForImageTextToText.from_pretrained(weights, dtype=getattr(torch, dtype), device_map=device).eval()
    image = Image.open(CASES / spec["image"]).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": spec["prompt"]}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                           return_tensors="pt", **(chat or {}))
    saved_inputs = {k: v.tolist() for k, v in inputs.items() if not v.is_floating_point()}
    floats = {k: v.float().contiguous() for k, v in inputs.items() if v.is_floating_point()}
    inputs = {k: (v.to(device, getattr(torch, dtype)) if v.is_floating_point() else v.to(device)) for k, v in inputs.items()}

    # The model's own `generate` keeps its multimodal positions; without an EOS
    # it runs the full `lm.STEPS`, and `output_logits` are the unprocessed logits.
    model.generation_config.eos_token_id = None
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=lm.STEPS, do_sample=False,
                                   output_logits=True, return_dict_in_generate=True)
    logits = [step.float() for step in generated.logits]
    tokens = generated.sequences[:, inputs["input_ids"].shape[1]:]
    margins = [step.topk(2, dim=-1).values for step in logits]
    margins = [top2[:, 0] - top2[:, 1] for top2 in margins]

    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps({"tokens": tokens.tolist(), "inputs": saved_inputs}))
    save_file({"logits": torch.stack(logits, 1).cpu().contiguous(),
               "margin": torch.stack(margins, 1).cpu().contiguous(), **floats},
              str(out / "tensors.safetensors"))


def tyr(probe_name: str, weights: Path, device: str, dtype: str, out: Path) -> None:
    """The probe gets the reference's processor inputs and tokens."""
    reference_out = out.parent / "reference"
    result = json.loads((reference_out / "result.json").read_text())
    request = {"dtype": dtype, "prompts": result["inputs"]["input_ids"], "inputs": result["inputs"],
               "tensors": str(reference_out / "tensors.safetensors"), "max_new_tokens": lm.STEPS,
               "forced": result["tokens"]}
    probe.run(probe_name, request, weights, device, out)
