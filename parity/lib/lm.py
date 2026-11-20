"""The protocol for models that emit tokens: the reference decodes greedily and
keeps every step's logits and top-1/top-2 gap; tyr decodes greedily with its
KV cache, then is fed the reference's tokens to give logits at the same steps.

A case file holds `{"prompts": [[token, ...], ...]}`."""

import json
from pathlib import Path

from lib import probe

STEPS = 64


def prompts(case: Path) -> list[list[int]]:
    return json.loads(case.read_text())["prompts"]


def reference(model, case: Path, device: str, out: Path) -> None:
    """Greedy decoding of `model` (a Hugging Face causal LM) over the case's
    prompts, left-padded into one batch, for `STEPS` tokens without stopping."""
    import inspect

    import torch
    from safetensors.torch import save_file

    rows = prompts(case)
    width = max(len(p) for p in rows)
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    mask = torch.zeros(len(rows), width, dtype=torch.long)
    for i, p in enumerate(rows):
        ids[i, width - len(p):] = torch.tensor(p)
        mask[i, width - len(p):] = 1
    ids, mask = ids.to(device), mask.to(device)
    positions = (mask.cumsum(-1) - 1).clamp(min=0)

    tokens, logits, margins = [], [], []
    with torch.inference_mode():
        last_only = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(model.forward).parameters else {}
        output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=True, **last_only)
        for step in range(STEPS):
            last = output.logits[:, -1, :].float()
            top2 = last.topk(2, dim=-1).values
            token = last.argmax(dim=-1)
            tokens.append(token)
            logits.append(last)
            margins.append(top2[:, 0] - top2[:, 1])
            if step + 1 == STEPS:
                break
            mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
            positions = positions[:, -1:] + 1
            output = model(
                input_ids=token[:, None], attention_mask=mask, position_ids=positions,
                past_key_values=output.past_key_values, use_cache=True,
            )

    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps({"tokens": torch.stack(tokens, 1).tolist()}))
    save_file(
        {"logits": torch.stack(logits, 1).cpu().contiguous(),
         "margin": torch.stack(margins, 1).cpu().contiguous()},
        str(out / "tensors.safetensors"),
    )


def tyr(probe_name: str, case: Path, weights: Path, device: str, dtype: str, out: Path) -> None:
    """Run the probe with the case's prompts and the reference's tokens, which
    `run.py` has written to `out.parent / "reference"`."""
    forced = json.loads((out.parent / "reference" / "result.json").read_text())["tokens"]
    request = {"dtype": dtype, "prompts": prompts(case), "max_new_tokens": STEPS, "forced": forced}
    probe.run(probe_name, request, weights, device, out)
