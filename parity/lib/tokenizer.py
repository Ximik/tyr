"""The protocol of the tokenizer tests: the reference is the Hugging Face
tokenizer of the family's checkpoint.

A case file holds `{"texts": [...], "prompts": [...]}`. Each text is encoded,
with the added tokens tyr's encoder documents for the family; tyr's decoder is
given the reference's ids; each prompt is rendered as a single user turn with
the generation prompt, thinking off and, where tyr has both, on."""

import json
from pathlib import Path

from lib import probe
from lib.compare import ListsEqual

CHECKS = [ListsEqual(keys=("ids", "decoded", "chat"))]

# Whether the family's encoder adds the tokenizer's own special tokens.
ADDS_SPECIAL_TOKENS = {"qwen3": False, "qwen35": False, "qwen36": False, "gemma4": False, "laguna": True}
THINKING = {"qwen3": (False,), "qwen35": (False, True), "qwen36": (False, True), "gemma4": (False, True),
            "laguna": (False, True)}


def reference(family: str, case: Path, weights: Path, out: Path) -> None:
    from transformers import AutoTokenizer

    spec = json.loads(case.read_text())
    tokenizer = AutoTokenizer.from_pretrained(weights)
    ids = [tokenizer.encode(text, add_special_tokens=ADDS_SPECIAL_TOKENS[family]) for text in spec["texts"]]
    decoded = [list(tokenizer.decode(row, skip_special_tokens=False).encode()) for row in ids]
    chat = [list(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=thinking).encode())
            for prompt in spec["prompts"] for thinking in THINKING[family]]
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps({"ids": ids, "decoded": decoded, "chat": chat}))


def tyr(probe_name: str, family: str, case: Path, weights: Path, device: str, out: Path) -> None:
    spec = json.loads(case.read_text())
    reference_ids = json.loads((out.parent / "reference" / "result.json").read_text())["ids"]
    request = {"family": family, "texts": spec["texts"], "prompts": spec["prompts"], "reference_ids": reference_ids}
    probe.run(probe_name, request, weights, device, out)
