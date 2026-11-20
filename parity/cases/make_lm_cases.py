"""Write the token cases `short`, `long` and `batch` for a model, tokenized with
its Hugging Face tokenizer and chat template (thinking off), from the messages
of the `qwen3` cases:

    python parity/cases/make_lm_cases.py <tokenizer dir> <case dir name>

`long` is the longest prefix, in words, of the qwen3 `long` message that stays
within 1500 tokens. `batch` pairs the `short` dialogue with the longest prefix
of `QUESTION` that gives exactly as many tokens."""

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

CASES = Path(__file__).resolve().parent
LONG_TOKENS = 1500
QUESTION = ("Explain in plain words how a compiler turns source code into machine code, naming the main stages "
            "such as parsing, type checking, optimization and code generation, and give one short example of an "
            "optimization that a compiler might apply to a simple loop written in a language like C or Rust")


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
    out = CASES / sys.argv[2]

    def ids(messages):
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                             tokenize=True, return_dict=False)

    def longest_prefix(words, fits):
        """Messages from the longest prefix of `words` for which `fits` holds."""
        best = None
        for n in range(1, len(words) + 1):
            messages = [{"role": "user", "content": " ".join(words[:n])}]
            if fits(len(ids(messages))):
                best = messages
        if best is None:
            sys.exit("no prefix fits")
        return best

    source = {name: json.loads((CASES / "qwen3" / f"{name}.json").read_text())["messages"]
              for name in ("short", "long")}
    dialogue = source["short"][0]
    long = longest_prefix(source["long"][0][0]["content"].split(" "), lambda n: n <= LONG_TOKENS)
    second = longest_prefix(QUESTION.split(" "), lambda n: n == len(ids(dialogue)))
    out.mkdir(exist_ok=True)
    for name, conversations in {"short": [dialogue], "long": [long], "batch": [dialogue, second]}.items():
        prompts = [ids(c) for c in conversations]
        print(name, [len(p) for p in prompts])
        (out / f"{name}.json").write_text(
            json.dumps({"messages": conversations, "prompts": prompts}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
