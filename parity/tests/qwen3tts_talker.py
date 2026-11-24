import json

from lib import audio, probe
from lib.compare import IntsEqual
from lib.test import Test

MAX_FRAMES = 512
REPETITION_PENALTY = 1.05

TEST = Test(
    name="qwen3tts_talker",
    weights="qwen3-tts-12hz-0.6b-base",
    probe="Qwen3TTSTalker",
    devices=("cpu", "cuda"),
    gpu_dtype="float32",
    cases=["qwen3tts_talker/short", "qwen3tts_talker/long"],
    env="qwen-tts",
)

CHECKS = [IntsEqual(keys=("codes",))]


def reference(case, weights, device, dtype, out):
    """The `qwen-tts` package's voice clone from the case's audio (speaker vector
    only), greedy in both talkers, with the package's default repetition
    penalty. The talker's inputs and the codes it returns are taken from inside
    its `generate`."""
    import torch
    from safetensors.torch import save_file
    from qwen_tts import Qwen3TTSModel

    spec = json.loads(case.read_text())
    rate, samples = audio.read(audio.case_audio(case))
    tts = Qwen3TTSModel.from_pretrained(str(weights), dtype=getattr(torch, dtype), device_map=device)
    core = tts.model
    captured = {}

    talker_generate = core.talker.generate

    def capture_inputs(**kwargs):
        for name in ("inputs_embeds", "trailing_text_hidden", "tts_pad_embed"):
            captured[name] = kwargs[name].float().cpu().contiguous()
        return talker_generate(**kwargs)

    generate = core.generate

    def capture_codes(*args, **kwargs):
        result = generate(*args, **kwargs)
        captured["codes"] = result[0][0].tolist()
        return result

    core.talker.generate = capture_inputs
    core.generate = capture_codes
    tts.generate_voice_clone(text=spec["text"], language=spec["language"], ref_audio=(samples, rate),
                             x_vector_only_mode=True, do_sample=False, subtalker_dosample=False,
                             repetition_penalty=REPETITION_PENALTY, max_new_tokens=MAX_FRAMES)

    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps({"codes": captured.pop("codes")}))
    save_file(captured, str(out / "tensors.safetensors"))


def tyr(case, weights, device, dtype, out):
    # The reference fills a frame's residual codes one step after sampling it, so its last frame is dropped.
    request = {"dtype": dtype, "tensors": str(out.parent / "reference" / "tensors.safetensors"),
               "max_frames": MAX_FRAMES - 1, "repetition_penalty": REPETITION_PENALTY}
    probe.run(TEST.probe, request, weights, device, out)
