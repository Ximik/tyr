import json

from lib import audio, lm, probe
from lib.compare import IntsEqual, LmTokens, TensorsClose
from lib.test import Test

TEST = Test(
    name="qwen3asr",
    weights="qwen3-asr-0.6b",
    reference_weights="qwen3-asr-0.6b-hf",
    probe="Qwen3ASR",
    devices=("cpu", "cuda"),
    gpu_dtype="float32",
    cases=["qwen3asr/short", "qwen3asr/long"],
)

CHECKS = [
    IntsEqual(keys=("prompt",)),
    TensorsClose(atol={"mel": 1e-4, "encoder": 1e-3}),
    LmTokens(margin=1e-3, logits_atol=1e-3),
]


def reference(case, weights, device, dtype, out):
    """`transformers`' Qwen3-ASR on the converted `-hf` checkpoint; the prompt is
    its transcription request with the language forced."""
    import torch
    from safetensors.torch import save_file
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    _, samples = audio.read(audio.case_audio(case))
    language = json.loads(case.read_text())["language"]
    processor = AutoProcessor.from_pretrained(weights)
    inputs = processor.apply_transcription_request(audio=samples, language=language, return_tensors="pt")
    torch_dtype = getattr(torch, dtype)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(weights, dtype=torch_dtype).to(device).eval()
    features = inputs["input_features"].to(device, torch_dtype)
    features_mask = inputs["input_features_mask"].to(device)
    prompt = inputs["input_ids"].to(device)

    tokens, logits, margins = [], [], []
    with torch.inference_mode():
        encoder = model.get_audio_features(features, features_mask).pooler_output
        output = model(input_ids=prompt, input_features=features, input_features_mask=features_mask, use_cache=True)
        for step in range(lm.STEPS):
            last = output.logits[:, -1, :].float()
            top2 = last.topk(2, dim=-1).values
            token = last.argmax(dim=-1)
            tokens.append(token)
            logits.append(last)
            margins.append(top2[:, 0] - top2[:, 1])
            if step + 1 == lm.STEPS:
                break
            output = model(input_ids=token[:, None], past_key_values=output.past_key_values, use_cache=True)

    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(
        {"tokens": torch.stack(tokens, 1).tolist(), "prompt": prompt[0].tolist()}))
    save_file({
        "mel": inputs["input_features"][..., :int(inputs["input_features_mask"].sum())].float().contiguous(),
        "encoder": encoder.float().reshape(-1, encoder.shape[-1]).cpu().contiguous(),
        "logits": torch.stack(logits, 1).cpu().contiguous(),
        "margin": torch.stack(margins, 1).cpu().contiguous(),
    }, str(out / "tensors.safetensors"))


def tyr(case, weights, device, dtype, out):
    forced = json.loads((out.parent / "reference" / "result.json").read_text())["tokens"][0]
    request = {"dtype": dtype, "audio": str(audio.case_audio(case)), "language": json.loads(case.read_text())["language"],
               "max_new_tokens": lm.STEPS, "forced": forced}
    probe.run(TEST.probe, request, weights, device, out)
