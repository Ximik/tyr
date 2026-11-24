import json

from lib import audio, lm, probe
from lib.compare import LmTokens, TensorsClose
from lib.test import Test

TEST = Test(
    name="whisper",
    weights="whisper-base.en",
    probe="Whisper",
    devices=("cpu", "cuda"),
    gpu_dtype="float32",
    cases=["whisper/short", "whisper/long"],
)

CHECKS = [
    TensorsClose(atol={"mel": 1e-4, "encoder": 1e-3}),
    LmTokens(margin=1e-3, logits_atol=5e-4),
]


def reference(case, weights, device, dtype, out):
    import torch
    from safetensors.torch import save_file
    from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration

    _, samples = audio.read(audio.case_audio(case))
    prompt = json.loads(case.read_text())["prompt"]
    mel = WhisperFeatureExtractor.from_pretrained(weights)(samples, sampling_rate=16000, return_tensors="pt").input_features
    model = WhisperForConditionalGeneration.from_pretrained(weights, dtype=getattr(torch, dtype)).to(device).eval()

    tokens, logits, margins = [], [], []
    with torch.inference_mode():
        encoder = model.model.encoder(mel.to(device, getattr(torch, dtype))).last_hidden_state
        output = model(encoder_outputs=(encoder,), decoder_input_ids=torch.tensor([prompt], device=device),
                       use_cache=True)
        for step in range(lm.STEPS):
            last = output.logits[:, -1, :].float()
            top2 = last.topk(2, dim=-1).values
            token = last.argmax(dim=-1)
            tokens.append(token)
            logits.append(last)
            margins.append(top2[:, 0] - top2[:, 1])
            if step + 1 == lm.STEPS:
                break
            output = model(encoder_outputs=(encoder,), decoder_input_ids=token[:, None],
                           past_key_values=output.past_key_values, use_cache=True)

    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps({"tokens": torch.stack(tokens, 1).tolist()}))
    save_file({
        "mel": mel.float().contiguous(),
        "encoder": encoder.float().cpu().contiguous(),
        "logits": torch.stack(logits, 1).cpu().contiguous(),
        "margin": torch.stack(margins, 1).cpu().contiguous(),
    }, str(out / "tensors.safetensors"))


def tyr(case, weights, device, dtype, out):
    forced = json.loads((out.parent / "reference" / "result.json").read_text())["tokens"][0]
    request = {"dtype": dtype, "audio": str(audio.case_audio(case)), "prompt": json.loads(case.read_text())["prompt"],
               "max_new_tokens": lm.STEPS, "forced": forced}
    probe.run(TEST.probe, request, weights, device, out)
