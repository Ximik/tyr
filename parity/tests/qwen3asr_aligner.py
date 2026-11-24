import json

from lib import audio, probe
from lib.compare import IntsEqual, ListsEqual, TensorsClose
from lib.test import Test

TEST = Test(
    name="qwen3asr_aligner",
    weights="qwen3-forcedaligner-0.6b",
    reference_weights="qwen3-forcedaligner-0.6b-hf",
    probe="Qwen3ASRAligner",
    devices=("cpu", "cuda"),
    gpu_dtype="float32",
    cases=["qwen3asr_aligner/short", "qwen3asr_aligner/long"],
)

CHECKS = [
    ListsEqual(keys=("words",)),
    IntsEqual(keys=("prompt", "classes", "timestamps")),
    TensorsClose(atol={"mel": 1e-4, "logits": 1e-3}),
]


def reference(case, weights, device, dtype, out):
    """`transformers`' Qwen3-ASR token classifier on the converted `-hf` checkpoint."""
    import torch
    from safetensors.torch import save_file
    from transformers import AutoProcessor, Qwen3ASRForTokenClassification

    spec = json.loads(case.read_text())
    _, samples = audio.read(audio.case_audio(case))
    processor = AutoProcessor.from_pretrained(weights)
    inputs, word_lists = processor.prepare_forced_aligner_inputs(
        samples, spec["transcript"], language=spec["language"], return_tensors="pt")
    torch_dtype = getattr(torch, dtype)
    model = Qwen3ASRForTokenClassification.from_pretrained(weights, dtype=torch_dtype).to(device).eval()
    marker = model.config.timestamp_token_id
    with torch.inference_mode():
        logits = model(input_ids=inputs["input_ids"].to(device),
                       input_features=inputs["input_features"].to(device, torch_dtype),
                       input_features_mask=inputs["input_features_mask"].to(device)).logits.float().cpu()
    prompt = inputs["input_ids"][0]
    classes = logits[0].argmax(-1)[prompt == marker].tolist()
    words = processor.decode_forced_alignment(logits, inputs["input_ids"], word_lists, marker)[0]
    timestamps = [[round(w["start_time"] * 1000), round(w["end_time"] * 1000)] for w in words]

    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps({
        "prompt": prompt.tolist(), "words": word_lists[0], "classes": [classes], "timestamps": [timestamps]}))
    save_file({
        "mel": inputs["input_features"][..., :int(inputs["input_features_mask"].sum())].float().contiguous(),
        "logits": logits.contiguous(),
    }, str(out / "tensors.safetensors"))


def tyr(case, weights, device, dtype, out):
    spec = json.loads(case.read_text())
    request = {"dtype": dtype, "audio": str(audio.case_audio(case)), "transcript": spec["transcript"],
               "language": spec["language"]}
    probe.run(TEST.probe, request, weights, device, out)
