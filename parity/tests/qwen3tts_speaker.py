import json

from lib import audio, probe
from lib.compare import TensorsClose
from lib.test import Test

TEST = Test(
    name="qwen3tts_speaker",
    weights="qwen3-tts-12hz-0.6b-base",
    probe="Qwen3TTSSpeaker",
    devices=("cpu", "cuda"),
    gpu_dtype="float32",
    cases=["qwen3tts/short", "qwen3tts/long"],
    env="qwen-tts",
)

CHECKS = [TensorsClose(atol={"embedding": 1e-3})]


def reference(case, weights, device, dtype, out):
    """The `qwen-tts` package's speaker encoder on its own mel of the case's audio,
    with the parameters of `extract_speaker_embedding`."""
    import torch
    from safetensors.torch import save_file
    from qwen_tts import Qwen3TTSModel
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

    rate, samples = audio.read(audio.case_audio(case))
    model = Qwen3TTSModel.from_pretrained(str(weights), dtype=getattr(torch, dtype), device_map=device).model
    mel = mel_spectrogram(torch.from_numpy(samples).unsqueeze(0), n_fft=1024, num_mels=128, sampling_rate=rate,
                          hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)
    with torch.inference_mode():
        embedding = model.speaker_encoder(mel.to(device, getattr(torch, dtype)))
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps({}))
    save_file({"mel": mel.float().contiguous(), "embedding": embedding.float().cpu().contiguous()},
              str(out / "tensors.safetensors"))


def tyr(case, weights, device, dtype, out):
    request = {"dtype": dtype, "mel": str(out.parent / "reference" / "tensors.safetensors")}
    probe.run(TEST.probe, request, weights, device, out)
