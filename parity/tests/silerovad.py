import json
from pathlib import Path

from lib import audio, probe
from lib.compare import IntsEqual, TensorsClose
from lib.test import Test

WINDOW = 512

TEST = Test(
    name="silerovad",
    weights=None,
    probe="SileroVAD",
    devices=("cpu",),
    gpu_dtype="float32",
    cases=["silerovad/short", "silerovad/long"],
)

CHECKS = [TensorsClose(atol={"probs": 1e-5}), IntsEqual(keys=("segments",))]

# The 16 kHz model's weights in the `silero-vad` package's TorchScript model, by tyr's names.
TYR_NAMES = {
    "_model.stft.forward_basis_buffer": "stft_conv.weight",
    **{f"_model.encoder.{i}.reparam_conv.{p}": f"conv{i + 1}.{p}" for i in range(4) for p in ("weight", "bias")},
    **{f"_model.decoder.rnn.{p}": f"lstm_cell.{p}" for p in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")},
    "_model.decoder.decoder.2.weight": "final_conv.weight",
    "_model.decoder.decoder.2.bias": "final_conv.bias",
}


def read_audio(case: Path):
    import torch

    return torch.from_numpy(audio.read(audio.case_audio(case))[1])


def reference(case, weights, device, dtype, out):
    """The `silero-vad` package's TorchScript model."""
    import torch
    from safetensors.torch import save_file
    from silero_vad import get_speech_timestamps, load_silero_vad

    samples = read_audio(case)
    model = load_silero_vad()
    probs = []
    with torch.inference_mode():
        for start in range(0, len(samples), WINDOW):
            chunk = torch.nn.functional.pad(samples[start:start + WINDOW], (0, max(0, start + WINDOW - len(samples))))
            probs.append(model(chunk, 16000).item())
        model.reset_states()
        segments = get_speech_timestamps(samples, model, sampling_rate=16000)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps({"segments": [[s["start"], s["end"]] for s in segments]}))
    save_file({"probs": torch.tensor(probs, dtype=torch.float32)}, str(out / "tensors.safetensors"))


def tyr(case, weights, device, dtype, out):
    """The package's weights, renamed to tyr's, are tyr's checkpoint."""
    from safetensors.torch import save_file
    from silero_vad import load_silero_vad

    state = load_silero_vad().state_dict()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "silero_vad.safetensors"
    save_file({tyr_name: state[name].float().contiguous() for name, tyr_name in TYR_NAMES.items()}, str(checkpoint))
    request = {"audio": str(audio.case_audio(case))}
    probe.run(TEST.probe, request, checkpoint, device, out)
