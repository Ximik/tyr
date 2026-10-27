#!/usr/bin/env python3
"""Exercise the native TTS CLI using generated tiny F32/BF16 checkpoints.

No downloads, Python ML packages, or audio decoder are required. This covers
prompt assembly and codec generation; the native WAV boundary has a Lean test.
"""
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def fixture(directory: Path, bf16: bool) -> None:
    base = dict(vocab_size=8, hidden_size=8, intermediate_size=16,
                num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=1, head_dim=4, num_code_groups=2)
    talker = dict(base, text_hidden_size=8, text_vocab_size=256,
                  codec_bos_id=1, codec_pad_id=0, codec_eos_token_id=7,
                  codec_think_id=2, codec_nothink_id=3,
                  codec_think_bos_id=4, codec_think_eos_id=5,
                  codec_language_id={"english": 6}, code_predictor_config=base)
    config = dict(tts_model_type="custom_voice", talker_config=talker,
                  tts_bos_token_id=1, tts_eos_token_id=2, tts_pad_token_id=0)
    (directory / "config.json").write_text(json.dumps(config))
    vocab = {chr(i): i for i in range(256)}
    vocab.update({"Ġ": 32, "Ċ": 10})
    (directory / "tokenizer.json").write_text(json.dumps({
        "model": {"vocab": vocab, "merges": [], "unk_token": "?"}}))
    (directory / "tokenizer_config.json").write_text('{"pad_token_id":0}')
    entries = {
        "talker.model.codec_embedding.weight": [8, 8],
        "talker.model.text_embedding.weight": [256, 8],
        "talker.text_projection.linear_fc1.weight": [8, 8],
        "talker.text_projection.linear_fc1.bias": [8],
        "talker.text_projection.linear_fc2.weight": [8, 8],
        "talker.text_projection.linear_fc2.bias": [8],
        "talker.model.norm.weight": [8], "talker.codec_head.weight": [8, 8],
        "talker.code_predictor.model.codec_embedding.0.weight": [8, 8],
        "talker.code_predictor.model.norm.weight": [8],
        "talker.code_predictor.lm_head.0.weight": [8, 8],
    }
    for prefix in ["talker.model.layers.0.", "talker.code_predictor.model.layers.0."]:
        for name, shape in {
            "input_layernorm.weight": [8], "post_attention_layernorm.weight": [8],
            "self_attn.q_proj.weight": [8, 8], "self_attn.k_proj.weight": [4, 8],
            "self_attn.v_proj.weight": [4, 8], "self_attn.o_proj.weight": [8, 8],
            "self_attn.q_norm.weight": [4], "self_attn.k_norm.weight": [4],
            "mlp.gate_proj.weight": [16, 8], "mlp.up_proj.weight": [16, 8],
            "mlp.down_proj.weight": [8, 16],
        }.items():
            entries[prefix + name] = shape
    header, data = {}, bytearray()
    for name, shape in entries.items():
        start = len(data)
        data.extend((b'\x80\x3f' if bf16 else struct.pack('<f', 1.0)) * math.prod(shape))
        header[name] = dict(dtype="BF16" if bf16 else "F32", shape=shape,
                            data_offsets=[start, len(data)])
    encoded = json.dumps(header).encode()
    (directory / "model.safetensors").write_bytes(struct.pack('<Q', len(encoded)) + encoded + data)
    (directory / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in entries}}))


def main() -> None:
    binary = ROOT / ".lake/build/bin/Qwen3TTSEndToEnd"
    subprocess.run([binary, "--help"], check=True, capture_output=True, timeout=60)
    cases = 0
    with tempfile.TemporaryDirectory(prefix="tyr-tts-cli-") as temporary:
        directory = Path(temporary)
        for bf16 in [False, True]:
            fixture(directory, bf16)
            for mode, language in [("auto", "Auto"), ("auto", "English"), ("think", "Auto"),
                                   ("think", "English"), ("no-think", "English")]:
                for streaming in [False, True]:
                    codes = directory / "codes.txt"
                    codes.unlink(missing_ok=True)
                    command = [str(binary), "--model-dir", str(directory), "--text", "Hello",
                               "--max-frames", "2", "--top-k", "1", "--subtalker-top-k", "1",
                               "--suppress-tail", "0", "--seed", "7", "--skip-decode",
                               "--think-mode", mode, "--language", language,
                               "--codes-path", str(codes), "--token-ids-path", str(directory / "ids.txt")]
                    if streaming:
                        command.append("--true-streaming")
                    result = subprocess.run(command, env=dict(os.environ, TYR_DEVICE="cpu"),
                                            capture_output=True, text=True, timeout=60)
                    if result.returncode:
                        raise RuntimeError(f"bf16={bf16} mode={mode} stream={streaming}\n{result.stdout}\n{result.stderr}")
                    rows = [[int(value) for value in line.split()] for line in codes.read_text().splitlines()]
                    assert len(rows) == 2 and all(len(row) == 2 for row in rows), rows
                    assert all(0 <= value < 8 for row in rows for value in row), rows
                    cases += 1
    print(f"TTS CLI: native help and {cases} generated-checkpoint codec generation cases passed")


if __name__ == "__main__":
    main()
