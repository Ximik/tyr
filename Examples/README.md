# Examples

Runnable examples demonstrating Tyr's capabilities. Each example is a self-contained training or inference script.

Before running any example, set up the environment (see [Environment Setup](../README.md#environment-setup)) or use the Lake helper scripts where available.

## TrainGPT

Character-level GPT training on Shakespeare, matching [nanoGPT](https://github.com/karpathy/nanoGPT) configuration.

**Data:** `data/shakespeare_char/{train,val}.bin` (nanoGPT format). Falls back to random tokens if data is missing.

```bash
lake build TrainGPT
lake run train
# or manually: .lake/build/bin/TrainGPT
```

Configs available in `Examples/GPT/GPT.lean`: `nanogpt_cpu_shakespeare`, `gpt2_micro`, `gpt2_mini`, `gpt2_small`.

## TrainDiffusion

Discrete masked diffusion model on ASCII text. Supports animated terminal output during sampling, multi-block generation with overlap for continuity, and checkpoint save/load.

**Data:** `data/shakespeare_char/input.txt` (plain text, optional -- falls back to random tokens).

```bash
lake build TrainDiffusion
.lake/build/bin/TrainDiffusion                          # Train from scratch
.lake/build/bin/TrainDiffusion --generate               # Generate from checkpoint
.lake/build/bin/TrainDiffusion -g -p "ROMEO:" -t 0.7    # Generate with prompt
.lake/build/bin/TrainDiffusion -g -n 8                   # Generate 8 blocks
```

**CLI flags:**
- `--generate, -g [path]` -- Load checkpoint and generate (skip training)
- `--checkpoint, -c <path>` -- Checkpoint directory (default: `checkpoints/diffusion`)
- `--prompt, -p <text>` -- Prompt for generation
- `--blocks, -n <num>` -- Number of blocks to generate (default: 4)
- `--temperature, -t <val>` -- Sampling temperature (default: 0.9)

## TrainNanoChat

Modded-nanogpt style distributed GPT training with NorMuon + DistAdam optimizers, dynamic batch size / window size schedules, and multi-GPU support via `torchrun`.

**Data:** `data/fineweb10B/` (sharded binary), `data/fineweb_val/` (validation shards).

```bash
lake build TrainNanoChat

# Single GPU
.lake/build/bin/TrainNanoChat

# Multi-GPU
torchrun --nproc_per_node=8 .lake/build/bin/TrainNanoChat

# With options
.lake/build/bin/TrainNanoChat --data data/fineweb10B --val data/fineweb_val \
  --checkpoint-dir checkpoints/modded --debug
```

**CLI flags:**
- `--data <path>` -- Training data directory (default: `data/fineweb10B`)
- `--val <path>` -- Validation data directory (default: `data/fineweb_val`)
- `--checkpoint-dir <path>` -- Checkpoint save directory
- `--resume <path>` -- Resume from checkpoint
- `--debug` -- Run with small model for testing

## FluxDemo

End-to-end Flux Klein 4B image generation: text encoding (Qwen) -> diffusion -> VAE decoding.

**Weights:** Download Flux Klein 4B weights and place them at:
- `weights/flux.safetensors` -- Flux transformer weights
- `weights/ae.safetensors` -- VAE decoder weights
- `weights/flux-klein-4b/text_encoder/` -- Qwen text encoder (sharded)

```bash
lake build FluxDemo
.lake/build/bin/FluxDemo
```

**Output:** `output.ppm`

## Qwen3TTSEndToEnd

End-to-end Qwen3-TTS demo: Lean loads real talker/speaker weights, does Lean tokenization + generation, and uses thin Python bridge scripts only for speech-tokenizer/audio frontend tasks (decode/encode/mel prep).

**Model directory layout (local HuggingFace export):**
- `config.json`
- sharded `.safetensors` + index for `talker.*` and (base models) `speaker_encoder.*`
- `tokenizer.json` and `tokenizer_config.json`
- `speech_tokenizer/` subdirectory

```bash
lake build Qwen3TTSEndToEnd
lake exe Qwen3TTSEndToEnd \
  --model-dir weights/qwen3-tts \
  --text "Hello from Lean." \
  --wav-path output/qwen3tts.wav
```

Useful flags:
- `--skip-decode` to only emit codec tokens
- `--codes-path <path>` output codec token file
- `--decode-script scripts/qwen3tts_decode_codes.py` Python decoder bridge
- `--encode-audio-path <wav>` + `--encode-only` for audio -> codec IDs bridge
- `--ref-audio-path <wav>` to derive speaker embedding from reference audio
- `--speaker-mel-path <path>` intermediate speaker mel safetensors file
- `--encode-script scripts/qwen3tts_encode_audio.py` speech-tokenizer encoder bridge
- `--speaker-mel-script scripts/qwen3tts_prepare_speaker_mel.py` mel extraction bridge
- `--qwen-repo ../Qwen3-TTS` local repo fallback for importing `qwen_tts` if not installed as a package

Bridge scripts are launched with `uv run python` by default (`--python uv`).

## Qwen35RunHF

Run Qwen3.5/Qwen3.6 text generation from either a local model directory or a Hugging Face repo ID. If you pass a repo ID, Tyr resolves a local HF snapshot if present, otherwise downloads `config.json` and model safetensors to cache.

Supported repo coverage:
- The loader path is wired for the public Qwen3.5 repo ids Tyr tracks explicitly, including the 0.8B instruct/base checkpoints and the larger dense/MoE/FP8 variants, plus `Qwen/Qwen3.6-35B-A3B`, via the same shared Qwen3.5/Qwen3.6 config+weight resolution path.
- The text config loader also understands the nested `text_config` schema used by the multimodal Qwen3.6 checkpoint, so text-only callers do not need a separate config shim.

```bash
lake build Qwen35RunHF

# Small smoke-test model
lake exe Qwen35RunHF --source tiny-random/qwen3.5 --prompt "Hello from Lean."

# Official Qwen repo
lake exe Qwen35RunHF --source Qwen/Qwen3.5-0.8B --prompt "Summarize dependent types."

# Base checkpoint variant
lake exe Qwen35RunHF --source Qwen/Qwen3.5-0.8B-Base --prompt "Summarize dependent types."

# Qwen3.6 text-only from the multimodal root checkpoint
lake exe Qwen35RunHF --source Qwen/Qwen3.6-35B-A3B --prompt "Summarize dependent types."

# Prefer GPU/MPS when available.
lake exe Qwen35RunHF --source Qwen/Qwen3.5-0.8B \
  --device mps --prompt "Summarize dependent types."

# Multimodal with Apple system media path (ImageIO/AVFoundation).
# Passing --image/--video auto-enables multimodal mode.
lake exe Qwen35RunHF --source Qwen/Qwen3.5-0.8B \
  --image input.jpg --prompt "Describe this image."

# Multimodal video with temporal downsampling
lake exe Qwen35RunHF --source Qwen/Qwen3.5-0.8B \
  --video clip.mp4 --video-max-frames 64 --video-frame-stride 4 \
  --prompt "Summarize this clip."

# Multimodal streaming decode
lake exe Qwen35RunHF --source Qwen/Qwen3.5-0.8B \
  --image input.jpg --prompt "Describe this image." --stream

# Batched prompts + streaming
lake exe Qwen35RunHF --source tiny-random/qwen3.5 --prompt-file prompts.txt \
  --batch-size 4 --stream
```

Useful flags:
- `--device <auto|cpu|mps|cuda[:n]>` execution device (default: `auto`; falls back to CPU if unavailable)
- `--revision <rev>` HF branch/tag/commit (default: `main`)
- `--cache-dir <path>` override model download cache directory
- `--prompt-file <path>` one prompt per line (batched decode input)
- `--batch-size <n>` prompts per decode batch
- `--max-new-tokens <n>` number of generated tokens
- `--stream` stream tokens as they are decoded
- `--multimodal` force `Qwen35ForConditionalGeneration` (auto-enabled by `--image/--video`)
- `--image <path>` image input for multimodal prefix-feature injection (Apple-only media path)
- `--video <path>` video input for multimodal prefix-feature injection (Apple-only media path)
- `--video-max-frames <n>` cap decoded video frames for preprocessing cost (streamed decode + patchify)
- `--video-frame-stride <n>` keep every Nth decoded frame before patchification

## Qwen25OmniRunHF

Run Qwen2.5-Omni thinker text generation (3B/7B) from either a local model directory or a Hugging Face repo ID.

Supported collection coverage:
- `https://huggingface.co/collections/Qwen/qwen25-omni` (3B, 7B, and quantized 7B ids in resolver list).

```bash
lake build Qwen25OmniRunHF

# 3B thinker text path
lake exe Qwen25OmniRunHF --source Qwen/Qwen2.5-Omni-3B --prompt "Hello from Lean."

# 7B thinker text path
lake exe Qwen25OmniRunHF --source Qwen/Qwen2.5-Omni-7B --prompt "Summarize dependent types."

# Batched prompts
lake exe Qwen25OmniRunHF --source Qwen/Qwen2.5-Omni-3B --prompt-file prompts.txt --batch-size 2
```

Useful flags:
- `--revision <rev>` HF branch/tag/commit (default: `main`)
- `--cache-dir <path>` override model download cache directory
- `--prompt-file <path>` one prompt per line (batched decode input)
- `--batch-size <n>` prompts per decode batch
- `--max-new-tokens <n>` number of generated tokens

## Gemma4RunHF

Run Gemma 4 generation from either a local model directory or a Hugging Face repo ID. Tyr resolves local HF snapshots when present, otherwise downloads `config.json`, `processor_config.json`, tokenizer files, and either `model.safetensors` or the sharded safetensors set into cache.

Supported repo coverage:
- Public Gemma 4 text checkpoints explicitly covered as of 2026-04-02:
  `google/gemma-4-E2B`, `google/gemma-4-E2B-it`, `google/gemma-4-E4B`, `google/gemma-4-E4B-it`,
  `google/gemma-4-26B-A4B`, `google/gemma-4-26B-A4B-it`, `google/gemma-4-31B`, `google/gemma-4-31B-it`.
- Tyr supports the Gemma 4 text branch plus the image-conditioned multimodal path:
  Gemma 4 vision patch embedder, 2D RoPE vision encoder, spatial pooler, multimodal projector, repeated image placeholders, multi-image prompt injection, and the larger-model vision bidirectional prefill mask.
- The text path covers hybrid sliding/full attention, small-model per-layer input blocks, E2B shared-layer double-wide MLPs, and the 26B-A4B MoE text branch.

```bash
lake build Gemma4RunHF

# Official Gemma 4 E4B
lake exe Gemma4RunHF --source google/gemma-4-E4B \
  --prompt "Summarize dependent types."

# Small E2B variant
lake exe Gemma4RunHF --source google/gemma-4-E2B \
  --prompt "Write one sentence about Lean."

# Larger dense checkpoint
lake exe Gemma4RunHF --source google/gemma-4-31B \
  --prompt "Summarize dependent types."

# Streaming decode
lake exe Gemma4RunHF --source google/gemma-4-E4B \
  --prompt "Write one sentence about Lean." --stream

# Batched prompts
lake exe Gemma4RunHF --source google/gemma-4-E4B \
  --prompt-file prompts.txt --batch-size 4

# Single-image captioning
lake exe Gemma4RunHF --source google/gemma-4-E2B-it \
  --image external/ThunderKittens/assets/kittens.png \
  --prompt "Describe this image."

# Multiple images with explicit placeholders
lake exe Gemma4RunHF --source google/gemma-4-E2B-it \
  --image external/ThunderKittens/assets/kittens.png \
  --image external/ThunderKittens/assets/attn.png \
  --prompt "Image A: <|image|> Image B: <|image|> Compare them briefly."

# Larger multimodal checkpoint
lake exe Gemma4RunHF --source google/gemma-4-26B-A4B-it \
  --image external/ThunderKittens/assets/kittens.png \
  --prompt "Describe this image."
```

Useful flags:
- `--device <auto|cpu|mps|cuda[:n]>` execution device (default: `auto`; falls back to CPU if unavailable)
- `--revision <rev>` HF branch/tag/commit (default: `main`)
- `--cache-dir <path>` override model download cache directory
- `--prompt-file <path>` one prompt per line (batched decode input)
- `--image <path>` add an image input; repeat the flag for multiple images
- `--batch-size <n>` prompts per decode batch
- `--max-new-tokens <n>` number of generated tokens
- `--stream` stream tokens as they are decoded
- `--multimodal` force multimodal model load even before `--image` is parsed
- Use literal `<|image|>` markers in the prompt to place images explicitly; if omitted, images are prefixed in input order
- `--enable-thinking` use the Gemma 4 thinking-enabled chat template
- `--debug-ids` print generated token ids alongside decoded text

## BranchingFlows

Combinatorial branching flow sampler -- a port of branching flow networks to Lean. Includes continuous, discrete, and mixed training demos. No external data needed.

```bash
lake exe BranchingFlowsContinuousTrain
lake exe BranchingFlowsMoleculeTrain
lake exe BranchingFlowsMoleculeTransformerTrain
lake exe BranchingFlowsMoleculeTrainGenerate
```

Demo files:
- `BranchingFlowsDemo.lean` -- Minimal deterministic demo
- `ContinuousTrainDemo.lean` -- Continuous state training with an overfit loss-decrease check
- `DiscreteTrainDemo.lean` -- Discrete state training
- `MixedTrainDemo.lean` -- Mixed continuous/discrete training
- `MoleculeTrainDemo.lean` -- QM9-shaped coordinate/label training with an overfit loss-decrease check
- `MoleculeTransformerTrainDemo.lean` -- QM9-shaped transformer training with coordinate, label, split, and deletion loss checks
- `MoleculeGenerationDemo.lean` -- Oracle molecule bridge/generation example with XYZ export
- `MoleculeTrainGenerate.lean` -- Dataset-backed molecule transformer training, evaluation, and learned XYZ generation

Torch-backed training demos should run through Lake executables or compiled
targets so Tyr's native Torch bridge is linked. The training executables print
initial/final losses and fail if the tiny model does not improve.

## NanoProof

Transformer architecture for theorem proving with dual policy/value heads for MCTS integration. Currently model-only; RL training loop depends on MCTS/Prover modules that need to be recreated.

Architecture features: rotary embeddings, RMSNorm, QK normalization, ReLU^2 activation, Group-Query Attention (GQA).
