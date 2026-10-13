# LLM infrastructure and model families

## Purpose and when to use

`Tyr/Model/` provides shape-indexed Lean 4 implementations of the Qwen LLM lineage and Gemma 4, the shared decoding/sampling infrastructure they use, and the FLM flow-matching language-model library. Every model is a plain structure parameterized by a config *value*, with tensor fields whose compile-time shapes are computed from that config, so a mismatched checkpoint shape fails at elaboration rather than at runtime. Use this component to load a HuggingFace SafeTensors checkpoint and run KV-cached autoregressive generation, or as reference architectures when building your own models. Generative and multimodal models (Flux, VAE, TTS, ASR, Whisper) live in the same directory but are covered by the [generative](generative.md) and [audio/speech](audio-speech.md) guides.

## Architecture and main abstractions

### Family layout

| Family | Namespace | Umbrella import | Config | Causal-LM type |
|---|---|---|---|---|
| Qwen base blocks + Flux text encoder | `torch.qwen` | `Tyr.Model.Qwen` | `qwen.QwenConfig` | `qwen.Qwen3Model cfg` (no LM head) |
| Qwen3 | `torch.qwen3` | `Tyr.Model.Qwen3` | `qwen3.Config` | `qwen3.Qwen3ForCausalLM cfg` |
| Qwen3.5 (dense + MoE, hybrid attention) | `torch.qwen35` | `Tyr.Model.Qwen35` | `qwen35.Config` | `qwen35.Qwen35ForCausalLM cfg` |
| Qwen3.6 | `torch.qwen36` | `Tyr.Model.Qwen36` | `qwen36.Config` | `qwen36.Qwen36ForCausalLM` |
| Qwen2.5-Omni (thinker text) | `torch.qwen25omni` | `Tyr.Model.Qwen25Omni` | `qwen25omni.Config` | `qwen25omni.Qwen25OmniForCausalLM` |
| Gemma 4 | `torch.gemma4` | `Tyr.Model.Gemma4` | `gemma4.Config` | `gemma4.Gemma4ForCausalLM cfg` |
| FLM (flow-matching LM) | `torch.flm` | `Tyr.Model.FLM` | `flm.FlowConfig` | backbone-agnostic |

The reuse structure matters more than the table suggests:

- `torch.qwen` (`Tyr/Model/Qwen/`) is the base layer: config, GQA attention with KV cache, SwiGLU MLP, pre-norm layer, and the `Qwen3Model` backbone. Its header still says "for Flux text encoding"; it is also the substrate for Qwen3 and Qwen2.5-Omni.
- `torch.qwen3` wraps `qwen.Qwen3Model` with an LM head and KV-cached greedy generation. `torch.qwen25omni` is just 3B/7B config defaults plus HF name-prefix-aware (`thinker.`/`language_model.`) weight loading over `qwen3.Qwen3ForCausalLM` (`abbrev` at `Tyr/Model/Qwen25Omni/Weights.lean:214`).
- `torch.qwen35` is an independent, much larger implementation (own config, norms, MoE, hybrid linear/full attention) but reuses `qwen.QwenAttention.KVCache` (aliased at `Tyr/Model/Qwen35/Model.lean:320`). `torch.qwen36` has no `Model.lean` at all — it is `abbrev`s/`export`s over `qwen35` with Qwen3.6-specific defaults (`Tyr/Model/Qwen36/Pretrained.lean:40-41`).
- `torch.gemma4` mirrors the qwen35 file layout (Config/Model/Weights/ConfigIO/Pretrained/VL*/Multimodal/Media).

### The per-family idiom

Each pretrained-capable family splits into the same files:

- `Config.lean` — a config structure whose fields mirror HF `config.json` names (`hidden_size`, `num_key_value_heads`, …), plus presets (`Config.qwen35_9B`, `Config.gemma4_E4B`, …).
- `ConfigIO.lean` — hand-rolled JSON parsing on `Lean.Data.Json`: `Config.loadFromFile`, `Config.loadFromPretrainedDir`. Qwen3 additionally validates invariants (head divisibility, nonzero dims) and throws `IO.userError` on violation (`Tyr/Model/Qwen3/ConfigIO.lean:66-88`).
- `Model.lean` — the model structure, forwards, and generation loops.
- `Weights.lean` — SafeTensors loaders: `loadSharded` (sharded HF directory, the de-facto path) and `load` (single `model.safetensors` file). Loaded tensors are frozen via `reqGradFalse`; FP8 checkpoints are dequantized on load.
- `Pretrained.lean` — `loadFromPretrained` built on `Tyr.Hub.loadModelFromPretrained`, plus the explicit list of covered HF repo ids.

The model itself is a structure over a config value (Qwen3 shown, `Tyr/Model/Qwen3/Model.lean:52`):

```lean
structure Qwen3ForCausalLM (cfg : Config) where
  model : qwen.Qwen3Model cfg
  lmHead : T #[cfg.vocab_size, cfg.hidden_size]
  tieWordEmbeddings : Bool := true
  deriving TensorStruct
```

Because `cfg` is a value parameter, one implementation serves every checkpoint size — the preset just picks the shapes. `deriving TensorStruct` makes the whole model traversable for device moves, checkpointing, and optimizers (see [TensorStruct](../core/tensorstruct.md)). Every family also follows the same decode convention: generation returns `IO (Sigma (fun outSeq => T #[batch, outSeq]))` because the output length is runtime-determined (EOS early stop), and batched per-row EOS masking happens on-device via `tokenInSet` / `applyFinishedEos`.

### Shared generation infrastructure

`Tyr/Model/Generation.lean` (namespace `torch.Model`) is the common sampling surface:

```lean
inductive SamplingStrategy where
  | greedy
  | multinomial (temperature : Float := 1.0) (topK : UInt64 := 0) (topP : Float := 1.0)

/-- Callback invoked on every generated token during streaming decode. -/
abbrev StreamCallback (batch : UInt64) := UInt64 → T #[batch] → IO Unit

def sampleFromLogits (logits : T #[batch, vocab]) (strategy : SamplingStrategy) : IO (T #[batch])
def tokenInSet (tokens : T #[n]) (values : Array UInt64) : T #[n]
def applyFinishedEos (tokens : T #[n]) (finished : T #[n]) (eosToken : UInt64) : T #[n]
```

`sampleFromLogits` semantics (`Generation.lean:26-47`): greedy is `nn.argmax`; multinomial throws unless `temperature > 0`, scales logits by `1/temperature` (skipped at exactly 1.0), applies a top-K filter unless `topK == 0`, then a top-P filter unless `topP >= 1.0`, then softmax + `nn.multinomial`. Qwen3.5 and Gemma 4 decode through this; Qwen3 (and therefore Qwen2.5-Omni) has its own private greedy loops and supports **greedy decoding only** — no temperature/top-k/top-p.

### KV caches

The shared incremental cache is `qwen.QwenAttention.KVCache` (`Tyr/Model/Qwen/Attention.lean:54`):

```lean
structure KVCache (batch num_kv_heads head_dim : UInt64) where
  kStoreDyn : T #[]
  vStoreDyn : T #[]
  seq : UInt64 := 0
  maxLen : UInt64 := 0
```

Buffers are preallocated at `[batch, num_kv_heads, maxLen, head_dim]` (`initKVCache maxLen device`) and stored shape-erased (`nn.eraseShape` → `T #[]`) so the cache *type* does not change as the sequence grows; each step restores the static shape with `reshape` + `castLike`, writes the new K/V via `data.sliceScatter`, attends over the filled prefix, and re-erases. This sidesteps dependent-shape bookkeeping inside IO decode loops.

Two families extend it: Qwen3.5's `HybridCache` (`Qwen35/Model.lean:765`) adds per-layer depthwise-conv states `[batch, convDim, kernel]` and recurrent states `[batch, vHeads, kDim, vDim]` for the gated-delta linear-attention layers; Gemma 4's `Gemma4Cache` (`Gemma4/Model.lean:797`) is one KV cache per layer. `QwenAttention.forwardStep` takes a `useTyrFlashAttn : Bool := false` flag that routes eligible decode shapes (BF16, qSeq = 1, head_dim 64 or 128, GQA-valid) through `nn.tyrFlashAttn4d` to a ThunderKittens H100 kernel and falls back to PyTorch SDPA otherwise (`Qwen/Attention.lean:149-157`).

Laguna generation uses `LagunaCacheSession`, an inference-only cache with private
storage and serialized IO updates. Full-attention layers reserve `maxLen`
positions; sliding layers reserve `min maxLen sliding_window` positions in a
ring. Prefill keeps the newest window, and decode writes one K/V position in
place. Every layer tracks the absolute sequence position separately from its
physical ring slot. `LagunaModel.initCacheSession`,
`LagunaForCausalLM.prefillSession`, and `decodeSession` expose this path;
`LagunaCacheSession.fork` clones storage for independent continuations.
Aliasing a session shares its state. Invalid positions and table coverage are
rejected before mutation; a failure during a transition invalidates the session.
The existing functional Laguna cache methods preserve earlier tensor snapshots.

Full Laguna attention uses implicit causal GQA. The portable SDPA fallback
processes bounded query chunks and avoids materialized K/V head repeats;
sliding attention also slices the keys reachable by each chunk. Native fused
backends remain available without changing process-global backend preferences.
The bounds concern inference temporaries: training autograd can retain
intermediates across all chunks. Rectangular cache decode uses non-causal SDPA,
since every retained key is attendable after its absolute-position RoPE.

The low-level `torch.Generator.KVCache` API also allocates through IO so Lean
cannot merge identical K/V allocations. `Cache.init`, append/attention calls,
views, and sequence advancement now return IO results. This cache remains
shared and mutable: callers serialize access and discard earlier cache values
after updates. Use Laguna sessions for locking and independent forks.

`TYR_LAGUNA_CACHE_BENCH=1 lake exe LagunaModelTest` runs an optional
functional/session comparison at capacities 128 and 8192, reporting synchronized
decode timings and cache tensor bytes. The normal model gate covers ring wrap,
forks, functional snapshots, and FP32/BF16 parity.

### Qwen base layer (`torch.qwen`)

`Tyr/Model/Qwen/` contains the shared transformer: `QwenConfig` (defaults are Qwen3-4B; `QwenConfig.fluxKleinTextEncoder` is the 36-layer Flux 2 Klein variant with Q/K norms), `QwenAttention` (GQA with optional per-head Q/K RMSNorm), `QwenMLP` (SwiGLU), `QwenLayer`, and `Qwen3Model cfg` (embedding + `Array QwenLayer` + final `RMSNorm`). `QwenFluxEmbedder` (`Qwen/Embedder.lean:25`) extracts hidden states from selected layers (default `#[8, 17, 26]`) and concatenates them into Flux text embeddings; `loadQwenFluxEmbedderSharded` (`Qwen/Weights.lean:233`) loads it from a sharded checkpoint.

### Qwen3.5 / Qwen3.6 (`torch.qwen35`, `torch.qwen36`)

`Qwen35Config` (`Tyr/Model/Qwen35/Config.lean:45`) covers dense and MoE checkpoints (`num_experts == 0` means dense — see presets below) and the hybrid layer schedule:

```lean
inductive LayerType where
  | linearAttention
  | fullAttention
```

Presets: `Config.qwen35_0_8B`, `Config.qwen35_9B`, `Config.qwen35_35B_A3B`, `Config.qwen36_35B_A3B`. Derived dimensions (`rotaryDim`, `linearKeyDim`, `linearConvDim`, `numHeadsPerKVGroup`, `isMoE`) are functions of the config; `Config.normalize` fills in `layer_types` from `full_attention_interval` when absent.

`Qwen35ForCausalLM` (`Qwen35/Model.lean:1065`) assembles: `Qwen35RMSNorm` (zero-centered `(1 + w)` convention; `fromCheckpointWeight` absorbs the HF offset), `Qwen35RMSNormGated`, `Qwen35MLP`, a pure-Lean MoE block (`Qwen35TopKRouter` + `Qwen35MoeExperts` + `Qwen35SparseMoeBlock`, one-hot dispatch + `einsum2`), `Qwen35Attention` (fused Q-gate projection, per-head Q/K norm, partial RoPE), and `Qwen35GatedDeltaNet` (linear attention: fused QKV projection + depthwise causal conv1d + gated delta rule through the C++ externs below). The multimodal wrapper `Qwen35ForConditionalGeneration (cfg : VLConfig)` (`Qwen35/Multimodal.lean:319`) pairs a `Qwen35VisionModel` tower with the text model and scatters image/video features into token embeddings; image/video patch loaders in `torch.qwen35.media` are Apple-only. `torch.qwen36` re-exports all of this with Qwen3.6 defaults and its own repo list.

### Gemma 4 (`torch.gemma4`)

`Gemma4Config` (`Tyr/Model/Gemma4/Config.lean:42`) uses a sliding/full attention schedule (`LayerType.slidingAttention | .fullAttention`), separate RoPE thetas and head dims for the two kinds, KV-shared layers, per-layer input projections, optional MoE, and final logit softcapping (`final_logit_softcapping`, default 30.0, applied via `nn.softcap`). Presets: `Config.gemma4_E2B`, `gemma4_E4B`, `gemma4_26B_A4B`, `gemma4_31B`. Embeddings are scaled by `sqrt(hidden_size)` (`Gemma4/Model.lean:1048`). Unlike the other families the MoE path is a fused C++ extern (`routedTextExpertsForward`). The generate family is `generate` / `generateStream` / `generateFromEmbeds(Stream)` (+ per-layer-input variants) — there is no `generateUncached` reference loop. `Gemma4ForConditionalGeneration` (`Gemma4/Multimodal.lean:350`) adds the vision tower. Note there are currently no Gemma 4 tests in `Tests/`.

### FFI-backed kernels

Three `@[extern]` opaque kernels sit inside this component (declared `private`, so you cannot call them directly — they run inside the layer forwards):

- `chunkGatedDeltaRule` / `recurrentGatedDeltaRule` — Qwen3.5 gated delta rule, `Qwen35/Model.lean:50-61` → `cc/src/tyr_qwen35.cpp:172`.
- `routedTextExpertsForward` — Gemma 4 fused MoE expert forward, `Gemma4/Model.lean:27-32` → `cc/src/tyr.cpp:1850`.

### Model utils (`torch.Model`)

`Tyr/Model/Utils.lean` holds cross-family helpers: `initWeight` (Kaiming-like `randn * sqrt(2 / fanIn)`, trainable), `initBias`, `reqGradFalse` (freeze loaded tensors), and FP8 blockwise dequantization `dequantizeFP8` / `dequantizeFP8Experts` (128×128 blocks, HF `weight_scale_inv` convention).

### FLM: flow-matching language models (`torch.flm`)

`Tyr/Model/FLM.lean` ports the FLM/FMLM tensor contracts (continuous one-hot corruption, flow-matching loss, ODE sampling, PSD semigroup targets) from the sibling FLM repository. It is backbone-agnostic: the caller supplies the denoiser through an interface structure, so it works with any transformer, including the families above.

```lean
structure FlowDenoiser (seq vocab : UInt64) (Params : Type) where
  forward : {batch : UInt64} → Params → T #[batch, seq, vocab] → T #[batch]
    → Option (T #[batch]) → IO (T #[batch, seq, vocab])
  forwardPure : Option ({batch : UInt64} → Params → T #[batch, seq, vocab] → T #[batch]
    → Option (T #[batch]) → T #[batch, seq, vocab]) := none
  lossWeight : Option ({batch : UInt64} → Params → T #[batch] → Option (T #[batch])
    → IO (T #[batch])) := none
```

`forwardPure` is required for generation because `Tyr.DiffEq` vector fields are pure functions; `generateFLM` throws if it is absent. Sampling treats the flow as an ODE from Gaussian noise to one-hot data and integrates it with `DiffEq.Euler` (`generateFLM`) or any `DiffEq.AbstractSolver` you supply (`generateFLMWithSolver`). FLM is currently library-only: its sole consumer in the repo is `Tests/TestFLM.lean`.

## Key APIs

### Generation (`Tyr/Model/Generation.lean`, `torch.Model`)

| Symbol | Signature / meaning |
|---|---|
| `SamplingStrategy` | `.greedy` or `.multinomial (temperature := 1.0) (topK := 0) (topP := 1.0)`; `topK = 0` / `topP >= 1` disable those filters |
| `StreamCallback batch` | `UInt64 → T #[batch] → IO Unit`, invoked per decode step |
| `sampleFromLogits` | `T #[batch, vocab] → SamplingStrategy → IO (T #[batch])` |
| `tokenInSet` | `T #[n] → Array UInt64 → T #[n]` boolean membership mask |
| `applyFinishedEos` | `T #[n] → T #[n] → UInt64 → T #[n]` overwrite finished rows with EOS |

### Model utils (`Tyr/Model/Utils.lean`, `torch.Model`)

| Symbol | Signature |
|---|---|
| `initWeight` | `(shape : Shape) → (fanIn : UInt64) → IO (T shape)` |
| `initBias` | `(shape : Shape) → T shape` |
| `reqGradFalse` | `T s → T s` |
| `dequantizeFP8` | `T #[out, in] → T #[out/128, in/128] → T #[out, in]` |
| `dequantizeFP8Experts` | `T #[e, out, in] → T #[e, out/128, in/128] → T #[e, out, in]` |

### Qwen3 (`torch.qwen3`)

| Symbol | Signature |
|---|---|
| `Config` | `abbrev Config := qwen.QwenConfig`; preset `Config.qwen3_4B` |
| `Config.loadFromFile` / `Config.loadFromPretrainedDir` | `(pathOrDir : String) (defaults := Config.qwen3_4B) : IO Config`, with invariant validation |
| `Qwen3ForCausalLM.init` | `(cfg : Config) (tieWordEmbeddings := true) : IO (Qwen3ForCausalLM cfg)` |
| `forward` / `forwardEmbeds` | `(m) (inputIds : T #[batch, seq]) (attnMask := none) : T #[batch, seq, cfg.vocab_size]` |
| `generateGreedy` | `(m) (inputIds) (maxNewTokens := 512) (eosTokenIds := #[]) : IO (Sigma (fun outSeq => T #[batch, outSeq]))`, KV-cached |
| `generateGreedyUncached` | same signature; full re-forward per step, used as the parity reference |
| `Qwen3ForCausalLM.loadSharded` / `load` | `(dirOrPath : String) (cfg := Config.qwen3_4B) (log := {}) : IO (Qwen3ForCausalLM cfg)`; ties embeddings when `lm_head.weight` is absent |

### Qwen3.5 / Qwen3.6 (`torch.qwen35`, `torch.qwen36`)

```lean
-- all in namespace Qwen35ForCausalLM; called as `model.generate cfg …`
def generate (cfg : Config) (m : Qwen35ForCausalLM cfg) (inputIds : T #[batch, seq])
    (maxNewTokens : UInt64 := 256) (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[]) : IO (Sigma (fun outSeq => T #[batch, outSeq]))
def generateStream      -- same, plus `(onStep : StreamCallback batch)` after `inputIds`
def generateFromEmbeds  -- same as `generate`, plus precomputed `inputsEmbeds`
def generateUncached    -- same as `generate`; full re-forward reference
def generateGreedy      -- same as `generate` minus the strategy argument

def Qwen35ForCausalLM.loadSharded (modelDir : String) (cfg : Config := Config.qwen35_9B)
    (device : Device := Device.CPU) (log : Handlers := {}) : IO (Qwen35ForCausalLM cfg)
def Qwen35ForCausalLM.loadFromPretrained
    (source : String) (defaults : Config := Config.qwen35_9B)
    (revision : String := "main") (cacheDir : String := Hub.defaultCacheDir)
    (device : Device := Device.CPU) : IO (Sigma (fun cfg => Qwen35ForCausalLM cfg))
```

`loadFromPretrained` accepts a local directory or an HF repo id, downloads via `Tyr.Hub`, parses `config.json` over the defaults, picks sharded vs single-file layout, and returns the config alongside the model as a `Sigma` (the config is only known at runtime). The covered repo ids are listed in `qwen35.hub.qwen35CollectionRepoIds` (`Qwen35/Pretrained.lean:19`). Qwen3.6 exposes the identical surface as `qwen36.Qwen36ForCausalLM` with `Config.qwen36_35B_A3B` defaults. The multimodal twins `Qwen35ForConditionalGeneration` / `Qwen36ForConditionalGeneration` have the same shape plus `getImageFeatures` / `getVideoFeatures`.

### Gemma 4 (`torch.gemma4`)

Same pattern: `Gemma4ForCausalLM.generate` / `generateStream` / `generateFromEmbeds` with `(maxNewTokens := 256) (strategy := .greedy) (eosTokenIds := #[])`; `loadSharded (modelDir) (cfg := Config.gemma4_E4B) (log := {})`; `loadFromPretrained (source) (defaults := Config.gemma4_E4B) (revision) (cacheDir)` (no device argument — loading is CPU-side; move the model with `TensorStruct` afterwards). Covered repos: `gemma4.hub.gemma4CollectionRepoIds`.

### FLM (`torch.flm`)

| Symbol | Purpose |
|---|---|
| `FlowConfig` | `tMin`/`tMax`/`softcap`/`eps`/`weightDecay`/`gradClip` knobs |
| `TimeMap.identity` | linear tau↔t map (plug LUT-based conversions in here) |
| `corruptContinuous` | `x_t = (1 - t)·noise + t·oneHot(x)` corruption, returns `(x_t, target)` |
| `flowLoss` / `flowLossGivenTau` | per-token flow-matching loss, `IO (T #[batch, seq])` |
| `trainStep` | full AdamW step; returns `(Params × Optim.AdamWState Params × LossReport)` |
| `generateFLM` | ODE sampling with DiffEq Euler, `(steps : Nat) → IO (T #[batch, seq])` |
| `generateFLMWithSolver` | same with a caller-supplied `DiffEq.AbstractSolver` |
| `psdLossGivenTau` / `diagonalLossGivenTau` / `samplePsdTimes` | FMLM semigroup losses and time sampling |

## Usage examples

### Pretrained Qwen3.5 text generation

Reconstructed example (from `Examples/Qwen35/RunHF.lean`, built as `lake exe Qwen35RunHF`):

```lean
import Tyr.Hub
import Tyr.Model.Qwen35
import Tyr.Tokenizer.Qwen35
import Examples.ModelRunner

open torch torch.Model torch.qwen35

def main : IO Unit := do
  -- Local dir or HF repo id; downloads tokenizer files on demand.
  let modelDir ← Hub.resolvePretrainedDir "Qwen/Qwen3.5-0.8B"
    { revision := "main", cacheDir := Hub.defaultCacheDir, includeTokenizer := true }
  let cfg ← Config.loadFromPretrainedDir modelDir Config.qwen35_9B
  let model ←
    if ← Hub.detectWeightLayout modelDir then
      Qwen35ForCausalLM.loadSharded modelDir cfg Device.CPU
    else
      Qwen35ForCausalLM.load s!"{modelDir}/model.safetensors" cfg Device.CPU

  let tok ← tokenizer.qwen35.loadTokenizer modelDir
  let encode (prompt : String) : IO (Array UInt64) :=
    pure ((tokenizer.qwen35.encodeText tok (tokenizer.qwen35.chatTemplate prompt)).map (·.toUInt64))
  -- Pads to a batch; shapes are existentially packed because seq is runtime data.
  let ⟨batch, ⟨seq, (inputIds, _promptLens)⟩⟩ ←
    Examples.ModelRunner.buildBatchInputWithEncoder tok.padToken.toUInt64 #["What is Lean?"] encode

  let eos : Array UInt64 := match cfg.eos_token_id with | some id => #[id] | none => #[]
  let ⟨_outSeq, out⟩ ← model.generate cfg inputIds 32 .greedy eos
  -- streaming variant: model.generateStream cfg inputIds onStep 32 (.multinomial 0.7 40 0.9) eos
  IO.println s!"output shape: {out.runtimeShape}"
```

### Random-weight Qwen3: forward and cached/uncached parity

Reconstructed example (from `Tests/TestQwen3Model.lean`):

```lean
import Tyr
import Tyr.Model.Qwen3

open torch torch.qwen3

private def tinyCfg : Config := {
  vocab_size := 64, hidden_size := 32, intermediate_size := 64
  num_hidden_layers := 2, num_attention_heads := 4, num_key_value_heads := 2
  head_dim := 8, rope_theta := 10000.0, rms_norm_eps := 1e-6
  max_position_embeddings := 128
}

def main : IO Unit := do
  let model ← Qwen3ForCausalLM.init tinyCfg
  let ids : T #[1, 3] := reshape (data.fromInt64Array #[3, 2, 1]) #[1, 3]
  let logits := model.forward ids                 -- T #[1, 3, 64]
  let ⟨outSeq, _out⟩ ← model.generateGreedy ids 4 #[]
  IO.println s!"logits: {logits.runtimeShape}, generated {outSeq} tokens total"
```

This tiny-config pattern is how the test suites exercise the models without checkpoints; `Tests/TestQwen35Model.lean` does the same for dense and MoE Qwen3.5 configs, including cached-vs-uncached parity.

### Flux text encoding with the Qwen embedder

Reconstructed example (from `Examples/Flux/FluxDemo.lean:128-174`):

```lean
let qwenCfg := QwenConfig.fluxKleinTextEncoder
let qwen ← loadQwenFluxEmbedderSharded qwenDir qwenCfg maxSeqLen #[8, 17, 26]
-- tokens, attnMask : T #[1, maxSeqLen]
let txtEmb := qwen.encodeMasked qwenCfg maxSeqLen tokens attnMask
-- txtEmb : T #[1, maxSeqLen, 3 * qwenCfg.hidden_size]
```

### FLM training loss and ODE sampling

Reconstructed example (from `Tests/TestFLM.lean`):

```lean
import Tyr.Model.FLM

open torch torch.flm

def identityDenoiser : FlowDenoiser 3 5 Unit :=
  { forward := fun {_batch} _ x _ _ => pure x
    forwardPure := some (fun {_batch} _ x _ _ => x) }

def main : IO Unit := do
  let tokens : T #[2, 3] := reshape (data.fromInt64Array #[0, 1, 2, 3, 4, 0]) #[2, 3]
  let loss ← flowLoss (cfg := {}) TimeMap.identity identityDenoiser () tokens
  IO.println s!"mean loss: {nn.item (nn.meanAll loss)}"
  let out ← generateFLM (batch := 2) (cfg := {}) TimeMap.identity identityDenoiser () 8
  IO.println s!"sampled tokens: {out.runtimeShape}"  -- T #[2, 3] after argmax
```

## Caveats

- Some parsed config fields are accepted but never read by the models — e.g. `Qwen35Config.attention_dropout`, `hidden_act`, `mamba_ssm_dtype`, `mrope_*`, `mtp_*`, `use_cache`, and `Gemma4Config.attention_dropout`, `use_bidirectional_attention`. Setting them is a silent no-op.
- Qwen3 / Qwen2.5-Omni generation is greedy-only; `SamplingStrategy.multinomial` is honored by Qwen3.5/Qwen3.6 and Gemma 4 only.
- `load` (single-file) exists next to `loadSharded` on every family for backwards compatibility; sharded directories are the tested, de-facto path.
- `torch.qwen35.media` / `torch.gemma4` media loaders are Apple-only; there is no runtime capability check.

## Related guides

- [Getting started](../getting-started.md) — build, FFI setup, first model run
- [Tensors](../core/tensors.md) — `T s`, shapes, and the ops used in forwards
- [TensorStruct](../core/tensorstruct.md) — the parameter-tree class behind `deriving TensorStruct`
- [Modules](../modules.md) — `Linear`/`RMSNorm` and the `Module` classes these models build on
- [Serialization](../serialization.md) — SafeTensors loading and `Tyr.Hub` downloads
- [Data and tokenizers](../data.md) — the `tokenizer.qwen35` / `tokenizer.gemma4` codecs
- [Generative models](generative.md) — Flux, which consumes `QwenFluxEmbedder`
- [Audio and speech models](audio-speech.md) — Qwen3-ASR/TTS, built on the same Qwen base
- [DiffEq](../diffeq.md) — the solver interface behind `generateFLMWithSolver`
- [GPU kernels](../gpu/kernels.md) — the ThunderKittens decode kernel behind `useTyrFlashAttn`
- [Examples and testing](../examples-and-testing.md) — the `*RunHF` executables and model test suites

Exhaustive symbol-level documentation for everything in this chapter is generated by doc-gen4 (see `docbuild/`); this guide covers the concepts and the main entry points only.
