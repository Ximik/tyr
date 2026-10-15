/-
  Tyr/Model/Laguna/Model.lean

  Full poolside Laguna text causal-LM (Laguna-S-2.1, NVFP4 variant):
  gated attention (per-head softplus output gate), dense SwiGLU / sparse MoE
  FFN, interleaved full + sliding-window attention layers, static KV cache,
  and cached greedy/sampled generation.

  Mirrors the HuggingFace reference in `dev/laguna_reference/modeling_laguna.py`
  (`LagunaAttention`, `LagunaMLP`, `LagunaDecoderLayer`, `LagunaModel`,
  `LagunaForCausalLM`) and follows the structural idioms of
  `Tyr/Model/Qwen35/Model.lean` (KV cache, `forwardWithCache`/`forwardStep`,
  `decodeLoopCached`) and `Tyr/Model/Gemma4/Model.lean` (sliding-window decode
  KV truncation: `useLen = min kvLen sliding_window`).

  Attention math per layer (H = 48 full / 72 sliding, KV = 8, head_dim = 128,
  no biases anywhere):

    q,k,v      = q_proj/k_proj/v_proj(xn)        xn = input_layernorm output
    q,k        = per-head RMSNorm over head_dim  (q_norm/k_norm, BEFORE rope)
    q,k        = RoPE  (full layers: partial rotary on the first
                        `cfg.rotaryDimFull` = 64 channels with YaRN tables;
                        sliding layers: full-width 128 rotary with plain tables)
    attn       = SDPA, GQA, scale = head_dim^-0.5 (the tyr SDPA default);
                 full layers causal, sliding layers causal + window
                 `cfg.sliding_window` = 512
    gate       = softplus(g_proj(xn).float()).to(bf16)   [B, T, H]
    attn       = attn * gate.unsqueeze(-1)               per-head gate
    out        = o_proj(attn.flatten(-2))

  Layer: `h = x + attn(input_layernorm(x))`;
         `out = h + ffn(post_attention_layernorm(h))` with ffn = dense SwiGLU
  (layers in `cfg.mlp_only_layers`) or `LagunaSparseMoeBlock` (wave 1).

  Deviations from the HF reference (all below BF16 rounding or documented):
  - RMSNorm multiplies by the weight in FP32 and casts back once, instead of
    casting first and multiplying in BF16 (same idiom as Qwen35).
  - `forwardWithCache` is a PREFILL entry point: it writes the KV cache at
    offset 0 and assumes a fresh cache (chunked prefill is not supported).
  - The random-init path (`*.init`) synthesizes NVFP4 expert banks from int64
    "bytes" + F32 scales (dequant only needs `toLong`/`floatDynamic`); checkpoint
    loading replaces them with real U8 / F8_E4M3 tensors (see Weights.lean).
  - Full prefill uses implicit causal GQA. Sliding attention bounds temporary
    masks and scores by query chunks and reachable window keys. The portable
    dispatcher consults available backends without changing global flags;
    its math fallback avoids materialized K/V head repeats. These bounds apply
    to inference; autograd may retain chunk intermediates for backward.
  - Generation uses an owned inference session: full layers append in place,
    sliding layers store one window in a ring, and fork explicitly clones
    storage. The functional cache APIs retain snapshot semantics.
-/
import Tyr.Model.Laguna.TypedOps
import Tyr.Inference.OwnedKV
import Std.Sync.Mutex
import Tyr.TensorStruct
import Tyr.TensorStruct.DeriveCast
import Tyr.Module.Derive
import Tyr.Model.Qwen.Attention
import Tyr.Model.Laguna.Config
import Tyr.Model.Laguna.Rope
import Tyr.Model.Laguna.MoE
import Tyr.Model.Utils
import Tyr.Model.Generation

namespace torch.laguna

variable {device : Device} {dtype : DType}
variable [DType.RealFloating dtype]
open torch
open torch.Model torch.qwen35

/-! ## RMSNorm helpers (plain one-centered LagunaRMSNorm) -/

/-- FP32 RMSNorm over the last dim of a 2D tensor, then weight scale.
    Matches HF `LagunaRMSNorm`: `weight * (x / sqrt(mean x² + eps))`. -/
private def rmsNormWeighted2d {n dim : UInt64}
    (weight : SomeDTypeTensor #[dim] device)
    (x : Tensor { dtype := dtype, device := device, shape := #[n, dim] })
    (eps : Float)
    : Tensor { dtype := dtype, device := device, shape := #[n, dim] } :=
  let xf : Tensor { dtype := .Float32, device := device, shape := #[n, dim] } := floatDynamic x
  let var : Tensor { dtype := .Float32, device := device, shape := #[n, 1] } := Tensor.meanDim (Tensor.mul xf xf) 1 true
  let inv : Tensor { dtype := .Float32, device := device, shape := #[n, 1] } := rsqrtDynamic (Tensor.addScalar var eps)
  let inv : Tensor { dtype := .Float32, device := device, shape := #[n, dim] } := Tensor.expand inv #[n, dim]
  let w : Tensor { dtype := .Float32, device := device, shape := #[n, dim] } := Tensor.expand (Tensor.reshapeChecked (floatDynamic weight.2) #[1, dim]) #[n, dim]
  Tensor.castLike x (Tensor.mul (Tensor.mul xf inv) w)

/-- FP32 RMSNorm over the last dim of a 3D `[batch, seq, dim]` tensor. -/
private def rmsNormWeighted3d {batch seq dim : UInt64}
    (weight : SomeDTypeTensor #[dim] device)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, seq, dim] })
    (eps : Float)
    : Tensor { dtype := dtype, device := device, shape := #[batch, seq, dim] } :=
  let flat : Tensor { dtype := dtype, device := device, shape := #[batch * seq, dim] } := Tensor.reshapeChecked x #[batch * seq, dim]
  Tensor.reshapeChecked (rmsNormWeighted2d weight flat eps) #[batch, seq, dim]

/-- Per-head RMSNorm (q_norm/k_norm) over the head_dim of a 4D
    `[batch, seq, nHeads, headDim]` tensor. -/
private def applyHeadNorm {batch seq nHeads headDim : UInt64}
    (weight : SomeDTypeTensor #[headDim] device)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, seq, nHeads, headDim] })
    (eps : Float)
    : Tensor { dtype := dtype, device := device, shape := #[batch, seq, nHeads, headDim] } :=
  let flat : Tensor { dtype := dtype, device := device, shape := #[batch * seq * nHeads, headDim] } := Tensor.reshapeChecked x #[batch * seq * nHeads, headDim]
  Tensor.reshapeChecked (rmsNormWeighted2d weight flat eps) #[batch, seq, nHeads, headDim]

/-! ## Dense SwiGLU MLP (layer 0 / `mlp_only_layers`) -/

/-- Dense SwiGLU MLP for Laguna (`LagunaMLP` in the HF reference). -/
structure LagunaMLP (cfg : Config) (dtype : DType := .Float32) where
  gate_proj : Tensor { dtype := dtype, device := device, shape := #[cfg.intermediate_size, cfg.hidden_size] }
  up_proj : Tensor { dtype := dtype, device := device, shape := #[cfg.intermediate_size, cfg.hidden_size] }
  down_proj : Tensor { dtype := dtype, device := device, shape := #[cfg.hidden_size, cfg.intermediate_size] }
  deriving TensorStruct


namespace LagunaMLP

def init (cfg : Config) (device : Device := .CPU) (dtype : DType := .Float32) : IO (LagunaMLP (device := device) (dtype := dtype) cfg) := do
  let gate ← initDynamicWeight (device := device) (dtype := dtype) #[cfg.intermediate_size, cfg.hidden_size] cfg.hidden_size
  let up ← initDynamicWeight (device := device) (dtype := dtype) #[cfg.intermediate_size, cfg.hidden_size] cfg.hidden_size
  let down ← initDynamicWeight (device := device) (dtype := dtype) #[cfg.hidden_size, cfg.intermediate_size] cfg.intermediate_size
  pure { gate_proj := gate, up_proj := up, down_proj := down }

/-- `down_proj(silu(gate_proj(x)) * up_proj(x))` (`hidden_act = "silu"`). -/
def forward {batch seq : UInt64}
    (m : LagunaMLP (device := device) (dtype := dtype) cfg)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] })
    : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] } :=
  Tensor.linear3d
    (Tensor.mul (Tensor.silu (Tensor.linear3d x m.gate_proj))
      (Tensor.linear3d x m.up_proj)) m.down_proj

end LagunaMLP

/-- 3D convenience wrapper around the wave-1 `LagunaSparseMoeBlock.forward2d`. -/
def LagunaSparseMoeBlock.forward3d {batch seq : UInt64}
    (cfg : Config)
    (m : LagunaSparseMoeBlock (device := device) (dtype := dtype) cfg)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] })
    : IO (Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] }) := do
  let flat : Tensor { dtype := dtype, device := device, shape := #[batch * seq, cfg.hidden_size] } := Tensor.reshapeChecked x #[batch * seq, cfg.hidden_size]
  let out ← m.forward2d cfg flat
  pure (Tensor.reshapeChecked out #[batch, seq, cfg.hidden_size])

/-! ## Attention -/

/-- Laguna attention block: no QKV bias, per-head Q/K RMSNorm before RoPE,
    per-head softplus output gate (`g_proj`) applied before `o_proj`.

    `numHeads` is the per-layer query head count
    (`cfg.num_attention_heads` on full layers, `cfg.num_attention_heads_sliding`
    on sliding-window layers); KV heads are always `cfg.num_key_value_heads`. -/
structure LagunaAttention (cfg : Config) (numHeads : UInt64) (dtype : DType := .Float32) where
  q_proj : Tensor { dtype := dtype, device := device, shape := #[numHeads * cfg.head_dim, cfg.hidden_size] }
  k_proj : Tensor { dtype := dtype, device := device, shape := #[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size] }
  v_proj : Tensor { dtype := dtype, device := device, shape := #[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size] }
  o_proj : Tensor { dtype := dtype, device := device, shape := #[cfg.hidden_size, numHeads * cfg.head_dim] }
  /-- Per-head gate projection: one softplus gate per head, broadcast over head_dim. -/
  g_proj : Tensor { dtype := dtype, device := device, shape := #[numHeads, cfg.hidden_size] }
  q_norm : SomeDTypeTensor #[cfg.head_dim] device
  k_norm : SomeDTypeTensor #[cfg.head_dim] device
  deriving TensorStruct


namespace LagunaAttention

/-- Per-layer static KV cache (existentially indexed K/V stores of shape
    `[batch, cfg.num_key_value_heads, maxLen, cfg.head_dim]`). -/
abbrev KVCache (cfg : Config) (batch : UInt64) (dtype : DType := .Float32) :=
  qwen.QwenAttention.KVCache (device := device) batch cfg.num_key_value_heads cfg.head_dim dtype

def init (cfg : Config) (numHeads : UInt64) (device : Device := .CPU) (dtype : DType := .Float32) : IO (LagunaAttention (device := device) (dtype := dtype) cfg numHeads) := do
  let q ← initDynamicWeight (device := device) (dtype := dtype) #[numHeads * cfg.head_dim, cfg.hidden_size] cfg.hidden_size
  let k ← initDynamicWeight (device := device) (dtype := dtype) #[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size] cfg.hidden_size
  let v ← initDynamicWeight (device := device) (dtype := dtype) #[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size] cfg.hidden_size
  let o ← initDynamicWeight (device := device) (dtype := dtype) #[cfg.hidden_size, numHeads * cfg.head_dim] (numHeads * cfg.head_dim)
  let g ← initDynamicWeight (device := device) (dtype := dtype) #[numHeads, cfg.hidden_size] cfg.hidden_size
  let qn ← Tensor.parameter (onesDynamic (device := device) #[cfg.head_dim])
  let kn ← Tensor.parameter (onesDynamic (device := device) #[cfg.head_dim])
  pure {
    q_proj := q
    k_proj := k
    v_proj := v
    o_proj := o
    g_proj := g
    q_norm := qn.packDType (by simp)
    k_norm := kn.packDType (by simp)
  }

/-- QKV projection + per-head Q/K RMSNorm + (partial) RoPE.
    `rotaryDim` is `cfg.rotaryDimFull` (64, YaRN) on full-attention layers and
    `cfg.rotaryDimSliding` (128, plain) on sliding-window layers; cos/sin are
    the matching rows sliced from `LagunaRotaryTables`. -/
private def qkvRope {batch seq rotaryDim numHeads : UInt64}
    (cfg : Config)
    (m : LagunaAttention (device := device) (dtype := dtype) cfg numHeads)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] })
    (cos : Tensor { device := device, shape := #[seq, rotaryDim / 2], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[seq, rotaryDim / 2], dtype := .Float32 })
    : Tensor { dtype := dtype, device := device, shape := #[batch, seq, numHeads, cfg.head_dim] }
      × Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] }
      × Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] } :=
  let qFlat : Tensor { dtype := dtype, device := device, shape := #[batch, seq, numHeads * cfg.head_dim] } := Tensor.linear3d x m.q_proj
  let kFlat : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.num_key_value_heads * cfg.head_dim] } := Tensor.linear3d x m.k_proj
  let vFlat : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.num_key_value_heads * cfg.head_dim] } := Tensor.linear3d x m.v_proj
  let q : Tensor { dtype := dtype, device := device, shape := #[batch, seq, numHeads, cfg.head_dim] } :=
    Tensor.reshapeChecked qFlat #[batch, seq, numHeads, cfg.head_dim]
  let k : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] } :=
    Tensor.reshapeChecked kFlat #[batch, seq, cfg.num_key_value_heads, cfg.head_dim]
  let v : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] } :=
    Tensor.reshapeChecked vFlat #[batch, seq, cfg.num_key_value_heads, cfg.head_dim]
  -- Per-head Q/K RMSNorm BEFORE RoPE (HF `LagunaAttention`).
  let q := applyHeadNorm m.q_norm q cfg.rms_norm_eps
  let k := applyHeadNorm m.k_norm k cfg.rms_norm_eps
  -- GLM-style rotate-half RoPE; `applyRotaryPartial` passes the tail through
  -- untouched when `rotaryDim < cfg.head_dim` and degenerates to full-width
  -- rotary when `rotaryDim == cfg.head_dim` (sliding layers).
  let q := applyRotaryPartial (rotary_dim := rotaryDim) q cos sin
  let k := applyRotaryPartial (rotary_dim := rotaryDim) k cos sin
  (q, k, v)

/-- Per-head softplus output gate + o_proj:
    `o_proj(softplus(g_proj(xn).float()).to(dtype).unsqueeze(-1) * attn)`. -/
private def gateAndOut {batch seq numHeads : UInt64}
    (cfg : Config)
    (m : LagunaAttention (device := device) (dtype := dtype) cfg numHeads)
    (xn : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] })
    (attn : Tensor { dtype := dtype, device := device, shape := #[batch, seq, numHeads, cfg.head_dim] })
    : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] } :=
  let gateRaw : Tensor { dtype := dtype, device := device, shape := #[batch, seq, numHeads] } := Tensor.linear3d xn m.g_proj
  let gate : Tensor { dtype := dtype, device := device, shape := #[batch, seq, numHeads] } := Tensor.castLike xn (softplusDynamic (floatDynamic gateRaw))
  let gate4 : Tensor { dtype := dtype, device := device, shape := #[batch, seq, numHeads, cfg.head_dim] } :=
    Tensor.expand (Tensor.unsqueeze gate 3) #[batch, seq, numHeads, cfg.head_dim]
  let gated : Tensor { dtype := dtype, device := device, shape := #[batch, seq, numHeads * cfg.head_dim] } :=
    Tensor.reshapeChecked (Tensor.mul attn gate4) #[batch, seq, numHeads * cfg.head_dim]
  Tensor.linear3d gated m.o_proj

/-- Uncached full-sequence forward: implicit causal attention for full layers,
    bounded window attention for sliding layers. -/
def forward {batch seq rotaryDim numHeads : UInt64}
    (cfg : Config)
    (m : LagunaAttention (device := device) (dtype := dtype) cfg numHeads)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] })
    (cos : Tensor { device := device, shape := #[seq, rotaryDim / 2], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[seq, rotaryDim / 2], dtype := .Float32 })
    (window : Option UInt64 := none)
    : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] } :=
  let (q, k, v) := qkvRope cfg m x cos sin
  let qh : Tensor { dtype := dtype, device := device, shape := #[batch, numHeads, seq, cfg.head_dim] } := transposeAttention q
  let kh : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] } := transposeAttention k
  let vh : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] } := transposeAttention v
  let attn : Tensor { dtype := dtype, device := device, shape := #[batch, numHeads, seq, cfg.head_dim] } :=
    match window with
    | some w => Tensor.attentionWindow qh kh vh true true w
    | none => Tensor.attentionGQA qh kh vh true true
  let attn : Tensor { dtype := dtype, device := device, shape := #[batch, seq, numHeads, cfg.head_dim] } := transposeAttention attn
  gateAndOut cfg m x attn

/-- Batch prefill with KV-cache update. Writes K/V at cache offset 0 and sets
    `cache.seq := seq` — call this on a FRESH cache (chunked prefill is not
    supported). Sliding layers attend with the window SDPA over the fresh K/V. -/
def forwardWithCache {batch seq rotaryDim numHeads : UInt64}
    (cfg : Config)
    (m : LagunaAttention (device := device) (dtype := dtype) cfg numHeads)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] })
    (cos : Tensor { device := device, shape := #[seq, rotaryDim / 2], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[seq, rotaryDim / 2], dtype := .Float32 })
    (window : Option UInt64)
    (cache : KVCache (device := device) (dtype := dtype) cfg batch)
    : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] } × KVCache (device := device) (dtype := dtype) cfg batch :=
  let (q, k, v) := qkvRope cfg m x cos sin
  let qh : Tensor { dtype := dtype, device := device, shape := #[batch, numHeads, seq, cfg.head_dim] } := transposeAttention q
  let kh : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] } := transposeAttention k
  let vh : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] } := transposeAttention v
  let kStore : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim] } :=
    Tensor.castLike kh (Tensor.reshapeChecked cache.kStoreDyn.2 #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim])
  let vStore : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim] } :=
    Tensor.castLike vh (Tensor.reshapeChecked cache.vStoreDyn.2 #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim])
  let kStore' := Tensor.sliceScatter kStore 2 0 kh
  let vStore' := Tensor.sliceScatter vStore 2 0 vh
  let cache' : KVCache (device := device) (dtype := dtype) cfg batch := {
    kStoreDyn := kStore'.packShape
    vStoreDyn := vStore'.packShape
    seq := seq
    maxLen := cache.maxLen
  }
  let attn : Tensor { dtype := dtype, device := device, shape := #[batch, numHeads, seq, cfg.head_dim] } :=
    match window with
    | some w => Tensor.attentionWindow qh kh vh true true w
    | none => Tensor.attentionGQA qh kh vh true true
  let attn : Tensor { dtype := dtype, device := device, shape := #[batch, seq, numHeads, cfg.head_dim] } := transposeAttention attn
  (gateAndOut cfg m x attn, cache')

/-- Single-token decode step with KV-cache update. Sliding-window layers
    truncate the cache READ to the last `min(kvLen, window)` entries
    (Gemma4 pattern) — every entry in the truncated span is attendable, so a
    plain non-causal GQA decode is exact. -/
def forwardStep {batch rotaryDim numHeads : UInt64}
    (cfg : Config)
    (m : LagunaAttention (device := device) (dtype := dtype) cfg numHeads)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, 1, cfg.hidden_size] })
    (cos : Tensor { device := device, shape := #[1, rotaryDim / 2], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[1, rotaryDim / 2], dtype := .Float32 })
    (window : Option UInt64)
    (cache : KVCache (device := device) (dtype := dtype) cfg batch)
    : Tensor { dtype := dtype, device := device, shape := #[batch, 1, cfg.hidden_size] } × KVCache (device := device) (dtype := dtype) cfg batch :=
  let (q, k, v) := qkvRope cfg m x cos sin
  let qh : Tensor { dtype := dtype, device := device, shape := #[batch, numHeads, 1, cfg.head_dim] } := transposeAttention q
  let kNew : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.num_key_value_heads, 1, cfg.head_dim] } := transposeAttention k
  let vNew : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.num_key_value_heads, 1, cfg.head_dim] } := transposeAttention v
  let kStore : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim] } :=
    Tensor.castLike kNew (Tensor.reshapeChecked cache.kStoreDyn.2 #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim])
  let vStore : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim] } :=
    Tensor.castLike vNew (Tensor.reshapeChecked cache.vStoreDyn.2 #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim])
  let writePos : UInt64 :=
    if cache.seq < cache.maxLen then cache.seq
    else if cache.maxLen == 0 then 0 else cache.maxLen - 1
  let kStore' : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim] } :=
    Tensor.sliceScatter kStore 2 writePos kNew
  let vStore' : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim] } :=
    Tensor.sliceScatter vStore 2 writePos vNew
  let kvLen : UInt64 := if cache.seq < cache.maxLen then cache.seq + 1 else cache.maxLen
  -- Sliding-window read truncation: keep only the last `useLen` entries.
  let useLen : UInt64 :=
    match window with
    | some w => if kvLen > w then w else kvLen
    | none => kvLen
  let start : UInt64 := kvLen - useLen
  let kAll : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.num_key_value_heads, useLen, cfg.head_dim] } := sliceTyped kStore' 2 start useLen
  let vAll : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.num_key_value_heads, useLen, cfg.head_dim] } := sliceTyped vStore' 2 start useLen
  -- Every retained key is attendable. Rectangular SDPA must be non-causal:
  -- its causal mode is upper-left aligned and would expose only key zero.
  let attn : Tensor { dtype := dtype, device := device, shape := #[batch, numHeads, 1, cfg.head_dim] } :=
    Tensor.attentionGQAQKV qh kAll vAll false true
  let attn : Tensor { dtype := dtype, device := device, shape := #[batch, 1, numHeads, cfg.head_dim] } := transposeAttention attn
  let cache' : KVCache (device := device) (dtype := dtype) cfg batch := {
    kStoreDyn := kStore'.packShape
    vStoreDyn := vStore'.packShape
    seq := kvLen
    maxLen := cache.maxLen
  }
  (gateAndOut cfg m x attn, cache')

/-- Internal effectful path. Cache tensors never escape the owning session.
Sliding stores use absolute-position modulo capacity; paired K/V order is
irrelevant to single-query attention after RoPE has been applied. -/
private def forwardOwned {batch seq rotaryDim numHeads : UInt64}
    (cfg : Config) (m : LagunaAttention (device := device) (dtype := dtype) cfg numHeads)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] })
    (cos : Tensor { device := device, shape := #[seq, rotaryDim / 2], dtype := .Float32 }) (sin : Tensor { device := device, shape := #[seq, rotaryDim / 2], dtype := .Float32 })
    (window : Option UInt64) (prefill : Bool) (cache : KVCache (device := device) (dtype := dtype) cfg batch)
    : IO (Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] } × KVCache (device := device) (dtype := dtype) cfg batch) := do
  let (q, k, v) := qkvRope cfg m x cos sin
  let qh : Tensor { dtype := dtype, device := device, shape := #[batch, numHeads, seq, cfg.head_dim] } := transposeAttention q
  let kh : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] } := transposeAttention k
  let vh : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] } := transposeAttention v
  let mut kStore : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim] } :=
    Tensor.castLike kh (Tensor.reshapeChecked cache.kStoreDyn.2 #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim])
  let mut vStore : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim] } :=
    Tensor.castLike vh (Tensor.reshapeChecked cache.vStoreDyn.2 #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim])
  if prefill then
    -- Retain only the prompt tail, placing each entry at its absolute ring slot.
    let retained := min seq cache.maxLen
    let sourceStart := seq - retained
    let slot := sourceStart % cache.maxLen
    let first := min retained (cache.maxLen - slot)
    kStore ← data.copySliceIO kStore 2 slot (sliceTyped kh 2 sourceStart first)
    vStore ← data.copySliceIO vStore 2 slot (sliceTyped vh 2 sourceStart first)
    if first < retained then
      kStore ← data.copySliceIO kStore 2 0 (sliceTyped kh 2 (sourceStart + first) (retained - first))
      vStore ← data.copySliceIO vStore 2 0 (sliceTyped vh 2 (sourceStart + first) (retained - first))
  else
    let slot := cache.seq % cache.maxLen
    kStore ← data.copySliceIO kStore 2 slot kh
    vStore ← data.copySliceIO vStore 2 slot vh
  let total := if prefill then seq else cache.seq + seq
  let attn : Tensor { dtype := dtype, device := device, shape := #[batch, numHeads, seq, cfg.head_dim] } :=
    if prefill then
      match window with
      | some w => Tensor.attentionWindow qh kh vh true true w
      | none => Tensor.attentionGQA qh kh vh true true
    else
      let useLen := min total cache.maxLen
      let kAll := sliceTyped kStore 2 0 useLen
      let vAll := sliceTyped vStore 2 0 useLen
      Tensor.attentionGQAQKV qh kAll vAll false true
  let out := gateAndOut cfg m x (transposeAttention attn)
  return (out, {
    kStoreDyn := kStore.packShape
    vStoreDyn := vStore.packShape
    seq := total
    maxLen := cache.maxLen })

end LagunaAttention

/-! ## KV cache -/

/-- Full-model KV cache: one static KV cache per decoder layer (sliding
    layers keep a full-length store and truncate reads at decode time). -/
structure LagunaCache (cfg : Config) (batch : UInt64) (dtype : DType := .Float32) where
  kvCaches : Array (LagunaAttention.KVCache (device := device) (dtype := dtype) cfg batch)

/-- An inference cache with private, mutable storage. Aliases refer to the same
serialized session; `fork` creates independent storage. Failed transitions
invalidate the session. Use the functional `LagunaCache` API for training or
persistent tensor snapshots. Sliding layers allocate only one window. -/
structure LagunaCacheSession (cfg : Config) (batch : UInt64) (dtype : DType := .Float32) where
  private mk ::
  private state : Std.Mutex (Option (LagunaCache (device := device) (dtype := dtype) cfg batch))
  private maxLen : UInt64

namespace LagunaCacheSession

/-- Copy a session at its current position, including independent K/V buffers. -/
def fork (session : LagunaCacheSession (device := device) (dtype := dtype) cfg batch) : IO (LagunaCacheSession (device := device) (dtype := dtype) cfg batch) :=
  session.state.atomically do
    let some cache ← get | throw <| IO.userError "Laguna cache session is invalid"
    let copied ← cache.kvCaches.mapM fun kv => do
      let k ← data.cloneInferenceIO kv.kStoreDyn.2
      let v ← data.cloneInferenceIO kv.vStoreDyn.2
      return { kv with kStoreDyn := k.packShape, vStoreDyn := v.packShape }
    return ⟨← Std.Mutex.new (some { kvCaches := copied }), session.maxLen⟩

/-- Layer capacities, for memory accounting without exposing mutable tensors. -/
def capacities (session : LagunaCacheSession (device := device) (dtype := dtype) cfg batch) : IO (Array UInt64) :=
  session.state.atomically do
    let some cache ← get | throw <| IO.userError "Laguna cache session is invalid"
    return cache.kvCaches.map (·.maxLen)

end LagunaCacheSession

/-! ## Decoder layer -/

/-- One Laguna decoder layer: RMSNorm → attention → residual → RMSNorm →
    FFN (dense SwiGLU or sparse MoE) → residual.  Exactly one attention
    variant (`attnFull`/`attnSliding`) and one FFN variant
    (`denseMlp`/`sparseMoe`) is set per layer. -/
structure LagunaLayer (cfg : Config) (dtype : DType := .Float32) where
  input_layernorm : SomeDTypeTensor #[cfg.hidden_size] device
  attnFull : Option (LagunaAttention (device := device) (dtype := dtype) cfg cfg.num_attention_heads) := none
  attnSliding : Option (LagunaAttention (device := device) (dtype := dtype) cfg cfg.num_attention_heads_sliding) := none
  post_attention_layernorm : SomeDTypeTensor #[cfg.hidden_size] device
  denseMlp : Option (LagunaMLP (device := device) (dtype := dtype) cfg) := none
  sparseMoe : Option (LagunaSparseMoeBlock (device := device) (dtype := dtype) cfg) := none
  deriving TensorStruct


namespace LagunaLayer

/-- Random-init NVFP4 expert bank stand-in for fresh (non-checkpoint) models.
    `packed` banks are int64 tensors with values in `[0, 256)` standing in for
    U8 byte pairs (dequant only applies `data.toLong`), and `scale` banks are
    F32 (dequant's `floatDynamic` is a no-op). Checkpoint loading replaces these
    with real U8 / F8_E4M3 tensors of the same runtime shapes. -/
private def initPackedExperts (cfg : Config) (device : Device := .CPU) : IO (LagunaPackedExperts (device := device) cfg) := do
  let e := cfg.num_experts
  let mi := cfg.moe_intermediate_size
  let h := cfg.hidden_size
  let scaleVal := 0.05
  let gatePacked ← Tensor.randint 0 256 #[e, mi, h / 2] (device := device)
  let upPacked ← Tensor.randint 0 256 #[e, mi, h / 2] (device := device)
  let downPacked ← Tensor.randint 0 256 #[e, h, mi / 2] (device := device)
  let gateScale := Tensor.mulScalar (← randDynamic (device := device) #[e, mi, h / 16]) scaleVal
  let upScale := Tensor.mulScalar (← randDynamic (device := device) #[e, mi, h / 16]) scaleVal
  let downScale := Tensor.mulScalar (← randDynamic (device := device) #[e, h, mi / 16]) scaleVal
  pure {
    gatePacked := gatePacked.packDType (by simp)
    gateScale := gateScale.packDType (by simp)
    gateGlobal := (onesDynamic (device := device) #[e]).packDType (by simp)
    upPacked := upPacked.packDType (by simp)
    upScale := upScale.packDType (by simp)
    upGlobal := (onesDynamic (device := device) #[e]).packDType (by simp)
    downPacked := downPacked.packDType (by simp)
    downScale := downScale.packDType (by simp)
    downGlobal := (onesDynamic (device := device) #[e]).packDType (by simp)
  }

/-- Random-init sparse MoE block (random router + packed-expert stand-ins +
    random shared expert). For fresh models/tests; checkpoint loading
    constructs `LagunaSparseMoeBlock` directly from real tensors. -/
def initSparseMoeBlock (cfg : Config) (device : Device := .CPU) (dtype : DType := .Float32) : IO (LagunaSparseMoeBlock (device := device) (dtype := dtype) cfg) := do
  let routerW ← initDynamicWeight (device := device) #[cfg.num_experts, cfg.hidden_size] cfg.hidden_size
  let experts ← initPackedExperts (device := device) cfg
  let sharedGate ← initDynamicWeight (device := device) (dtype := dtype) #[cfg.shared_expert_intermediate_size, cfg.hidden_size] cfg.hidden_size
  let sharedUp ← initDynamicWeight (device := device) (dtype := dtype) #[cfg.shared_expert_intermediate_size, cfg.hidden_size] cfg.hidden_size
  let sharedDown ← initDynamicWeight (device := device) (dtype := dtype) #[cfg.hidden_size, cfg.shared_expert_intermediate_size]
    cfg.shared_expert_intermediate_size
  pure {
    router := { weight := routerW.packDType (by simp), eScoreCorrectionBias := none }
    experts := experts
    sharedGateProj := sharedGate
    sharedUpProj := sharedUp
    sharedDownProj := sharedDown
  }

def init (cfg : Config) (layerIdx : UInt64) (device : Device := .CPU) (dtype : DType := .Float32) : IO (LagunaLayer (device := device) (dtype := dtype) cfg) := do
  let inputNorm ← Tensor.parameter (onesDynamic (device := device) #[cfg.hidden_size])
  let postNorm ← Tensor.parameter (onesDynamic (device := device) #[cfg.hidden_size])
  let attnFull ←
    match cfg.layerType layerIdx with
    | .fullAttention =>
      let m ← LagunaAttention.init (device := device) (dtype := dtype) cfg cfg.num_attention_heads
      pure (some m)
    | .slidingAttention => pure none
  let attnSliding ←
    match cfg.layerType layerIdx with
    | .slidingAttention =>
      let m ← LagunaAttention.init (device := device) (dtype := dtype) cfg cfg.num_attention_heads_sliding
      pure (some m)
    | .fullAttention => pure none
  let denseMlp ←
    if cfg.isDenseMlpLayer layerIdx || !cfg.isMoE then
      let m ← LagunaMLP.init (device := device) (dtype := dtype) cfg
      pure (some m)
    else
      pure none
  let sparseMoe ←
    if cfg.isDenseMlpLayer layerIdx || !cfg.isMoE then
      pure none
    else
      let m ← initSparseMoeBlock (device := device) (dtype := dtype) cfg
      pure (some m)
  pure {
    input_layernorm := inputNorm.packDType (by simp)
    attnFull := attnFull
    attnSliding := attnSliding
    post_attention_layernorm := postNorm.packDType (by simp)
    denseMlp := denseMlp
    sparseMoe := sparseMoe
  }

/-- FFN dispatch shared by all three layer entry points. -/
private def ffnForward {batch seq : UInt64}
    (cfg : Config)
    (layer : LagunaLayer (device := device) (dtype := dtype) cfg)
    (h : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] })
    : IO (Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] }) := do
  match layer.denseMlp, layer.sparseMoe with
  | some mlp, _ => pure (mlp.forward h)
  | _, some moe => moe.forward3d cfg h
  | _, _ => pure h

/-- Uncached full-sequence layer forward. -/
def forward {batch seq : UInt64}
    (cfg : Config)
    (layer : LagunaLayer (device := device) (dtype := dtype) cfg)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] })
    (tables : (LagunaRotaryTables (device := device)))
    : IO (Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] }) := do
  let h1 := rmsNormWeighted3d layer.input_layernorm x cfg.rms_norm_eps
  let mixed ←
    match layer.attnFull, layer.attnSliding with
    | some a, _ =>
      let cos : Tensor { device := device, shape := #[seq, cfg.rotaryDimFull / 2], dtype := .Float32 } := sliceRotaryRows tables.fullCos 0 seq
      let sin : Tensor { device := device, shape := #[seq, cfg.rotaryDimFull / 2], dtype := .Float32 } := sliceRotaryRows tables.fullSin 0 seq
      pure (a.forward (rotaryDim := cfg.rotaryDimFull) cfg h1 cos sin none)
    | _, some a =>
      let cos : Tensor { device := device, shape := #[seq, cfg.rotaryDimSliding / 2], dtype := .Float32 } := sliceRotaryRows tables.slidingCos 0 seq
      let sin : Tensor { device := device, shape := #[seq, cfg.rotaryDimSliding / 2], dtype := .Float32 } := sliceRotaryRows tables.slidingSin 0 seq
      pure (a.forward (rotaryDim := cfg.rotaryDimSliding) cfg h1 cos sin (some cfg.sliding_window))
    | _, _ => pure h1
  let h2 := Tensor.add x mixed
  let h3 := rmsNormWeighted3d layer.post_attention_layernorm h2 cfg.rms_norm_eps
  let ffn ← ffnForward cfg layer h3
  pure (Tensor.add h2 ffn)

/-- Batch prefill with KV-cache update (fresh cache; see
    `LagunaAttention.forwardWithCache`). -/
def forwardWithCache {batch seq : UInt64}
    (cfg : Config)
    (layer : LagunaLayer (device := device) (dtype := dtype) cfg)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] })
    (tables : (LagunaRotaryTables (device := device)))
    (cache : LagunaCache (device := device) (dtype := dtype) cfg batch)
    (layerIdx : Nat)
    : IO (Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] } × LagunaCache (device := device) (dtype := dtype) cfg batch) := do
  let h1 := rmsNormWeighted3d layer.input_layernorm x cfg.rms_norm_eps
  let (mixed, cache') ←
    match layer.attnFull, layer.attnSliding, cache.kvCaches[layerIdx]? with
    | some a, _, some kv =>
      let cos : Tensor { device := device, shape := #[seq, cfg.rotaryDimFull / 2], dtype := .Float32 } := sliceRotaryRows tables.fullCos 0 seq
      let sin : Tensor { device := device, shape := #[seq, cfg.rotaryDimFull / 2], dtype := .Float32 } := sliceRotaryRows tables.fullSin 0 seq
      let (out, kv') := a.forwardWithCache (rotaryDim := cfg.rotaryDimFull) cfg h1 cos sin none kv
      pure (out, { cache with kvCaches := cache.kvCaches.set! layerIdx kv' })
    | _, some a, some kv =>
      let cos : Tensor { device := device, shape := #[seq, cfg.rotaryDimSliding / 2], dtype := .Float32 } := sliceRotaryRows tables.slidingCos 0 seq
      let sin : Tensor { device := device, shape := #[seq, cfg.rotaryDimSliding / 2], dtype := .Float32 } := sliceRotaryRows tables.slidingSin 0 seq
      let (out, kv') :=
        a.forwardWithCache (rotaryDim := cfg.rotaryDimSliding) cfg h1 cos sin (some cfg.sliding_window) kv
      pure (out, { cache with kvCaches := cache.kvCaches.set! layerIdx kv' })
    | _, _, _ => pure (h1, cache)
  let h2 := Tensor.add x mixed
  let h3 := rmsNormWeighted3d layer.post_attention_layernorm h2 cfg.rms_norm_eps
  let ffn ← ffnForward cfg layer h3
  pure (Tensor.add h2 ffn, cache')

/-- Single-token decode step with KV-cache update. -/
def forwardStep {batch : UInt64}
    (cfg : Config)
    (layer : LagunaLayer (device := device) (dtype := dtype) cfg)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, 1, cfg.hidden_size] })
    (tables : (LagunaRotaryTables (device := device)))
    (position : UInt64)
    (cache : LagunaCache (device := device) (dtype := dtype) cfg batch)
    (layerIdx : Nat)
    : IO (Tensor { dtype := dtype, device := device, shape := #[batch, 1, cfg.hidden_size] } × LagunaCache (device := device) (dtype := dtype) cfg batch) := do
  let h1 := rmsNormWeighted3d layer.input_layernorm x cfg.rms_norm_eps
  let (mixed, cache') ←
    match layer.attnFull, layer.attnSliding, cache.kvCaches[layerIdx]? with
    | some a, _, some kv =>
      let cos : Tensor { device := device, shape := #[1, cfg.rotaryDimFull / 2], dtype := .Float32 } := sliceRotaryRows tables.fullCos position 1
      let sin : Tensor { device := device, shape := #[1, cfg.rotaryDimFull / 2], dtype := .Float32 } := sliceRotaryRows tables.fullSin position 1
      let (out, kv') := a.forwardStep (rotaryDim := cfg.rotaryDimFull) cfg h1 cos sin none kv
      pure (out, { cache with kvCaches := cache.kvCaches.set! layerIdx kv' })
    | _, some a, some kv =>
      let cos : Tensor { device := device, shape := #[1, cfg.rotaryDimSliding / 2], dtype := .Float32 } := sliceRotaryRows tables.slidingCos position 1
      let sin : Tensor { device := device, shape := #[1, cfg.rotaryDimSliding / 2], dtype := .Float32 } := sliceRotaryRows tables.slidingSin position 1
      let (out, kv') :=
        a.forwardStep (rotaryDim := cfg.rotaryDimSliding) cfg h1 cos sin (some cfg.sliding_window) kv
      pure (out, { cache with kvCaches := cache.kvCaches.set! layerIdx kv' })
    | _, _, _ => pure (h1, cache)
  let h2 := Tensor.add x mixed
  let h3 := rmsNormWeighted3d layer.post_attention_layernorm h2 cfg.rms_norm_eps
  let ffn ← ffnForward cfg layer h3
  pure (Tensor.add h2 ffn, cache')

private def forwardOwned {batch seq : UInt64}
    (cfg : Config) (layer : LagunaLayer (device := device) (dtype := dtype) cfg)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] }) (tables : (LagunaRotaryTables (device := device)))
    (position : UInt64) (prefill : Bool) (kv : LagunaAttention.KVCache (device := device) (dtype := dtype) cfg batch)
    : IO (Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] } × LagunaAttention.KVCache (device := device) (dtype := dtype) cfg batch) := do
  let h1 := rmsNormWeighted3d layer.input_layernorm x cfg.rms_norm_eps
  let (mixed, kv') ← match layer.attnFull, layer.attnSliding with
    | some a, _ =>
      let cos : Tensor { device := device, shape := #[seq, cfg.rotaryDimFull / 2], dtype := .Float32 } := sliceRotaryRows tables.fullCos position seq
      let sin : Tensor { device := device, shape := #[seq, cfg.rotaryDimFull / 2], dtype := .Float32 } := sliceRotaryRows tables.fullSin position seq
      a.forwardOwned (rotaryDim := cfg.rotaryDimFull) cfg h1 cos sin none prefill kv
    | _, some a =>
      let cos : Tensor { device := device, shape := #[seq, cfg.rotaryDimSliding / 2], dtype := .Float32 } := sliceRotaryRows tables.slidingCos position seq
      let sin : Tensor { device := device, shape := #[seq, cfg.rotaryDimSliding / 2], dtype := .Float32 } := sliceRotaryRows tables.slidingSin position seq
      a.forwardOwned (rotaryDim := cfg.rotaryDimSliding) cfg h1 cos sin
        (some cfg.sliding_window) prefill kv
    | _, _ => throw <| IO.userError "Laguna cache session requires attention in every layer"
  let h2 := Tensor.add x mixed
  let h3 := rmsNormWeighted3d layer.post_attention_layernorm h2 cfg.rms_norm_eps
  let ffn ← ffnForward cfg layer h3
  return (Tensor.add h2 ffn, kv')

end LagunaLayer

/-! ## Model -/

/-- Laguna base model: token embedding → `num_hidden_layers` decoder layers →
    final RMSNorm. -/
structure LagunaModel (cfg : Config) (dtype : DType := .Float32) where
  embed_tokens : Tensor { dtype := dtype, device := device, shape := #[cfg.vocab_size, cfg.hidden_size] }
  layers : Array (LagunaLayer (device := device) (dtype := dtype) cfg)
  norm : SomeDTypeTensor #[cfg.hidden_size] device
  deriving TensorStruct


namespace LagunaModel

def init (cfg : Config) (device : Device := .CPU) (dtype : DType := .Float32) : IO (LagunaModel (device := device) (dtype := dtype) cfg) := do
  let floating ← IO.ofExcept (DType.realFloating? dtype)
  let _ := floating.down
  let embRaw ← randnDynamic (device := device) (dtype := dtype) #[cfg.vocab_size, cfg.hidden_size]
  let embedTokens ← Tensor.parameter (Tensor.mulScalar embRaw 0.02)
  let mut layers : Array (LagunaLayer (device := device) (dtype := dtype) cfg) := Array.mkEmpty cfg.num_hidden_layers.toNat
  for i in [:cfg.num_hidden_layers.toNat] do
    layers := layers.push (← LagunaLayer.init (device := device) (dtype := dtype) cfg i.toUInt64)
  let norm ← Tensor.parameter (onesDynamic (device := device) #[cfg.hidden_size])
  pure { embed_tokens := embedTokens, layers := layers, norm := norm.packDType (by simp) }

/-- Allocate a fresh per-layer KV cache with capacity `maxLen` on `device`. -/
def initCache {batch : UInt64}
    (cfg : Config)
    (m : LagunaModel (device := device) (dtype := dtype) cfg)
    (maxLen : UInt64)
    (device : Device)
    [DType.ConstantSupported dtype device]
    : LagunaCache (device := device) (dtype := dtype) cfg batch :=
  Id.run do
    let mut caches : Array (LagunaAttention.KVCache (device := device) (dtype := dtype) cfg batch) := Array.mkEmpty m.layers.size
    for _ in [:m.layers.size] do
      caches := caches.push (qwen.QwenAttention.initKVCache
        maxLen
        (batch := batch)
        (num_kv_heads := cfg.num_key_value_heads)
        (head_dim := cfg.head_dim)
        device dtype)
    return { kvCaches := caches }

/-- Uncached full-sequence forward (token ids → final hidden states).
    RoPE tables are precomputed for `[0, seq)` on the fly. -/
def forward {batch seq : UInt64}
    (cfg : Config)
    (m : LagunaModel (device := device) (dtype := dtype) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    : IO (Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] }) := do
  let x0 : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] } := Tensor.embedding inputIds m.embed_tokens
  let tables ← precomputeRotaryTables cfg seq x0.device
  let mut h := x0
  for layer in m.layers do
    h ← layer.forward cfg h tables
  pure (rmsNormWeighted3d m.norm h cfg.rms_norm_eps)

/-- Batch prefill over embeddings with KV-cache update (fresh cache).
    `tables` must cover positions `[0, seq)`. -/
def forwardWithCache {batch seq : UInt64}
    (cfg : Config)
    (m : LagunaModel (device := device) (dtype := dtype) cfg)
    (x0 : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] })
    (tables : (LagunaRotaryTables (device := device)))
    (cache : LagunaCache (device := device) (dtype := dtype) cfg batch)
    : IO (Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] } × LagunaCache (device := device) (dtype := dtype) cfg batch) := do
  let mut h := x0
  let mut c := cache
  for i in [:m.layers.size] do
    let layer ←
      match m.layers[i]? with
      | some l => pure l
      | none => throw <| IO.userError s!"missing Laguna layer at index {i}"
    let (hNext, cNext) ← layer.forwardWithCache cfg h tables c i
    h := hNext
    c := cNext
  return (rmsNormWeighted3d m.norm h cfg.rms_norm_eps, c)

/-- Single-token decode step over one token embedding at absolute `position`. -/
def forwardStep {batch : UInt64}
    (cfg : Config)
    (m : LagunaModel (device := device) (dtype := dtype) cfg)
    (tokenEmbed : Tensor { dtype := dtype, device := device, shape := #[batch, 1, cfg.hidden_size] })
    (tables : (LagunaRotaryTables (device := device)))
    (position : UInt64)
    (cache : LagunaCache (device := device) (dtype := dtype) cfg batch)
    : IO (Tensor { dtype := dtype, device := device, shape := #[batch, 1, cfg.hidden_size] } × LagunaCache (device := device) (dtype := dtype) cfg batch) := do
  let mut h := tokenEmbed
  let mut c := cache
  for i in [:m.layers.size] do
    let layer ←
      match m.layers[i]? with
      | some l => pure l
      | none => throw <| IO.userError s!"missing Laguna layer at index {i}"
    let (hNext, cNext) ← layer.forwardStep cfg h tables position c i
    h := hNext
    c := cNext
  return (rmsNormWeighted3d m.norm h cfg.rms_norm_eps, c)

/-- Allocate an owned inference session. Full layers reserve `maxLen`; sliding
layers reserve `min maxLen sliding_window`. No cache tensor is publicly exposed. -/
def initCacheSession {batch : UInt64} (cfg : Config) (m : LagunaModel (device := device) (dtype := dtype) cfg)
    (maxLen : UInt64) (device : Device) : IO (LagunaCacheSession (device := device) (dtype := dtype) cfg batch) := do
  let support ← dtype.acquireConstantSupport device
  let _ := support.down
  if maxLen == 0 || cfg.sliding_window == 0 then
    throw <| IO.userError "Laguna cache capacity and sliding window must be positive"
  if m.layers.isEmpty then
    throw <| IO.userError "Laguna cache session requires at least one layer"
  let mut caches := #[]
  for layer in m.layers do
    let capacity := if layer.attnFull.isSome then maxLen else min maxLen cfg.sliding_window
    -- Pure zero/clone calls can be commoned by Lean. Allocate both mutable
    -- buffers through IO, even when they have identical shape and contents.
    let template := qwen.QwenAttention.initKVCache capacity
      (batch := batch) (num_kv_heads := cfg.num_key_value_heads) (head_dim := cfg.head_dim) device dtype
    let k ← data.cloneInferenceIO template.kStoreDyn.2
    let v ← data.cloneInferenceIO template.vStoreDyn.2
    caches := caches.push { template with kStoreDyn := k.packShape, vStoreDyn := v.packShape }
  return ⟨← Std.Mutex.new (some { kvCaches := caches }), maxLen⟩

/-- Execute one serialized inference transition, invalidating on partial failure.
Only fresh prefill and subsequent single-token steps are supported. -/
private def forwardSession {batch seq : UInt64} (cfg : Config) (m : LagunaModel (device := device) (dtype := dtype) cfg)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] }) (tables : (LagunaRotaryTables (device := device)))
    (position : UInt64) (prefill : Bool) (session : LagunaCacheSession (device := device) (dtype := dtype) cfg batch)
    : IO (Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] }) := session.state.atomically do
  let some cache ← get | throw <| IO.userError "Laguna cache session is invalid"
  if seq == 0 || position > session.maxLen || seq > session.maxLen - position then
    throw <| IO.userError "Laguna cache transition exceeds capacity or has an empty input"
  if x.runtimeShape != #[batch, seq, cfg.hidden_size] then
    throw <| IO.userError "Laguna cache input shape mismatch"
  for (table, dim) in #[(tables.fullCos, cfg.rotaryDimFull / 2),
      (tables.fullSin, cfg.rotaryDimFull / 2), (tables.slidingCos, cfg.rotaryDimSliding / 2),
      (tables.slidingSin, cfg.rotaryDimSliding / 2)] do
    let shape := table.1
    if shape.size != 2 || shape.getD 0 0 < position + seq || shape.getD 1 0 != dim ||
        table.2.device != x.device then
      throw <| IO.userError "Laguna cache rotary table shape, coverage or device mismatch"
  if cache.kvCaches.size != m.layers.size then
    throw <| IO.userError "Laguna cache layer count mismatch"
  if prefill && position != 0 || !prefill && seq != 1 then
    throw <| IO.userError "Laguna cache expects fresh prefill or single-token decode"
  for kv in cache.kvCaches do
    if kv.seq != position || (!prefill && position == 0) then
      throw <| IO.userError "Laguna cache position mismatch; prefill once before decode"
  -- Validation failures above leave the session usable. Any failure below may
  -- have mutated a subset of layers, so no partially updated state is reusable.
  set (none : Option (LagunaCache (device := device) (dtype := dtype) cfg batch))
  let (hidden, updated) ← do
    let m := TensorStruct.detach m
    let tables := tables.detach
    let mut h := x.detach
    let mut caches := cache.kvCaches
    for i in [:m.layers.size] do
      let some layer := m.layers[i]? | throw <| IO.userError "missing Laguna layer"
      let some cached := caches[i]? | throw <| IO.userError "missing Laguna cache"
      let (next, kv) ← layer.forwardOwned cfg h tables position prefill cached
      h := next
      caches := caches.set! i kv
    pure (rmsNormWeighted3d m.norm h cfg.rms_norm_eps, { kvCaches := caches : LagunaCache (device := device) (dtype := dtype) cfg batch })
  set (some updated)
  return hidden

end LagunaModel

/-! ## ForCausalLM + generation -/

/-- Full Laguna causal language model (untied `lm_head` by default). -/
structure LagunaForCausalLM (cfg : Config) (dtype : DType := .Float32) where
  model : LagunaModel (device := device) (dtype := dtype) cfg
  lmHead : Tensor { dtype := dtype, device := device, shape := #[cfg.vocab_size, cfg.hidden_size] }
  tieWordEmbeddings : Bool := false
  deriving TensorStruct


namespace LagunaForCausalLM

def init (cfg : Config) (tieWordEmbeddings : Bool := false) (device : Device := .CPU) (dtype : DType := .Float32) : IO (LagunaForCausalLM (device := device) (dtype := dtype) cfg) := do
  let model ← LagunaModel.init (device := device) (dtype := dtype) cfg
  let lmHead ←
    if tieWordEmbeddings then
      pure model.embed_tokens
    else
      initDynamicWeight (device := device) (dtype := dtype) #[cfg.vocab_size, cfg.hidden_size] cfg.hidden_size
  pure { model := model, lmHead := lmHead, tieWordEmbeddings := tieWordEmbeddings }

def embedTokens {batch seq : UInt64}
    (m : LagunaForCausalLM (device := device) (dtype := dtype) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] } :=
  Tensor.embedding inputIds m.model.embed_tokens

/-- Uncached full-sequence forward (token ids → logits). -/
def forward {batch seq : UInt64}
    (cfg : Config)
    (m : LagunaForCausalLM (device := device) (dtype := dtype) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    : IO (Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.vocab_size] }) := do
  let hidden ← m.model.forward cfg inputIds
  pure (Tensor.linear3d hidden m.lmHead)

/-- Single-pass prompt prefill; returns last-position logits + updated cache. -/
def prefill {batch seq : UInt64}
    (cfg : Config)
    (m : LagunaForCausalLM (device := device) (dtype := dtype) cfg)
    (tables : (LagunaRotaryTables (device := device)))
    (inputsEmbeds : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] })
    (cache : LagunaCache (device := device) (dtype := dtype) cfg batch)
    : IO (Tensor { dtype := dtype, device := device, shape := #[batch, cfg.vocab_size] } × LagunaCache (device := device) (dtype := dtype) cfg batch) := do
  let (hidden, cache') ← LagunaModel.forwardWithCache cfg m.model inputsEmbeds tables cache
  let lastHidden : Tensor { dtype := dtype, device := device, shape := #[batch, 1, cfg.hidden_size] } := sliceTyped hidden 1 (seq - 1) 1
  let logits3 : Tensor { dtype := dtype, device := device, shape := #[batch, 1, cfg.vocab_size] } := Tensor.linear3d lastHidden m.lmHead
  pure (Tensor.reshapeChecked logits3 #[batch, cfg.vocab_size], cache')

/-- Single decode step from one token embedding at absolute `position`. -/
def decodeStep {batch : UInt64}
    (cfg : Config)
    (m : LagunaForCausalLM (device := device) (dtype := dtype) cfg)
    (tables : (LagunaRotaryTables (device := device)))
    (tokenEmbed : Tensor { dtype := dtype, device := device, shape := #[batch, 1, cfg.hidden_size] })
    (position : UInt64)
    (cache : LagunaCache (device := device) (dtype := dtype) cfg batch)
    : IO (Tensor { dtype := dtype, device := device, shape := #[batch, cfg.vocab_size] } × LagunaCache (device := device) (dtype := dtype) cfg batch) := do
  let (hidden, cache') ← m.model.forwardStep cfg tokenEmbed tables position cache
  let logits3 : Tensor { dtype := dtype, device := device, shape := #[batch, 1, cfg.vocab_size] } := Tensor.linear3d hidden m.lmHead
  pure (Tensor.reshapeChecked logits3 #[batch, cfg.vocab_size], cache')

/-- Inference-only prefill into an owned cache session. The session must be fresh. -/
def prefillSession {batch seq : UInt64} (cfg : Config) (m : LagunaForCausalLM (device := device) (dtype := dtype) cfg)
    (tables : (LagunaRotaryTables (device := device))) (inputsEmbeds : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.hidden_size] })
    (session : LagunaCacheSession (device := device) (dtype := dtype) cfg batch) : IO (Tensor { dtype := dtype, device := device, shape := #[batch, cfg.vocab_size] }) := do
  let hidden ← m.model.forwardSession cfg inputsEmbeds tables 0 true session
  let lastHidden : Tensor { dtype := dtype, device := device, shape := #[batch, 1, cfg.hidden_size] } := sliceTyped hidden 1 (seq - 1) 1
  return Tensor.reshapeChecked (Tensor.linear3d lastHidden m.lmHead.detach) #[batch, cfg.vocab_size]

/-- Append one token at the next absolute position without copying cache buffers. -/
def decodeSession {batch : UInt64} (cfg : Config) (m : LagunaForCausalLM (device := device) (dtype := dtype) cfg)
    (tables : (LagunaRotaryTables (device := device))) (tokenEmbed : Tensor { dtype := dtype, device := device, shape := #[batch, 1, cfg.hidden_size] })
    (position : UInt64) (session : LagunaCacheSession (device := device) (dtype := dtype) cfg batch)
    : IO (Tensor { dtype := dtype, device := device, shape := #[batch, cfg.vocab_size] }) := do
  let hidden ← m.model.forwardSession cfg tokenEmbed tables position false session
  return Tensor.reshapeChecked (Tensor.linear3d hidden m.lmHead.detach) #[batch, cfg.vocab_size]

private partial def decodeLoopCached {batch : UInt64}
    (cfg : Config)
    (m : LagunaForCausalLM (device := device) (dtype := dtype) cfg)
    (tables : (LagunaRotaryTables (device := device)))
    (strategy : SamplingStrategy)
    (eosTokenIds : Array UInt64)
    (finished : Tensor { device := device, shape := #[batch], dtype := .Bool })
    (remaining : Nat)
    (cache : LagunaCacheSession (device := device) (dtype := dtype) cfg batch)
    (lastLogits : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.vocab_size] })
    (onStep : Option (StreamCallback (device := device) batch))
    (generatedSoFar : UInt64)
    {curSeq : UInt64}
    (curIds : Tensor { device := device, shape := #[batch, curSeq], dtype := .Int64 })
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  if remaining == 0 then
    return ⟨curSeq, curIds⟩

  let nextTokRaw ← sampleFromLogits lastLogits.toFloat32 strategy
  let finished' : Tensor { device := device, shape := #[batch], dtype := .Bool } :=
    if eosTokenIds.isEmpty then
      finished
    else
      Tensor.logicalOr finished (tokenInSet nextTokRaw eosTokenIds)
  let nextTok : Tensor { device := device, shape := #[batch], dtype := .Int64 } :=
    if eosTokenIds.isEmpty then
      nextTokRaw
    else
      applyFinishedEos nextTokRaw finished (eosTokenIds.getD 0 0)

  match onStep with
  | some cb => cb generatedSoFar nextTok
  | none => pure ()

  let nextCol : Tensor { device := device, shape := #[batch, 1], dtype := .Int64 } := Tensor.reshapeChecked nextTok #[batch, 1]
  let appended : Tensor { device := device, shape := #[batch, curSeq + 1], dtype := .Int64 } := Tensor.catManyChecked #[curIds.packShape, nextCol.packShape] 1 #[batch, curSeq + 1]

  let stop :=
    if eosTokenIds.isEmpty then
      false
    else
      !(Tensor.any (Tensor.logicalNot finished'))
  if stop || remaining == 1 then
    return ⟨curSeq + 1, appended⟩
  else
    let nextEmb : Tensor { dtype := dtype, device := device, shape := #[batch, 1, cfg.hidden_size] } := m.embedTokens nextCol
    let nextLogits ← decodeSession cfg m tables nextEmb curSeq cache
    decodeLoopCached cfg m tables strategy eosTokenIds finished' (remaining - 1) cache
      nextLogits onStep (generatedSoFar + 1) appended

/-- Exact generation entry point using KV-caching. -/
private def generateCore {batch seq : UInt64}
    (cfg : Config)
    (m : LagunaForCausalLM (device := device) (dtype := dtype) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (maxNewTokens : UInt64)
    (strategy : SamplingStrategy)
    (eosTokenIds : Array UInt64)
    (onStep : Option (StreamCallback (device := device) batch))
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  let m := TensorStruct.detach m
  if seq == 0 then
    throw <| IO.userError "generate requires non-empty prompt sequence"
  if maxNewTokens == 0 then
    return ⟨seq, inputIds⟩
  let inputsEmbeds := m.embedTokens inputIds
  if maxNewTokens > (0xffffffffffffffff : UInt64) - seq then
    throw <| IO.userError "Laguna generation length overflow"
  let maxLen := seq + maxNewTokens
  let tables ← precomputeRotaryTables cfg maxLen inputsEmbeds.device
  let cache ← LagunaModel.initCacheSession cfg m.model maxLen inputsEmbeds.device
  let logits ← prefillSession cfg m tables inputsEmbeds cache
  let finished0 : Tensor { device := device, shape := #[batch], dtype := .Bool } := falseMaskTyped (n := batch) inputIds.device
  decodeLoopCached cfg m tables strategy eosTokenIds finished0 maxNewTokens.toNat
    cache logits onStep 0 inputIds

/-- Greedy/sampled generation from token ids. Terminates when every row is
    finished (`eosTokenIds`, default `cfg.eos_token_ids`) or the
    `maxNewTokens` budget is spent. -/
def generate {batch seq : UInt64}
    (cfg : Config)
    (m : LagunaForCausalLM (device := device) (dtype := dtype) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := cfg.eos_token_ids)
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) :=
  generateCore cfg m inputIds maxNewTokens strategy eosTokenIds none

/-- `generate` with a per-step token callback (streaming). -/
def generateStream {batch seq : UInt64}
    (cfg : Config)
    (m : LagunaForCausalLM (device := device) (dtype := dtype) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (onStep : StreamCallback (device := device) batch)
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := cfg.eos_token_ids)
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) :=
  generateCore cfg m inputIds maxNewTokens strategy eosTokenIds (some onStep)

/-- Convenience wrapper for greedy generation. -/
def generateGreedy {batch seq : UInt64}
    (cfg : Config)
    (m : LagunaForCausalLM (device := device) (dtype := dtype) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (maxNewTokens : UInt64 := 256)
    (eosTokenIds : Array UInt64 := cfg.eos_token_ids)
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) :=
  generate cfg m inputIds maxNewTokens .greedy eosTokenIds

end LagunaForCausalLM

end torch.laguna
