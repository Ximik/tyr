/-
  Tyr/Model/Laguna/Rope.lean

  Dual rotary-embedding tables for the poolside Laguna hybrid-attention model
  (Laguna-S-2.1): plain RoPE for sliding-window layers and YaRN RoPE for
  full-attention layers.

  ## Layout contract (hard requirement)

  Every table is a row-major `[maxLen, rotaryDim/2]` fp32 tensor with

    `table[p, j] = cos|sin(p * inv_freq[j]) * scale`

  which is exactly the layout produced by `torch.rotary.computeFreqsOnDevicePure`
  and consumed by `torch.Tensor.applyRotary` (see `cc/src/tyr.cpp`
  `compute_rotary_freqs_impl` / `lean_torch_apply_rotary_emb`).

  `torch.Tensor.applyRotary` implements the GLM-style NON-interleaved
  "rotate-half" convention used by HF `apply_rotary_pos_emb` in
  `dev/laguna_reference/modeling_laguna.py`: for an input slice of width
  `rotaryDim` it rotates channel pairs `(j, j + rotaryDim/2)` as
  `y1 = x1*cos - x2*sin`, `y2 = x1*sin + x2*cos`.  Feeding it
  `cos[p, j] = cos(p * inv_freq[j])` is therefore equivalent to HF's
  `emb = cat(freqs, freqs)` formulation.

  ## Precision note

  Both tables are computed in Lean `Float` (IEEE double) math and rounded to
  fp32 only on upload, so they match a float64 NumPy/torch reference to ~1e-7.
  The C++ fast path `rotary.computeFreqsOnDevicePure` computes the same layout
  in fp32 on-device; its values agree with `slidingCos/slidingSin` to ~6e-5 at
  position 1024 (fp32 accumulation of `p * inv_freq`), so the two are
  layout-compatible but NOT bit-identical.  Table dtypes are fp32;
  `Tensor.applyRotary` casts its result back to the activation dtype.

  ## Usage (for the Model.lean author)

    let tables ← precomputeRotaryTables cfg maxLen device
    -- sliding layers (full rotary, rotaryDim = headDim = 128):
    let cos : DTensor #[seq, 64] .Float32 := sliceRotaryRows tables.slidingCos 0 seq
    let q' := Tensor.applyRotary q cos sin            -- q : STensor #[b, seq, h, 128]
    -- full layers (partial rotary 0.5, rotaryDim = 64):
    let cos : DTensor #[seq, 32] .Float32 := sliceRotaryRows tables.fullCos position 1
    let q' := applyRotaryPartial q cos sin               -- q : STensor #[b, 1, h, 128]
-/
import Tyr.Basic
import Tyr.Typed.Neural
import Tyr.Model.Laguna.Config

namespace torch.laguna

/-- Dual rotary-embedding cos/sin tables for Laguna's hybrid attention.

    All four tensors retain existential shape indices and Float32 dtype on the device passed to
    `precomputeRotaryTables`, with runtime shapes
    `slidingCos/slidingSin : [maxLen, cfg.rotaryDimSliding / 2]` (=`[maxLen, 64]`)
    and `fullCos/fullSin : [maxLen, cfg.rotaryDimFull / 2]` (=`[maxLen, 32]`). -/
structure LagunaRotaryTables where
  /-- cos table for sliding-window layers (plain rope, theta = `rope_theta_sliding`). -/
  slidingCos : ShapePackedTensor .Float32
  /-- sin table for sliding-window layers (plain rope, theta = `rope_theta_sliding`). -/
  slidingSin : ShapePackedTensor .Float32
  /-- cos table for full-attention layers (YaRN, scaled by `yarn_attention_factor`). -/
  fullCos : ShapePackedTensor .Float32
  /-- sin table for full-attention layers (YaRN, scaled by `yarn_attention_factor`). -/
  fullSin : ShapePackedTensor .Float32

/-- `2π` as a literal (Lean core has no `Float.pi`). -/
private def twoPi : Float := 6.283185307179586

/-- Plain-rope inverse frequencies `inv_freq[j] = base^(-2j/dim)` for
    `j ∈ [0, dim/2)`, computed in double precision.
    Matches HF `compute_default_rope_parameters` and the C++
    `compute_rotary_freqs_impl` formula. -/
def plainInvFreq (rotaryDim : UInt64) (base : Float) : Array Float :=
  let dim := rotaryDim.toFloat
  Array.ofFn (n := rotaryDim.toNat / 2) fun j =>
    Float.pow base (-(2.0 * j.val.toFloat / dim))

/-- YaRN correction-dimension helper (HF `find_correction_dim`):
    `dim * ln(maxPos / (numRotations * 2π)) / (2 * ln base)`. -/
private def yarnCorrectionDim (numRotations dim base maxPos : Float) : Float :=
  (dim * Float.log (maxPos / (numRotations * twoPi))) / (2.0 * Float.log base)

/-- YaRN `(low, high)` correction range (HF `find_correction_range`, truncated):
    `low = floor(corrDim(beta_fast))`, `high = ceil(corrDim(beta_slow))`,
    each clamped to `[0, dim-1]`.  For Laguna-S-2.1 this is `(9, 18)`. -/
def yarnCorrectionRange (cfg : Config) : UInt64 × UInt64 :=
  let dim := cfg.rotaryDimFull.toFloat
  let base := cfg.rope_theta_full
  let maxPos := cfg.yarn_original_max_position_embeddings.toFloat
  let low := Float.floor (yarnCorrectionDim cfg.yarn_beta_fast dim base maxPos)
  let high := Float.ceil (yarnCorrectionDim cfg.yarn_beta_slow dim base maxPos)
  let low := min (max low 0.0) (dim - 1.0)
  let high := min (max high 0.0) (dim - 1.0)
  (low.toUInt64, high.toUInt64)

/-- YaRN inverse frequencies (HF `transformers._compute_yarn_parameters`),
    computed in double precision:

    `pos_freqs[j]  = base^(2j/dim)`
    `inv_interp[j] = 1 / (factor * pos_freqs[j])`
    `inv_extrap[j] = 1 / pos_freqs[j]`
    `ramp[j]       = clamp((j - low) / max(high - low, 1), 0, 1)`
    `inv_freq[j]   = inv_interp[j] * ramp[j] + inv_extrap[j] * (1 - ramp[j])`

    for `j ∈ [0, dim/2)` with `dim = cfg.rotaryDimFull` (= 64, so 32 pairs). -/
def yarnInvFreq (cfg : Config) : Array Float :=
  let dim := cfg.rotaryDimFull
  let (low, high) := yarnCorrectionRange cfg
  let denom := max (high.toFloat - low.toFloat) 1.0
  Array.ofFn (n := dim.toNat / 2) fun j =>
    let posFreq := Float.pow cfg.rope_theta_full (2.0 * j.val.toFloat / dim.toFloat)
    let invInterp := 1.0 / (cfg.yarn_factor * posFreq)
    let invExtrap := 1.0 / posFreq
    let ramp := min (max ((j.val.toFloat - low.toFloat) / denom) 0.0) 1.0
    invInterp * ramp + invExtrap * (1.0 - ramp)

/-- Build `[maxLen, invFreq.size]` cos/sin tables from inverse frequencies.

    Computed in double precision on CPU, then uploaded to `device` as fp32
    row-major tensors with `table[p, j] = cos|sin(p * invFreq[j]) * scale` —
    the layout consumed by `Tensor.applyRotary`. -/
def buildCosSinTables (invFreq : Array Float) (maxLen : UInt64) (scale : Float := 1.0)
    (device : Device := Device.CPU) :
    DTensor #[maxLen, invFreq.size.toUInt64] .Float32 ×
    DTensor #[maxLen, invFreq.size.toUInt64] .Float32 :=
  let half := invFreq.size
  let n := maxLen.toNat
  let (cosArr, sinArr) := Id.run do
    let mut cosArr := Array.mkEmpty (n * half)
    let mut sinArr := Array.mkEmpty (n * half)
    for p in [:n] do
      let pf := p.toFloat
      for j in [:half] do
        let f := pf * invFreq.getD j 0.0
        cosArr := cosArr.push (Float.cos f * scale)
        sinArr := sinArr.push (Float.sin f * scale)
    pure (cosArr, sinArr)
  let shape : Shape := #[maxLen, half.toUInt64]
  ((Tensor.reshapeChecked (Tensor.fromFloatArray cosArr) shape).to device,
   (Tensor.reshapeChecked (Tensor.fromFloatArray sinArr) shape).to device)

/-- Precompute Laguna's dual rotary tables on `device` for positions `[0, maxLen)`.

    * `slidingCos/slidingSin`: plain rope, `inv_freq[j] = rope_theta_sliding^(-2j/128)`,
      scale 1.0 — same formula as `rotary.computeFreqsOnDevicePure maxLen
      cfg.rotaryDimSliding cfg.rope_theta_sliding device`, but computed in double
      precision (the C++ fp32 fast path is layout-compatible, see module doc).
    * `fullCos/fullSin`: YaRN rope over `cfg.rotaryDimFull` with
      `yarn_attention_factor` scaling.

    Cost note: tables are built in Lean, so construction is `O(maxLen * rotaryDim/2)`
    scalar sin/cos evaluations — pick `maxLen` accordingly (the decode loop only
    ever slices rows). -/
def precomputeRotaryTables (cfg : Config) (maxLen : UInt64) (device : Device)
    : IO LagunaRotaryTables := do
  let (slidingCos, slidingSin) :=
    buildCosSinTables (plainInvFreq cfg.rotaryDimSliding cfg.rope_theta_sliding)
      maxLen 1.0 device
  let (fullCos, fullSin) :=
    buildCosSinTables (yarnInvFreq cfg) maxLen cfg.yarn_attention_factor device
  pure {
    slidingCos := slidingCos.packShape
    slidingSin := slidingSin.packShape
    fullCos := fullCos.packShape
    fullSin := fullSin.packShape
  }

/-- Slice rows `[start, start + len)` of a `[maxLen, half]` rotary table,
    returning a typed `[len, half]` tensor ready for `Tensor.applyRotary`.

    Prefill: `sliceRotaryRows table 0 seq`.  Decode step at `position`:
    `sliceRotaryRows table position 1`. -/
def sliceRotaryRows {half : UInt64} (table : ShapePackedTensor .Float32)
    (start len : UInt64) : DTensor #[len, half] .Float32 :=
  table.2.sliceChecked 0 start len #[len, half]

/-- Slice rows `[start, start + len)` of a cos/sin table pair. -/
def sliceRotaryRowsPair {half : UInt64}
    (cosTable sinTable : ShapePackedTensor .Float32) (start len : UInt64)
    : DTensor #[len, half] .Float32 × DTensor #[len, half] .Float32 :=
  (sliceRotaryRows cosTable start len, sliceRotaryRows sinTable start len)

/-- Apply rotary embeddings to the first `rotary_dim` channels of `x`
    (partial rotary), passing the remaining channels through unchanged.
    Mirrors HF `apply_rotary_pos_emb` in `modeling_laguna.py` and
    `applyRotaryPartial` in `Tyr/Model/Qwen35/Model.lean`.

    * Full-attention layers: `rotary_dim = cfg.rotaryDimFull` (= 64),
      cos/sin from `fullCos/fullSin`.
    * Sliding-window layers (`rotary_dim = head_dim = 128`) can equally call
      `Tensor.applyRotary` directly on the full head; this helper handles
      that case too (the pass-through slice is then empty). -/
def applyRotaryPartial {batch seq n_head head_dim rotary_dim : UInt64} {d dc : DType}
    (x : DTensor #[batch, seq, n_head, head_dim] d)
    (cos : DTensor #[seq, rotary_dim / 2] dc)
    (sin : DTensor #[seq, rotary_dim / 2] dc)
    : DTensor #[batch, seq, n_head, head_dim] d :=
  let xRot := x.sliceChecked 3 0 rotary_dim #[batch, seq, n_head, rotary_dim]
  let xRot := Tensor.applyRotary xRot cos sin
  let xPassLen : UInt64 := head_dim - rotary_dim
  let xPass := x.sliceChecked 3 rotary_dim xPassLen #[batch, seq, n_head, xPassLen]
  Tensor.catManyChecked #[xRot.packShape, xPass.packShape] 3 #[batch, seq, n_head, head_dim]

end torch.laguna
