/-
  Tyr/Model/Laguna/MoE.lean

  Laguna sparse MoE block for poolside Laguna-S-2.1 (NVFP4 variant):
  sigmoid top-k router (aux-loss-free bias supported), NVFP4-packed expert
  banks dequantized on the fly, and a dense shared expert.

  Mirrors the HuggingFace reference math in
  `dev/laguna_reference/modeling_laguna.py`
  (`LagunaTopKRouter`, `LagunaExperts`, `LagunaSparseMoeBlock`, `LagunaMLP`):

    logits           = x @ routerWeightᵀ                (FP32; see note below)
    scores           = sigmoid(logits)                  (optional tanh softcap)
    selection        = scores + eScoreCorrectionBias    (bias only for picking)
    idx              = topk(selection, k)
    weights          = gather(scores, idx)              (UNbiased)
    weights          = weights / Σ weights              (norm_topk_prob)
    out              = Σ_k weights_k · expert_{idx_k}(x)   (weights on OUTPUT)
    out              = moe_routed_scaling_factor · out + sharedExpert(x)
    expert(x)        = down(silu(gate x) * up x)        (SwiGLU, BF16 matmuls)

  Deviation from the HF reference, by design: HF computes
  `F.linear(hidden, weight).float()` (BF16 matmul, then cast), while this
  implementation follows the task spec and computes the router matmul in FP32
  (cast first, then matmul). The difference is below BF16 output rounding and
  is covered by the test tolerance.

  Token-dispatch strategy: per-unique-expert gather (HF `LagunaExperts`
  style). For each expert hit by the batch we gather its token rows, dequantize
  that expert's three NVFP4 matrices ONCE per forward (regardless of token
  count), run the SwiGLU in BF16, weight the outputs, and scatter-add them into
  the accumulator. Iteration is in ascending expert index order, matching the
  HF `expert_hit` enumeration, so BF16 accumulation order (and thus rounding)
  matches the reference as closely as the op set allows.

  Dense-BF16 expert banks: BF16 checkpoints (e.g. the tiny parity fixture)
  ship experts as fused dense 3-D banks (`mlp.experts.gate_up_proj` /
  `mlp.experts.down_proj`) instead of per-expert NVFP4-packed tensors. Those
  fit `LagunaDenseExperts` (set on `LagunaSparseMoeBlock.denseExperts` by the
  checkpoint loader); `forward2d` then slices the dense banks in the same
  ascending-expert dispatch order and skips NVFP4 dequantization entirely.
-/
import Tyr.Model.Laguna.TypedOps
import Tyr.TensorStruct
import Tyr.TensorStruct.DeriveCast
import Tyr.Module.Derive
import Tyr.Model.Laguna.Config
import Tyr.Model.Laguna.NvFp4
import Tyr.Model.Laguna.Fused

namespace torch.laguna

variable {device : Device} {dtype : DType}
variable [DType.RealFloating dtype]
open torch.qwen35

/-- Top-k sigmoid router for Laguna MoE (aux-loss-free load balancing). -/
structure LagunaTopKRouter (cfg : Config) where
  /-- Router projection `[num_experts, hidden_size]` (BF16 in the checkpoint). -/
  weight : SomeDTypeTensor #[cfg.num_experts, cfg.hidden_size] device
  /-- Optional per-expert selection bias (added to selection scores only). -/
  eScoreCorrectionBias : Option (SomeDTypeTensor #[cfg.num_experts] device) := none
  deriving TensorStruct


namespace LagunaTopKRouter

/-- Route tokens to experts.
    Returns `(selectedExperts, routingWeights)`, both `[tokens, num_experts_per_tok]`;
    `selectedExperts` is int64, `routingWeights` has `x`'s dtype. -/
def route {tokens : Nat}
    (cfg : Config) (r : LagunaTopKRouter (device := device) cfg) (x : Tensor { dtype := dtype, device := device, shape := #[tokens, cfg.hidden_size] })
    : IO (Tensor { device := device, shape := #[tokens, cfg.num_experts_per_tok], dtype := .Int64 } × Tensor { dtype := dtype, device := device, shape := #[tokens, cfg.num_experts_per_tok] }) := do
  let logits0 : Tensor { dtype := .Float32, device := device, shape := #[tokens, cfg.num_experts] } :=
    einsumChecked "eh,th->te" (floatDynamic r.weight.2) (floatDynamic x)
  let logits :=
    if cfg.moe_router_logit_softcapping > 0.0 then
      let c := cfg.moe_router_logit_softcapping
      Tensor.mulScalar (Tensor.tanh (Tensor.divScalar logits0 c)) c
    else logits0
  let scores := Tensor.sigmoid logits
  let selectionScores := match r.eScoreCorrectionBias with
    | some bias => Tensor.add scores (Tensor.expand (floatDynamic bias.2) #[tokens, cfg.num_experts])
    | none => scores
  let (_, topIdx) := topK2d selectionScores cfg.num_experts_per_tok 1
  let weights := Tensor.gather scores 1 topIdx
  let weights := if cfg.norm_topk_prob then
      Tensor.div weights (Tensor.expand (Tensor.sumDim weights 1 true) #[tokens, cfg.num_experts_per_tok])
    else weights
  pure (topIdx, Tensor.castLike x weights)

end LagunaTopKRouter

/-- NVFP4-packed expert weight banks for Laguna MoE (shape-indexed device tensors).
    Runtime shapes:
    - gate/up: packed U8 `[E, moe_intermediate, hidden/2]`,
      scales F8_E4M3 `[E, moe_intermediate, hidden/16]`
    - down: packed U8 `[E, hidden, moe_intermediate/2]`,
      scales F8_E4M3 `[E, hidden, moe_intermediate/16]`
    - globals: F32 `[E]` -/
structure LagunaPackedExperts (cfg : Config) where
  gatePacked : SomeDTypeTensor #[cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size / 2] device
  gateScale : SomeDTypeTensor #[cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size / 16] device
  gateGlobal : SomeDTypeTensor #[cfg.num_experts] device
  upPacked : SomeDTypeTensor #[cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size / 2] device
  upScale : SomeDTypeTensor #[cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size / 16] device
  upGlobal : SomeDTypeTensor #[cfg.num_experts] device
  downPacked : SomeDTypeTensor #[cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size / 2] device
  downScale : SomeDTypeTensor #[cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size / 16] device
  downGlobal : SomeDTypeTensor #[cfg.num_experts] device
  deriving TensorStruct


/-- Dense BF16 expert weight banks for Laguna MoE (shape-indexed device
    tensors). Used for BF16 checkpoints that store experts as fused 3-D banks
    instead of NVFP4-packed per-expert tensors. Runtime shapes:
    - gateProj/upProj: BF16 `[E, moe_intermediate, hidden]`
    - downProj: BF16 `[E, hidden, moe_intermediate]` -/
structure LagunaDenseExperts (cfg : Config) (dtype : DType := .Float32) where
  gateProj : Tensor { dtype := dtype, device := device, shape := #[cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size] }
  upProj : Tensor { dtype := dtype, device := device, shape := #[cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size] }
  downProj : Tensor { dtype := dtype, device := device, shape := #[cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size] }
  deriving TensorStruct


/-- Sparse MoE block: sigmoid-routed packed experts + dense shared expert. -/
structure LagunaSparseMoeBlock (cfg : Config) (dtype : DType := .Float32) where
  router : LagunaTopKRouter (device := device) cfg
  experts : LagunaPackedExperts (device := device) cfg
  /-- Optional dense BF16 expert banks. When set, `forward2d` uses these
      instead of the NVFP4-packed `experts` banks (which may be unused
      placeholders); the per-expert dispatch order is unchanged. -/
  denseExperts : Option (LagunaDenseExperts (device := device) (dtype := dtype) cfg) := none
  sharedGateProj : Tensor { dtype := dtype, device := device, shape := #[cfg.shared_expert_intermediate_size, cfg.hidden_size] }
  sharedUpProj : Tensor { dtype := dtype, device := device, shape := #[cfg.shared_expert_intermediate_size, cfg.hidden_size] }
  sharedDownProj : Tensor { dtype := dtype, device := device, shape := #[cfg.hidden_size, cfg.shared_expert_intermediate_size] }
  deriving TensorStruct


namespace LagunaSparseMoeBlock

/-- Dequantize one expert row while retaining the bank's three dimensions. -/
private def dequantExpertSlice {numExperts outFeatures inFeatures : Nat} {pd sd gd : DType}
    (bankPacked : Tensor { dtype := pd, device := device, shape := #[numExperts, outFeatures, inFeatures / 2] })
    (bankScales : Tensor { dtype := sd, device := device, shape := #[numExperts, outFeatures, inFeatures / 16] })
    (bankGlobal : Tensor { dtype := gd, device := device, shape := #[numExperts] }) (e : Nat)
    : IO (Tensor { dtype := .BFloat16, device := device, shape := #[outFeatures, inFeatures] }) := do
  let out ← nvfp4.dequantMatrix
    ((bankPacked.sliceChecked 0 e 1 #[1, outFeatures, inFeatures / 2]).reshape #[outFeatures, inFeatures / 2])
    ((bankScales.sliceChecked 0 e 1 #[1, outFeatures, inFeatures / 16]).reshape #[outFeatures, inFeatures / 16])
    (bankGlobal.sliceChecked 0 e 1 #[1]) outFeatures inFeatures
  pure out

/-- Fetch one expert's three SwiGLU matrices in BF16: slices of the dense
    banks when `denseExperts` is set, otherwise NVFP4 dequant of the packed
    bank rows. -/
private def expertWeights
    (cfg : Config)
    (m : LagunaSparseMoeBlock (device := device) (dtype := dtype) cfg)
    (e : Nat)
    : IO (Tensor { dtype := dtype, device := device, shape := #[cfg.moe_intermediate_size, cfg.hidden_size] }
        × Tensor { dtype := dtype, device := device, shape := #[cfg.moe_intermediate_size, cfg.hidden_size] }
        × Tensor { dtype := dtype, device := device, shape := #[cfg.hidden_size, cfg.moe_intermediate_size] }) := do
  match m.denseExperts with
  | some de =>
    let gate3 := de.gateProj
    let up3 := de.upProj
    let down3 := de.downProj
    pure (
      Tensor.reshape (sliceTyped gate3 0 e 1) #[cfg.moe_intermediate_size, cfg.hidden_size],
      Tensor.reshape (sliceTyped up3 0 e 1) #[cfg.moe_intermediate_size, cfg.hidden_size],
      Tensor.reshape (sliceTyped down3 0 e 1) #[cfg.hidden_size, cfg.moe_intermediate_size])
  | none =>
    let gateW : Tensor { dtype := .BFloat16, device := device, shape := #[cfg.moe_intermediate_size, cfg.hidden_size] } :=
      Tensor.reshape (← dequantExpertSlice (inFeatures := cfg.hidden_size) m.experts.gatePacked.2 m.experts.gateScale.2 m.experts.gateGlobal.2
        e) #[cfg.moe_intermediate_size, cfg.hidden_size]
    let upW : Tensor { dtype := .BFloat16, device := device, shape := #[cfg.moe_intermediate_size, cfg.hidden_size] } :=
      Tensor.reshape (← dequantExpertSlice (inFeatures := cfg.hidden_size) m.experts.upPacked.2 m.experts.upScale.2 m.experts.upGlobal.2
        e) #[cfg.moe_intermediate_size, cfg.hidden_size]
    let downW : Tensor { dtype := .BFloat16, device := device, shape := #[cfg.hidden_size, cfg.moe_intermediate_size] } :=
      Tensor.reshape (← dequantExpertSlice (inFeatures := cfg.moe_intermediate_size) m.experts.downPacked.2 m.experts.downScale.2 m.experts.downGlobal.2
        e) #[cfg.hidden_size, cfg.moe_intermediate_size]
    pure (← IO.ofExcept (Tensor.ofRaw gateW.raw), ← IO.ofExcept (Tensor.ofRaw upW.raw), ← IO.ofExcept (Tensor.ofRaw downW.raw))

/-- Batched single-token decode path. For one token the k router-selected
    experts are distinct, so the k expert triples can be gathered and
    dequantized as one bank and the SwiGLU runs as batched GEMVs (`bmm`) —
    ~15 kernel launches per layer instead of ~15 per expert. Same math as the
    general path (weights on expert output, scaling factor, + shared expert);
    only the reduction order over k differs (BF16 tolerance).

    On CUDA with BF16 input and NVFP4-packed banks (`denseExperts = none`),
    the routed sum is computed by the fused kernel in
    `Tyr/Model/Laguna/Fused.lean` (one streaming pass over the packed
    weights, no BF16 materialization) unless `useFused` is false; everything
    else keeps the eager gather+dequant+bmm path below. -/
private def forwardDecode1
    (cfg : Config)
    (m : LagunaSparseMoeBlock (device := device) (dtype := dtype) cfg)
    (x : Tensor { dtype := dtype, device := device, shape := #[1, cfg.hidden_size] })
    (flatIdx : Tensor { device := device, shape := #[cfg.num_experts_per_tok], dtype := .Int64 })
    (flatW : Tensor { dtype := dtype, device := device, shape := #[cfg.num_experts_per_tok] })
    (useFused : Bool := true)
    : IO (Tensor { dtype := dtype, device := device, shape := #[1, cfg.hidden_size] }) := do
  let k := cfg.num_experts_per_tok
  let E := cfg.num_experts
  let mi := cfg.moe_intermediate_size
  let h := cfg.hidden_size
  let useFusedKernel : Bool :=
    useFused && m.denseExperts.isNone && h % 32 == 0 && mi % 32 == 0 &&
    (match x.device.kind with | .CUDA _ => true | _ => false) &&
    x.actualSpec.dtype == .BFloat16 &&
    -- Only real NVFP4 storage uses the fused path. Random-init stand-ins
    -- retain their independent Int64 bytes and Float32 scale indices.
    m.experts.gatePacked.2.actualSpec.dtype == .UInt8 && m.experts.upPacked.2.actualSpec.dtype == .UInt8 &&
    m.experts.downPacked.2.actualSpec.dtype == .UInt8
  let acc : Tensor { dtype := dtype, device := device, shape := #[1, h] } ←
    if useFusedKernel then do
      -- Fused NVFP4 kernel: routed[t] = Σ_p w[t,p] · expert_{idx[t,p]}(x[t]).
      let xE := x
      let topIdx2d := Tensor.reshape flatIdx #[1, k]
      let topW2d := Tensor.reshape flatW #[1, k]
      let fused ← lagunaMoeFp4Forward xE topIdx2d topW2d
        m.experts.gatePacked.2 m.experts.gateScale.2 m.experts.gateGlobal.2
        m.experts.upPacked.2 m.experts.upScale.2 m.experts.upGlobal.2
        m.experts.downPacked.2 m.experts.downScale.2 m.experts.downGlobal.2
        E mi h
      IO.ofExcept (Tensor.ofRaw fused.raw)
    else do
      let hitT : Tensor { device := device, shape := #[k], dtype := .Int64 } := Tensor.reshape flatIdx #[k]
      let (gateW, upW, downW) ← do
        match m.denseExperts with
        | some de =>
          let g : Tensor { dtype := dtype, device := device, shape := #[k, mi, h] } := Tensor.reshape (Tensor.embedding1d hitT (Tensor.reshape de.gateProj #[E, mi * h])) #[k, mi, h]
          let u : Tensor { dtype := dtype, device := device, shape := #[k, mi, h] } := Tensor.reshape (Tensor.embedding1d hitT (Tensor.reshape de.upProj #[E, mi * h])) #[k, mi, h]
          let d : Tensor { dtype := dtype, device := device, shape := #[k, h, mi] } := Tensor.reshape (Tensor.embedding1d hitT (Tensor.reshape de.downProj #[E, h * mi])) #[k, h, mi]
          pure (g, u, d)
        | none => do
          let gp := Tensor.reshape (Tensor.embedding1d hitT (Tensor.reshape m.experts.gatePacked.2 #[E, mi * (h / 2)])) #[k, mi, h / 2]
          let gs := Tensor.reshape (Tensor.embedding1d hitT (Tensor.reshape m.experts.gateScale.2 #[E, mi * (h / 16)])) #[k, mi, h / 16]
          let gg := Tensor.reshape (Tensor.embedding1d hitT (Tensor.reshape m.experts.gateGlobal.2 #[E, 1])) #[k]
          let up := Tensor.reshape (Tensor.embedding1d hitT (Tensor.reshape m.experts.upPacked.2 #[E, mi * (h / 2)])) #[k, mi, h / 2]
          let us := Tensor.reshape (Tensor.embedding1d hitT (Tensor.reshape m.experts.upScale.2 #[E, mi * (h / 16)])) #[k, mi, h / 16]
          let ug := Tensor.reshape (Tensor.embedding1d hitT (Tensor.reshape m.experts.upGlobal.2 #[E, 1])) #[k]
          let dp := Tensor.reshape (Tensor.embedding1d hitT (Tensor.reshape m.experts.downPacked.2 #[E, h * (mi / 2)])) #[k, h, mi / 2]
          let ds := Tensor.reshape (Tensor.embedding1d hitT (Tensor.reshape m.experts.downScale.2 #[E, h * (mi / 16)])) #[k, h, mi / 16]
          let dg := Tensor.reshape (Tensor.embedding1d hitT (Tensor.reshape m.experts.downGlobal.2 #[E, 1])) #[k]
          let g ← nvfp4.dequantBank gp gs gg k mi h
          let u ← nvfp4.dequantBank up us ug k mi h
          let d ← nvfp4.dequantBank dp ds dg k h mi
          pure (← IO.ofExcept (Tensor.ofRaw g.raw), ← IO.ofExcept (Tensor.ofRaw u.raw), ← IO.ofExcept (Tensor.ofRaw d.raw))
      -- Batched SwiGLU: [k,1,h] @ Wᵀ per projection.
      let x3 : Tensor { dtype := dtype, device := device, shape := #[k, 1, h] } := Tensor.reshape (Tensor.expand (Tensor.reshape x #[1, 1, h]) #[k, 1, h]) #[k, 1, h]
      let g : Tensor { dtype := dtype, device := device, shape := #[k, 1, mi] } := bmm x3 (transposeMatrixBatch gateW)
      let u : Tensor { dtype := dtype, device := device, shape := #[k, 1, mi] } := bmm x3 (transposeMatrixBatch upW)
      let hid : Tensor { dtype := dtype, device := device, shape := #[k, 1, mi] } := mulSame (Tensor.silu g) u
      let o : Tensor { dtype := dtype, device := device, shape := #[k, 1, h] } := bmm hid (transposeMatrixBatch downW)
      let ow : Tensor { dtype := dtype, device := device, shape := #[k, 1, h] } := mulBroadcast (Tensor.reshape flatW #[k, 1, 1]) o
      pure (show Tensor ⟨#[1, h], dtype, device⟩ from by
        simpa using (Tensor.reshape (Tensor.sumDim ow 0 false) #[1, h]))
  -- Dense shared expert (SwiGLU MLP, no gating score).
  let sharedH : Tensor { dtype := dtype, device := device, shape := #[1, cfg.shared_expert_intermediate_size] } :=
    mulSame (Tensor.silu (Tensor.linear x m.sharedGateProj)) (Tensor.linear x m.sharedUpProj)
  let sharedOut : Tensor { dtype := dtype, device := device, shape := #[1, h] } := Tensor.linear sharedH m.sharedDownProj
  pure (Tensor.reshape (addSame (Tensor.mulScalar acc cfg.moe_routed_scaling_factor) sharedOut) #[1, h])

/-- MoE forward over a 2D `[tokens, hidden]` batch.
    Returns `moe_routed_scaling_factor · Σ_k w_k · expert_k(x) + sharedExpert(x)`
    in `x`'s dtype. Single-token decode on CUDA uses the fused NVFP4 kernel
    for the packed-bank path unless `useFused` is false. -/
def forward2d {tokens : Nat}
    (cfg : Config)
    (m : LagunaSparseMoeBlock (device := device) (dtype := dtype) cfg)
    (x : Tensor { dtype := dtype, device := device, shape := #[tokens, cfg.hidden_size] })
    (useFused : Bool := true)
    : IO (Tensor { dtype := dtype, device := device, shape := #[tokens, cfg.hidden_size] }) := do
  let k := cfg.num_experts_per_tok
  let (topIdx, weights) ← m.router.route cfg x
  -- Flattened dispatch views: pair p = token * k + slot.
  let flatIdx : Tensor { device := device, shape := #[tokens * k], dtype := .Int64 } := Tensor.reshape topIdx #[tokens * k]
  let flatW : Tensor { dtype := dtype, device := device, shape := #[tokens * k] } := Tensor.reshape weights #[tokens * k]
  let device := x.device
  if tokens == 1 then
    -- Single-token decode: batched path (k distinct experts, no dispatch scan).
    let x1 : Tensor { dtype := dtype, device := device, shape := #[1, cfg.hidden_size] } := Tensor.reshapeChecked x #[1, cfg.hidden_size]
    pure (Tensor.reshapeChecked (← forwardDecode1 cfg m x1 (flatIdx.reshapeChecked #[k]) (flatW.reshapeChecked #[k]) useFused) #[tokens, cfg.hidden_size])
  else do
    let h := cfg.hidden_size
    let mi := cfg.moe_intermediate_size
    let fusedAvailable : Bool :=
      useFused && m.denseExperts.isNone && h % 32 == 0 && mi % 32 == 0 &&
      (match device.kind with | .CUDA _ => true | _ => false) &&
      x.actualSpec.dtype == .BFloat16 &&
      m.experts.gatePacked.2.actualSpec.dtype == .UInt8 && m.experts.upPacked.2.actualSpec.dtype == .UInt8 &&
      m.experts.downPacked.2.actualSpec.dtype == .UInt8
    if fusedAvailable then do
      -- Multi-token fused path: the kernel grids over pairs = tokens*k.
      let xE := x
      let topIdx2d := Tensor.reshape flatIdx #[tokens, k]
      let topW2d := Tensor.reshape flatW #[tokens, k]
      let routedBF16 ← lagunaMoeFp4Forward xE topIdx2d topW2d
        m.experts.gatePacked.2 m.experts.gateScale.2 m.experts.gateGlobal.2
        m.experts.upPacked.2 m.experts.upScale.2 m.experts.upGlobal.2
        m.experts.downPacked.2 m.experts.downScale.2 m.experts.downGlobal.2
        cfg.num_experts mi h
      let routed : Tensor { dtype := dtype, device := device, shape := #[tokens, cfg.hidden_size] } ← IO.ofExcept (Tensor.ofRaw routedBF16.raw)
      -- Dense shared expert (SwiGLU MLP, no gating score).
      let sharedH : Tensor { dtype := dtype, device := device, shape := #[tokens, cfg.shared_expert_intermediate_size] } :=
        mulSame (Tensor.silu (Tensor.linear x m.sharedGateProj)) (Tensor.linear x m.sharedUpProj)
      let sharedOut : Tensor { dtype := dtype, device := device, shape := #[tokens, cfg.hidden_size] } := Tensor.linear sharedH m.sharedDownProj
      pure (Tensor.reshape (addSame (Tensor.mulScalar routed cfg.moe_routed_scaling_factor) sharedOut) #[tokens, cfg.hidden_size])
    else do
      -- Per-pair token row ids: [0..tokens) repeated k times each.
      -- NOTE: `arange` always allocates on CPU; move to the model device so
      -- `masked_select`/`embedding1d` see input and mask/index on one device.
      let tokIdsFlat :=
        Tensor.reshape (Tensor.expand (Tensor.reshape ((Tensor.arange tokens).to device) #[tokens, 1]) #[tokens, k]) #[tokens * k]
      -- Accumulator in the model dtype (HF: torch.zeros_like(hidden_states)).
      let mut acc : Tensor { dtype := dtype, device := device, shape := #[tokens, cfg.hidden_size] } := Tensor.castLike x (zerosDynamic #[tokens, cfg.hidden_size] false device)
      -- Per-unique-expert dispatch: materialize each hit expert once per forward.
      -- One CPU readback of the (small) dispatch list — <= tokens*k entries —
      -- instead of scanning all num_experts with a GPU mask + host sync per
      -- expert (that is ~256 syncs/layer, fatal at decode). Iteration stays in
      -- ascending expert order, matching the HF `expert_hit` accumulation order.
      let flatIdxHost ← Tensor.toUInt64Array (Tensor.reshape (Tensor.toInt64 flatIdx) #[tokens * k])
      let hitExperts : Array Nat := Id.run do
        let mut seen : Array Nat := #[]
        for id in flatIdxHost do
          if !(seen.contains id.toNat) then
            seen := seen.push id.toNat
        pure (seen.qsort (· < ·))
      for eU in hitExperts do
        let maskE := Tensor.eqScalar flatIdx eU.toInt64
        let tokEr := Tensor.maskedSelect tokIdsFlat maskE
        let wE := Tensor.maskedSelect flatW maskE
        let nE := tokEr.1
        let tokE := tokEr.2
        let (gateW, upW, downW) ← expertWeights cfg m eU
        -- SwiGLU expert MLP in BF16.
        let xE : Tensor { dtype := dtype, device := device, shape := #[nE, cfg.hidden_size] } := Tensor.embedding1d tokE x
        let hid : Tensor { dtype := dtype, device := device, shape := #[nE, cfg.moe_intermediate_size] } :=
          mulSame (Tensor.silu (Tensor.linear xE gateW)) (Tensor.linear xE upW)
        let outE : Tensor { dtype := dtype, device := device, shape := #[nE, cfg.hidden_size] } := Tensor.linear hid downW
        -- Routing weights multiply the expert OUTPUT (moe_apply_router_weight_on_input=false).
        let outEW : Tensor { dtype := dtype, device := device, shape := #[nE, cfg.hidden_size] } := mulBroadcast (Tensor.reshapeChecked wE.2 #[nE, 1]) outE
        let idxRows : Tensor { device := device, shape := #[nE, cfg.hidden_size], dtype := .Int64 } :=
          Tensor.reshape (Tensor.expand (Tensor.reshape tokE #[nE, 1]) #[nE, cfg.hidden_size]) #[nE, cfg.hidden_size]
        acc := scatterAdd acc 0 idxRows outEW
      -- Dense shared expert (SwiGLU MLP, no gating score).
      let sharedH : Tensor { dtype := dtype, device := device, shape := #[tokens, cfg.shared_expert_intermediate_size] } :=
        mulSame (Tensor.silu (Tensor.linear x m.sharedGateProj)) (Tensor.linear x m.sharedUpProj)
      let sharedOut : Tensor { dtype := dtype, device := device, shape := #[tokens, cfg.hidden_size] } := Tensor.linear sharedH m.sharedDownProj
      -- HF multiplies the routed output by the scaling factor, then adds the shared expert.
      let routed : Tensor { dtype := dtype, device := device, shape := #[tokens, cfg.hidden_size] } := Tensor.mulScalar acc cfg.moe_routed_scaling_factor
      pure (Tensor.reshape (addSame routed sharedOut) #[tokens, cfg.hidden_size])

end LagunaSparseMoeBlock

end torch.laguna
