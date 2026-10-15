/- Experimental high-level wrappers for `mha_h100` GPU kernels.
   Layering:
   1) raw launchers: `Tyr.GPU.Kernels.MhaH100`
   2) op wrappers: this module
   3) explicit forward/backward API: `mhaFwd` / `mhaBwd`
   4) TK-style dispatch helpers: `mhaFwdDispatch` / `mhaBwdDispatch` -/
import Tyr.Torch
import Tyr.Typed.Neural
import Tyr.GPU.Ops.AttentionProblem
import Tyr.GPU.Kernels.MhaH100

namespace Tyr.GPU.Ops.MhaH100

open torch

variable {device : Device}
open Tyr.GPU.Kernels

/-- BF16 tensor shape used by `mha_h100` kernels: `[1, 1, seq, 64]`. -/
abbrev BF16 (seq : UInt64) : Type := Tensor { shape := #[1, 1, seq, 64], dtype := .BFloat16, device := device }

/-- FP32 gradient/output tensor shape used by `mha_h100` kernels. -/
abbrev F32 (seq : UInt64) : Type := Tensor { shape := #[1, 1, seq, 64], dtype := .Float32, device := device }

/-- Per-query L (device := device)/D vector tensor shape matching TK vector TMA layouts: `[1, 1, 1, seq]`. -/
abbrev L (kvBlocks : UInt64) : Type := Tensor { shape := #[1, 1, 1, kvBlocks * 64], dtype := .Float32, device := device }

/-- Launch configuration shared across wrapper entry points. -/
structure LaunchCfg where
  gridX : UInt64 := 1
  gridY : UInt64
  gridZ : UInt64 := 1
  blockX : UInt64 := 128
  blockY : UInt64 := 1
  blockZ : UInt64 := 1
  sharedMem : UInt64 := 0
  stream : UInt64 := 0

def LaunchCfg.default (kvBlocks stream : UInt64) : LaunchCfg :=
  { gridY := kvBlocks, stream := stream }

def h100MaxDynamicSharedMem : UInt64 := 227 * 1024 - 1024

def fwdSharedMem : UInt64 := h100MaxDynamicSharedMem

def bwdPrepSharedMem : UInt64 := h100MaxDynamicSharedMem

def bwdSweepSharedMem : UInt64 := 117760

/-- Forward outputs reused by backward (`out`, `lOut`). -/
structure FwdCtx (seq kvBlocks : UInt64) where
  out : BF16 (device := device) seq
  lOut : L (device := device) kvBlocks

/-- Forward context for dispatch helpers.
    For kernel-backed variants:
    - `seq=128` stores `lOut2`
    - `seq=768` stores `lOut12`
    Otherwise both remain `none` and backward uses the portable SDPA path. -/
structure FwdCtxDispatch (seq : UInt64) where
  out : BF16 (device := device) seq
  selection : AttentionSpecialization := .portable
  lOut2 : Option (L (device := device) 2) := none
  lOut12 : Option (L (device := device) 12) := none

/-- Variant (device := device) hooks for supported kernel families. -/
class Variant (seq kvBlocks : UInt64) where
  launchFwd :
      BF16 (device := device) seq → BF16 (device := device) seq → BF16 (device := device) seq → BF16 (device := device) seq → L (device := device) kvBlocks → LaunchCfg → IO Unit
  launchBwd :
      BF16 (device := device) seq → BF16 (device := device) seq → BF16 (device := device) seq → BF16 (device := device) seq → L (device := device) kvBlocks → L (device := device) kvBlocks →
      F32 (device := device) seq → F32 (device := device) seq → F32 (device := device) seq → LaunchCfg → IO Unit

instance : Variant (device := device) 128 2 where
  launchFwd q k v out lOut cfg := do
    tkMhaH100Fwd2Block.launch q.raw k.raw v.raw out.raw lOut.raw 128 64
      cfg.gridX cfg.gridY cfg.gridZ
      cfg.blockX cfg.blockY cfg.blockZ
      cfg.sharedMem cfg.stream

  launchBwd q k v dO lOut dVec dQ dK dV cfg := do
    tkMhaH100Bwd2BlockKvSweep.launch q.raw k.raw v.raw dO.raw lOut.raw dVec.raw
      dQ.raw dK.raw dV.raw 128 64
      cfg.gridX cfg.gridY cfg.gridZ
      cfg.blockX cfg.blockY cfg.blockZ
      cfg.sharedMem cfg.stream

instance : Variant (device := device) 768 12 where
  launchFwd q k v out lOut cfg := do
    tkMhaH100Fwd12Block.launch q.raw k.raw v.raw out.raw lOut.raw 768 64
      cfg.gridX cfg.gridY cfg.gridZ
      cfg.blockX cfg.blockY cfg.blockZ
      cfg.sharedMem cfg.stream

  launchBwd q k v dO lOut dVec dQ dK dV cfg := do
    tkMhaH100Bwd12BlockKvSweep.launch q.raw k.raw v.raw dO.raw lOut.raw dVec.raw
      dQ.raw dK.raw dV.raw 768 64
      cfg.gridX cfg.gridY cfg.gridZ
      cfg.blockX cfg.blockY cfg.blockZ
      cfg.sharedMem cfg.stream

private def launchBwdPrep {seq kvBlocks : UInt64}
    (dO out : BF16 (device := device) seq) (dVec : L (device := device) kvBlocks) (cfg : LaunchCfg) : IO Unit := do
  tkMhaH100BwdPrep2Block.launch dO.raw out.raw dVec.raw seq 64
    cfg.gridX cfg.gridY cfg.gridZ
    cfg.blockX cfg.blockY cfg.blockZ
    cfg.sharedMem cfg.stream

/-- High-level forward op for supported `mha_h100` variants. -/
def mhaFwd {seq kvBlocks : UInt64} [Variant (device := device) seq kvBlocks]
    (q k v : BF16 (device := device) seq) (stream : UInt64 := 0) : IO (FwdCtx (device := device) seq kvBlocks) := do
  let cfg := { LaunchCfg.default kvBlocks stream with sharedMem := fwdSharedMem }
  let out := Tensor.zerosLike q
  let lOut : L (device := device) kvBlocks := Tensor.zeros #[1, 1, 1, kvBlocks * 64] false q.device
  Variant.launchFwd (seq := seq) (kvBlocks := kvBlocks) q k v out lOut cfg
  pure { out, lOut }

/-- High-level backward op for supported `mha_h100` variants.
    Returns `(dQ, dK, dV)` as FP32 tensors. -/
def mhaBwd {seq kvBlocks : UInt64} [Variant (device := device) seq kvBlocks]
    (q k v dO : BF16 (device := device) seq) (ctx : FwdCtx (device := device) seq kvBlocks) (stream : UInt64 := 0)
    : IO (F32 (device := device) seq × F32 (device := device) seq × F32 (device := device) seq) := do
  let cfg := { LaunchCfg.default kvBlocks stream with sharedMem := bwdPrepSharedMem }
  let bwdCfg := { cfg with gridX := kvBlocks / 2, gridY := 1, blockX := 384, sharedMem := bwdSweepSharedMem }

  let dVec : L (device := device) kvBlocks := Tensor.Preserving.mulScalar ctx.lOut 0.0
  launchBwdPrep (seq := seq) (kvBlocks := kvBlocks) dO ctx.out dVec cfg

  let dQ : F32 (device := device) seq := Tensor.zeros #[1, 1, seq, 64] false q.device
  let dK : F32 (device := device) seq := Tensor.zeros #[1, 1, seq, 64] false q.device
  let dV : F32 (device := device) seq := Tensor.zeros #[1, 1, seq, 64] false q.device

  Variant.launchBwd (seq := seq) (kvBlocks := kvBlocks)
    q k v dO ctx.lOut dVec dQ dK dV bwdCfg
  pure (dQ, dK, dV)

private def sdpaFwdPortable {seq : UInt64}
    (q k v : BF16 (device := device) seq) (isCausal : Bool) : BF16 (device := device) seq :=
  Tensor.attention q k v 0.0 isCausal

private def sdpaBwdPortable {seq : UInt64}
    (q k v dO : BF16 (device := device) seq) (isCausal : Bool) : IO (F32 (device := device) seq × F32 (device := device) seq × F32 (device := device) seq) := do
  let qLeaf ← Tensor.parameter q
  let kLeaf ← Tensor.parameter k
  let vLeaf ← Tensor.parameter v
  let outRef : BF16 (device := device) seq :=
    Tensor.attention qLeaf kLeaf vLeaf 0.0 isCausal
  Tensor.backward outRef dO
  let dQ : F32 (device := device) seq := Tensor.toFloat32 (← Tensor.gradOf qLeaf)
  let dK : F32 (device := device) seq := Tensor.toFloat32 (← Tensor.gradOf kLeaf)
  let dV : F32 (device := device) seq := Tensor.toFloat32 (← Tensor.gradOf vLeaf)
  pure (dQ, dK, dV)

/-- Portable forward path (arbitrary `seq`, causal/non-causal) backed by SDPA. -/
def mhaFwdPortable {seq : UInt64}
    (q k v : BF16 (device := device) seq) (isCausal : Bool := false) : IO (BF16 (device := device) seq) :=
  pure (sdpaFwdPortable q k v isCausal)

/-- Portable backward path (arbitrary `seq`, causal/non-causal) backed by autograd+SDPA. -/
def mhaBwdPortable {seq : UInt64}
    (q k v dO : BF16 (device := device) seq) (isCausal : Bool := false) : IO (F32 (device := device) seq × F32 (device := device) seq × F32 (device := device) seq) :=
  sdpaBwdPortable q k v dO isCausal

/-- Recommended sequence lengths for current kernel-first experiments. -/
def recommendedKernelSeqs : List UInt64 := [128, 768]

/-- Current kernel coverage predicate used by dispatch helpers. -/
def supportsKernelShape (seq : UInt64) (isCausal : Bool := false) : Bool :=
  match AttentionProblem.currentSpecialization <|
      AttentionProblem.selfAttention seq 64 (.CUDA 0) .BFloat16 isCausal .SM90 with
  | .tkMhaH1002Block | .tkMhaH10012Block => true
  | .portable | .tkMhaH100Decode => false

/-- ThunderKittens-style explicit dispatch:
    - use kernels for non-causal `seq=128` and `seq=768`
    - use portable SDPA path for all other shapes/modes.

    The kernel arms re-check the sequence length against the variant's fixed
    tile count: the selector derives its route from the same static shape, so
    the check always succeeds; the legacy erased-shape code passed `seq` to a
    fixed-shape kernel without any evidence. -/
def mhaFwdDispatch {seq : UInt64}
    (q k v : BF16 (device := device) seq) (stream : UInt64 := 0) (isCausal : Bool := false)
    : IO (FwdCtxDispatch (device := device) seq) := do
  let problem := AttentionProblem.ofQKV q k v none 0.0 isCausal none false none .SM90
  match AttentionProblem.currentSpecialization problem with
  | .tkMhaH100Decode =>
      pure { out := sdpaFwdPortable q k v isCausal }
  | .tkMhaH1002Block =>
      if h : seq = 128 then
        let castTo : BF16 (device := device) seq → BF16 (device := device) 128 := cast (congrArg (BF16 (device := device)) h)
        let castBack : BF16 (device := device) 128 → BF16 (device := device) seq := cast (congrArg (BF16 (device := device)) h).symm
        let fwdCtx ← mhaFwd (seq := 128) (kvBlocks := 2) (castTo q) (castTo k) (castTo v) stream
        pure { out := castBack fwdCtx.out, selection := .tkMhaH1002Block, lOut2 := some fwdCtx.lOut }
      else
        pure { out := sdpaFwdPortable q k v isCausal }
  | .tkMhaH10012Block =>
      if h : seq = 768 then
        let castTo : BF16 (device := device) seq → BF16 (device := device) 768 := cast (congrArg (BF16 (device := device)) h)
        let castBack : BF16 (device := device) 768 → BF16 (device := device) seq := cast (congrArg (BF16 (device := device)) h).symm
        let fwdCtx ← mhaFwd (seq := 768) (kvBlocks := 12) (castTo q) (castTo k) (castTo v) stream
        pure { out := castBack fwdCtx.out, selection := .tkMhaH10012Block, lOut12 := some fwdCtx.lOut }
      else
        pure { out := sdpaFwdPortable q k v isCausal }
  | .portable =>
      pure { out := sdpaFwdPortable q k v isCausal }

/-- Backward matching `mhaFwdDispatch`:
    - use kernel backward iff matching saved `L (device := device)` state is present
    - otherwise use portable SDPA autograd gradients. -/
def mhaBwdDispatch {seq : UInt64}
    (q k v dO : BF16 (device := device) seq) (ctx : FwdCtxDispatch (device := device) seq)
    (stream : UInt64 := 0) (isCausal : Bool := false)
    : IO (F32 (device := device) seq × F32 (device := device) seq × F32 (device := device) seq) := do
  match ctx.selection with
  | .portable =>
      sdpaBwdPortable q k v dO isCausal
  | .tkMhaH100Decode =>
      sdpaBwdPortable q k v dO isCausal
  | .tkMhaH1002Block =>
      match ctx.lOut2 with
      | some lOut =>
          if h : seq = 128 then
            let castTo : BF16 (device := device) seq → BF16 (device := device) 128 := cast (congrArg (BF16 (device := device)) h)
            let castBack : F32 (device := device) 128 → F32 (device := device) seq := cast (congrArg (F32 (device := device)) h).symm
            let kernelCtx : FwdCtx (device := device) 128 2 := { out := castTo ctx.out, lOut := lOut }
            let (dQ, dK, dV) ← mhaBwd (seq := 128) (kvBlocks := 2)
              (castTo q) (castTo k) (castTo v) (castTo dO) kernelCtx stream
            pure (castBack dQ, castBack dK, castBack dV)
          else
            sdpaBwdPortable q k v dO false
      | none =>
          sdpaBwdPortable q k v dO false
  | .tkMhaH10012Block =>
      match ctx.lOut12 with
      | some lOut =>
          if h : seq = 768 then
            let castTo : BF16 (device := device) seq → BF16 (device := device) 768 := cast (congrArg (BF16 (device := device)) h)
            let castBack : F32 (device := device) 768 → F32 (device := device) seq := cast (congrArg (F32 (device := device)) h).symm
            let kernelCtx : FwdCtx (device := device) 768 12 := { out := castTo ctx.out, lOut := lOut }
            let (dQ, dK, dV) ← mhaBwd (seq := 768) (kvBlocks := 12)
              (castTo q) (castTo k) (castTo v) (castTo dO) kernelCtx stream
            pure (castBack dQ, castBack dK, castBack dV)
          else
            sdpaBwdPortable q k v dO false
      | none =>
          sdpaBwdPortable q k v dO false

/-- Main typed forward entrypoint for experimental integration.
    Dispatches to the best available kernel variant for `(seq, isCausal)` and
    otherwise uses the portable SDPA path. -/
@[inline] def mhaFwdMain {seq : UInt64}
    (q k v : BF16 (device := device) seq) (stream : UInt64 := 0) (isCausal : Bool := false)
    : IO (FwdCtxDispatch (device := device) seq) :=
  mhaFwdDispatch (seq := seq) q k v stream isCausal

/-- Main typed backward entrypoint paired with `mhaFwdMain`. -/
@[inline] def mhaBwdMain {seq : UInt64}
    (q k v dO : BF16 (device := device) seq) (ctx : FwdCtxDispatch (device := device) seq)
    (stream : UInt64 := 0) (isCausal : Bool := false)
    : IO (F32 (device := device) seq × F32 (device := device) seq × F32 (device := device) seq) :=
  mhaBwdDispatch (seq := seq) q k v dO ctx stream isCausal

end Tyr.GPU.Ops.MhaH100
