/-
  Tyr/GPU/Kernels/Hedgehog.lean

  Hedgehog hybrid attention kernels aligned to
  `external/git/ThunderKittens/kernels/hedgehog/hedgehog.cu`.

  This module now provides a canonical surface:

  - `Tyr.GPU.Kernels.tkHedgehogFwd` tracks the source-backed chunk/state flow:
    long-resident feature maps, previous/current sliding-window blocks,
    recurrent `k_state` / `kv_state` accumulation, and final state writeout.

  Current DSL limitation:

  - The vendored CUDA kernel uses a double-buffered Q path, a 3-ring K/V path,
    and per-head scalar `alpha` / `beta`. The Lean DSL currently expresses this
    as a previous/current sliding-window staging with row-broadcast mixing
    vectors, which preserves the state/data flow even though the warpgroup-level
    schedule is compressed.
-/

import Tyr.GPU.Kernels.Prelude
import Tyr.GPU.Codegen.Macros

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen

private def asCol {dtype : GpuFloat} {rows cols : Nat}
    (src : RT dtype rows cols .Row) : KernelM (RT dtype rows cols .Col) := do
  let dst : RT dtype rows cols .Col ← allocRT dtype rows cols .Col
  swapLayout dst src
  pure dst

private def softmaxFeatureMapInplace {rows cols : Nat}
    (tile : RT GpuFloat.Float32 rows cols .Row) : KernelM Unit := do
  let rowMaxVec : RV GpuFloat.Float32 rows ← allocRV .Float32 rows
  let rowSumVec : RV GpuFloat.Float32 rows ← allocRV .Float32 rows
  rowMax rowMaxVec tile
  subCol tile tile rowMaxVec
  exp2 tile tile
  rowSum rowSumVec tile
  divCol tile tile rowSumVec

private def onlineSoftmaxExp2 {rows cols outCols : Nat}
    (scores : RT GpuFloat.Float32 rows cols .Row)
    (output : RT GpuFloat.Float32 rows outCols .Row)
    (state : SoftmaxState GpuFloat.Float32 rows) : KernelM Unit := do
  copyVec state.prevMax state.rowMax
  rowMaxAccum state.rowMax scores state.rowMax
  subVec state.scale state.prevMax state.rowMax
  expVec state.scale state.scale
  mulCol output output state.scale
  mulVec state.rowSum state.rowSum state.scale
  subCol scores scores state.rowMax
  exp2 scores scores
  rowSumAccum state.rowSum scores state.rowSum

private def slidingWindowBlock
    (q : RT GpuFloat.BFloat16 64 64 .Row)
    (k : RT GpuFloat.BFloat16 64 64 .Row)
    (v : RT GpuFloat.BFloat16 64 64 .Col)
    (beta : RV GpuFloat.Float32 64)
    (scores : RT GpuFloat.Float32 64 64 .Row)
    (weights : RT GpuFloat.BFloat16 64 64 .Row)
    (output : RT GpuFloat.Float32 64 64 .Row)
    (state : SoftmaxState GpuFloat.Float32 64)
    (causal : Bool) : KernelM Unit := do
  let zeros : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  mmaT scores q k zeros
  if causal then
    makeCausal scores scores (some (-1.0e10))
  mulCol scores scores beta
  onlineSoftmaxExp2 scores output state
  convert weights scores
  mma output weights v output

/-- Canonical ThunderKittens-aligned Hedgehog surface.

This keeps the major source-backed phases from `hedgehog.cu`:

- per-head feature-map loads (`qmap`, `kmap`, mixing vectors)
- chunk loop over `(batch, head)` tiles
- sliding-window path over previous/current K/V blocks
- linear path over recurrent `kv_state` and cumulative `k_state`
- final writeout of output, `kv_state`, and `k_state`

The source kernel's multi-stage producer/consumer schedule is compressed to
previous/current block staging because the Lean DSL does not yet expose the full
ring-buffer/TMA protocol. -/
@[gpu_kernel .SM90]
def tkHedgehogFwd
    (Q_ptr : GPtr GpuFloat.BFloat16)
    (K_ptr : GPtr GpuFloat.BFloat16)
    (V_ptr : GPtr GpuFloat.BFloat16)
    (QMap_ptr : GPtr GpuFloat.BFloat16)
    (KMap_ptr : GPtr GpuFloat.BFloat16)
    (alpha_ptr : GPtr GpuFloat.Float32)
    (beta_ptr : GPtr GpuFloat.Float32)
    (O_ptr : GPtr GpuFloat.BFloat16)
    (k_state_ptr : GPtr GpuFloat.Float32)
    (kv_state_ptr : GPtr GpuFloat.Float32)
    (_batch_size : KVal UInt64)
    (_num_heads : KVal UInt64)
    (_seq_len : KVal UInt64) : KernelM Unit := do
  comment "ThunderKittens hedgehog.cu: chunked sliding + recurrent linear attention"

  let numChunks : Nat := 16

  let batchIdx ← getBlockIdxY
  let headIdx ← getBlockIdxX
  let zeroIdx ← freshVar
  emit (.constInt zeroIdx 0)

  let mapCoord : RTileCoord := { b := zeroIdx, d := headIdx, r := zeroIdx, c := zeroIdx }

  let qShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let kShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let vShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let kvStateShared : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64
  let kStateShared : ST GpuFloat.Float32 1 64 ← allocST .Float32 1 64
  let qMapShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let kMapShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let alphaShared : SV GpuFloat.Float32 64 ← allocSV .Float32 64
  let betaShared : SV GpuFloat.Float32 64 ← allocSV .Float32 64

  let q : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let k : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let v : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let kPrev : RT GpuFloat.BFloat16 64 64 ← zeroRT .BFloat16 64 64
  let vPrev : RT GpuFloat.BFloat16 64 64 .Col ← zeroRT .BFloat16 64 64 .Col

  let qMap : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let kMap : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let alpha : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let beta : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  let qFeat : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let kFeat : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let qFeatBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let kFeatBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let kFeatCol : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let kvState : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let kvStateBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let kvStateCol : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let kState : RV GpuFloat.Float32 64 ← zeroRV .Float32 64
  let kStateRows : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let linearNormTile : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let linearNorm : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let kFeatSum : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  let slidingScores : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let slidingWeights : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let slidingOut : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let slidingState ← allocSoftmaxState .Float32 64

  let linearOut : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let combinedOut : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let combinedNorm : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  comment "Long-resident feature maps and mixing vectors for this head"
  loadGlobal qMapShared QMap_ptr mapCoord
  loadGlobal kMapShared KMap_ptr mapCoord
  loadVecGlobalCoord alphaShared alpha_ptr headIdx
  loadVecGlobalCoord betaShared beta_ptr headIdx
  sync
  load qMap qMapShared
  load kMap kMapShared
  loadVec alpha alphaShared
  loadVec beta betaShared

  comment "Chunk loop: previous/current staging compresses the source 3-ring schedule"
  for chunkIdx in krange 0 numChunks do
    let chunkCoord : RTileCoord :=
      { b := batchIdx, d := headIdx, r := chunkIdx.id, c := zeroIdx }

    loadGlobal qShared Q_ptr chunkCoord
    loadGlobal kShared K_ptr chunkCoord
    loadGlobal vShared V_ptr chunkCoord
    sync
    load q qShared
    load k kShared
    load v vShared

    comment "Sliding-window path across previous and current K/V chunks"
    zero slidingOut
    zeroVec slidingState.rowSum
    copyVec slidingState.prevMax slidingState.rowSum
    copyVec slidingState.scale slidingState.rowSum
    let negInf : RV GpuFloat.Float32 64 ← negInftyRV .Float32 64
    copyVec slidingState.rowMax negInf

    slidingWindowBlock q kPrev vPrev beta slidingScores slidingWeights slidingOut slidingState false
    slidingWindowBlock q k v beta slidingScores slidingWeights slidingOut slidingState true
    divCol slidingOut slidingOut slidingState.rowSum

    comment "Linear path from feature maps and recurrent state"
    let zeroFeatQ : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
    let zeroFeatK : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
    mma qFeat q qMap zeroFeatQ
    mma kFeat k kMap zeroFeatK
    mmaCommitGroup
    mmaAsyncWait 0
    softmaxFeatureMapInplace qFeat
    softmaxFeatureMapInplace kFeat
    mulCol qFeat qFeat alpha
    convert qFeatBf qFeat
    convert kvStateBf kvState
    swapLayout kvStateCol kvStateBf
    let zeroLinear : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
    mma linearOut qFeatBf kvStateCol zeroLinear

    broadcastRow kStateRows kState
    mul linearNormTile qFeat kStateRows
    rowSum linearNorm linearNormTile

    comment "Combine sliding and linear paths with shared normalization"
    add combinedOut slidingOut linearOut
    addVec combinedNorm slidingState.rowSum linearNorm
    divCol combinedOut combinedOut combinedNorm

    convert out combinedOut
    store outShared out
    storeGlobal O_ptr outShared chunkCoord

    comment "Update recurrent K and KV states for the next chunk"
    convert kFeatBf kFeat
    swapLayout kFeatCol kFeatBf
    mmaAtB kvState kFeatCol v kvState
    rowSum kFeatSum kFeat
    addVec kState kState kFeatSum

    copy kPrev k
    copy vPrev v
    sync

  comment "Write final recurrent state tensors"
  store kvStateShared kvState
  storeGlobal kv_state_ptr kvStateShared { b := batchIdx, d := headIdx, r := zeroIdx, c := zeroIdx }

  let kStateTile : RT GpuFloat.Float32 1 64 ← allocRT .Float32 1 64
  broadcastRow kStateTile kState
  store kStateShared kStateTile
  storeGlobal k_state_ptr kStateShared { b := batchIdx, d := headIdx, r := zeroIdx, c := zeroIdx }
end Tyr.GPU.Kernels
