/-
  Tyr/GPU/Kernels/Mamba2.lean

  Mamba2 state-space model forward kernels.
  Based on ThunderKittens patterns:
  - Hillis-Steele prefix sum for cumulative decay
  - Exponential state decay computation
  - Attention with decay masking
  - State accumulation across chunks

  This module owns the source-backed Mamba2 forward surface.
-/

import Tyr.GPU.Kernels.Prelude

namespace Tyr.GPU.Kernels.Mamba2

open Tyr.GPU
open Tyr.GPU.Codegen

private abbrev chunkSize : Nat := 64
private abbrev featDim : Nat := 64

private def toBf16Col {rows cols : Nat}
    (src : RT GpuFloat.Float32 rows cols .Row) : KernelM (RT GpuFloat.BFloat16 rows cols .Col) := do
  let rowTile : RT GpuFloat.BFloat16 rows cols ← allocRT .BFloat16 rows cols
  let colTile : RT GpuFloat.BFloat16 rows cols .Col ← allocRT .BFloat16 rows cols .Col
  convert rowTile src
  swapLayout colTile rowTile
  pure colTile

private def buildMambaDecay
    (a : RV GpuFloat.Float32 chunkSize)
    : KernelM
        (RT GpuFloat.Float32 chunkSize chunkSize .Row ×
         RV GpuFloat.Float32 chunkSize ×
         RV GpuFloat.Float32 chunkSize ×
         RV GpuFloat.Float32 chunkSize) := do
  let aRows : RT GpuFloat.Float32 chunkSize chunkSize ← allocRT .Float32 chunkSize chunkSize
  let aCols : RT GpuFloat.Float32 chunkSize chunkSize ← allocRT .Float32 chunkSize chunkSize
  let aCumsumRows : RT GpuFloat.Float32 chunkSize chunkSize ← allocRT .Float32 chunkSize chunkSize
  let aCumsumCols : RT GpuFloat.Float32 chunkSize chunkSize ← allocRT .Float32 chunkSize chunkSize
  let localDecay : RT GpuFloat.Float32 chunkSize chunkSize ← allocRT .Float32 chunkSize chunkSize

  let firstCol : RT GpuFloat.Float32 chunkSize 1 ← allocRT .Float32 chunkSize 1
  let lastCol : RT GpuFloat.Float32 chunkSize 1 ← allocRT .Float32 chunkSize 1
  let prefixVec : RV GpuFloat.Float32 chunkSize ← allocRV .Float32 chunkSize
  let totalDecay : RV GpuFloat.Float32 chunkSize ← allocRV .Float32 chunkSize
  let qDecay : RV GpuFloat.Float32 chunkSize ← allocRV .Float32 chunkSize
  let kDecay : RV GpuFloat.Float32 chunkSize ← allocRV .Float32 chunkSize

  broadcastCol aRows a
  broadcastRow aCols a
  cumsumCol aCumsumRows aRows
  cumsumRow aCumsumCols aCols

  sub localDecay aCumsumRows aCumsumCols
  exp localDecay localDecay
  makeCausal localDecay localDecay (some 0.0)

  sliceCols firstCol aCumsumRows 0 1
  rowSum prefixVec firstCol
  copyVec qDecay prefixVec
  expVec qDecay qDecay

  sliceCols lastCol aCumsumCols 63 1
  rowSum totalDecay lastCol
  subVec kDecay totalDecay prefixVec
  expVec kDecay kDecay
  expVec totalDecay totalDecay

  pure (localDecay, qDecay, kDecay, totalDecay)

/-! ## Mamba2 Forward Kernel

The Mamba2 architecture uses selective state spaces with:
1. Per-position decay factors (A vector)
2. Input-dependent state updates
3. Causal attention with exponential decay

Key computation flow:
1. Compute cumulative sum of decay factors (log-space)
2. Convert to decay matrix: decay[i,j] = exp(cumsum[i] - cumsum[j])
3. Apply causal mask to decay matrix
4. Compute attention with decay: O = softmax(Q @ K^T * decay) @ V
5. Update running state: KV_state = KV_state * total_decay + K^T @ V
-/

/-- Source-faithful Mamba2 forward surface aligned with
`external/ThunderKittens/kernels/mamba2/mamba2.cu`. -/
@[gpu_kernel .SM90]
def mamba2Fwd (Q_ptr : GPtr GpuFloat.BFloat16) (K_ptr : GPtr GpuFloat.BFloat16)
    (V_ptr : GPtr GpuFloat.BFloat16) (A_ptr : GPtr GpuFloat.Float32)
    (O_ptr : GPtr GpuFloat.BFloat16)
    (seq_len : KVal UInt64) (head_dim : KVal UInt64) : KernelM Unit := do
  let _ := (seq_len, head_dim)
  comment "=== Mamba2 Forward Pass ==="
  comment "ThunderKittens lcsf pipeline compressed into a typed chunk/state shell"
  comment "The exact producer/consumer CTA choreography is flattened to one logical (batch, head) recurrence per kernel instance."

  let zero ← constIntVal 0 "mamba2_zero"
  let headIdx : KVal UInt32 := ⟨← getBlockIdxX, "mamba2_head_idx"⟩
  let batchIdx : KVal UInt32 := ⟨← getBlockIdxY, "mamba2_batch_idx"⟩
  let totalRows ← layoutRows K_ptr "mamba2_total_rows"
  let tileRows ← constIntVal 64 "mamba2_tile_rows"
  let numChunks ← scalarDivVal totalRows tileRows "mamba2_num_chunks"

  let qShared : ST GpuFloat.BFloat16 chunkSize featDim ← allocST .BFloat16 chunkSize featDim
  let kShared : ST GpuFloat.BFloat16 chunkSize featDim ← allocST .BFloat16 chunkSize featDim
  let vShared : ST GpuFloat.BFloat16 chunkSize featDim .Col ← allocST .BFloat16 chunkSize featDim .Col
  let oShared : ST GpuFloat.BFloat16 chunkSize featDim ← allocST .BFloat16 chunkSize featDim
  let aShared : SV GpuFloat.Float32 chunkSize ← allocSV .Float32 chunkSize

  let q : RT GpuFloat.BFloat16 chunkSize featDim ← allocRT .BFloat16 chunkSize featDim
  let k : RT GpuFloat.BFloat16 chunkSize featDim ← allocRT .BFloat16 chunkSize featDim
  let v : RT GpuFloat.BFloat16 chunkSize featDim .Col ← allocRT .BFloat16 chunkSize featDim .Col
  let qF : RT GpuFloat.Float32 chunkSize featDim ← allocRT .Float32 chunkSize featDim
  let kF : RT GpuFloat.Float32 chunkSize featDim ← allocRT .Float32 chunkSize featDim
  let a : RV GpuFloat.Float32 chunkSize ← allocRV .Float32 chunkSize
  let state : RT GpuFloat.Float32 featDim featDim ← zeroRT .Float32 featDim featDim

  for chunkIdx in kvrange 0 numChunks do
    let qkCoord := makeRTileCoord batchIdx.id zero.id chunkIdx.id zero.id
    let vCoord := makeRTileCoord batchIdx.id headIdx.id chunkIdx.id zero.id
    let aCoord := makeRTileCoord batchIdx.id headIdx.id zero.id chunkIdx.id
    let out : RT GpuFloat.Float32 chunkSize featDim ← zeroRT .Float32 chunkSize featDim
    let outBf16 : RT GpuFloat.BFloat16 chunkSize featDim ← allocRT .BFloat16 chunkSize featDim

    loadGlobal qShared Q_ptr qkCoord
    loadGlobal kShared K_ptr qkCoord
    loadGlobal vShared V_ptr vCoord
    loadVecGlobal aShared A_ptr aCoord
    sync

    load q qShared
    load k kShared
    load v vShared
    loadVec a aShared
    convert qF q
    convert kF k

    let (localDecay, qDecay, kDecay, totalDecay) ← buildMambaDecay a

    comment "Local decayed causal chunk contribution"
    let localScores : RT GpuFloat.Float32 chunkSize featDim ← zeroRT .Float32 chunkSize featDim
    let localScoresBf16 : RT GpuFloat.BFloat16 chunkSize featDim ← allocRT .BFloat16 chunkSize featDim
    mmaT localScores q k localScores
    mul localScores localScores localDecay
    convert localScoresBf16 localScores
    mma out localScoresBf16 v out

    comment "Recurrent state contribution Q @ KV_prev with per-row decay"
    let qDecayed : RT GpuFloat.Float32 chunkSize featDim ← allocRT .Float32 chunkSize featDim
    let qDecayedBf16 : RT GpuFloat.BFloat16 chunkSize featDim ← allocRT .BFloat16 chunkSize featDim
    let stateBf16Col ← toBf16Col state
    mulCol qDecayed qF qDecay
    convert qDecayedBf16 qDecayed
    mma out qDecayedBf16 stateBf16Col out

    convert outBf16 out
    store oShared outBf16
    storeGlobal O_ptr oShared vCoord

    comment "Recurrent K^T V state update with final-decay carry"
    let kDecayed : RT GpuFloat.Float32 chunkSize featDim ← allocRT .Float32 chunkSize featDim
    let kDecayedBf16 : RT GpuFloat.BFloat16 chunkSize featDim ← allocRT .BFloat16 chunkSize featDim
    let kDecayedCol : RT GpuFloat.BFloat16 chunkSize featDim .Col ← allocRT .BFloat16 chunkSize featDim .Col
    let updatedState : RT GpuFloat.Float32 featDim featDim ← allocRT .Float32 featDim featDim
    mulCol state state totalDecay
    mulCol kDecayed kF kDecay
    convert kDecayedBf16 kDecayed
    swapLayout kDecayedCol kDecayedBf16
    mmaAtB updatedState kDecayedCol v state
    copy state updatedState
    sync

end Tyr.GPU.Kernels.Mamba2
