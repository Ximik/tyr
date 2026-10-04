import Tyr.GPU.Kernels.Prelude
import Tyr.GPU.Kernels.Support

/-!
  Tyr/GPU/Kernels/LinearAttn.lean

  ThunderKittens-shaped decayed linear attention kernels.

  This module now mirrors the actual structure of
  `external/ThunderKittens/kernels/linear_attention/linear_attention.cu` much
  more closely:

  - `Q/K` are `64x128` chunk tiles,
  - `V/O` are `64x128`,
  - recurrent state is two `64x128` `KV` halves,
  - the kernel builds `q_decay`, `k_decay`, and `block_decay` from a runtime
    slope scalar instead of taking pre-expanded decay tables,
  - the output is the sum of the local masked-decayed path and the recurrent
    `Q @ KV_prev` path.

  The one remaining compression is launch arithmetic: the Lean DSL still models
  one logical chunk pipeline per kernel instance rather than the exact
  multi-worker CTA packing from the CUDA source.
-/

namespace Tyr.GPU.Kernels.LinearAttn

open Tyr.GPU
open Tyr.GPU.Codegen

private abbrev chunkSize : Nat := 64
private abbrev featDim : Nat := 128
private abbrev halfFeat : Nat := 64
private abbrev numChunks : Nat := 16

/-- Load one 64-token chunk of Q/K/V tiles. -/
private def loadQKVChunk
    (qShared : ST GpuFloat.BFloat16 chunkSize featDim .Row)
    (kShared : ST GpuFloat.BFloat16 chunkSize featDim .Row)
    (vShared : ST GpuFloat.BFloat16 chunkSize featDim .Col)
    (Q_ptr : GPtr GpuFloat.BFloat16)
    (K_ptr : GPtr GpuFloat.BFloat16)
    (V_ptr : GPtr GpuFloat.BFloat16)
    (coord : RTileCoord) : KernelM Unit := do
  loadGlobal qShared Q_ptr coord
  loadGlobal kShared K_ptr coord
  loadGlobal vShared V_ptr coord

/-- Convert an FP32 row-major tile into BF16 column-major form for `mma`. -/
private def toBf16Col {rows cols : Nat}
    (src : RT GpuFloat.Float32 rows cols .Row) : KernelM (RT GpuFloat.BFloat16 rows cols .Col) := do
  let rowTile : RT GpuFloat.BFloat16 rows cols ← allocRT .BFloat16 rows cols
  let colTile : RT GpuFloat.BFloat16 rows cols .Col ← allocRT .BFloat16 rows cols .Col
  convert rowTile src
  swapLayout colTile rowTile
  pure colTile

/-- Apply the local masked-decayed `QK^T V` path for one chunk. -/
private def accumulateLocalChunk
    (out : RT GpuFloat.Float32 chunkSize featDim)
    (q : RT GpuFloat.BFloat16 chunkSize featDim)
    (k : RT GpuFloat.BFloat16 chunkSize featDim)
    (v : RT GpuFloat.BFloat16 chunkSize featDim .Col)
    (qDecay : RV GpuFloat.Float32 chunkSize)
    (kDecay : RV GpuFloat.Float32 chunkSize) : KernelM Unit := do
  let rawScores : RT GpuFloat.Float32 chunkSize chunkSize ← zeroRT .Float32 chunkSize chunkSize
  let maskedScores : RT GpuFloat.Float32 chunkSize chunkSize ← allocRT .Float32 chunkSize chunkSize
  let maskedScoresBf : RT GpuFloat.BFloat16 chunkSize chunkSize ← allocRT .BFloat16 chunkSize chunkSize
  mmaT rawScores q k rawScores
  Support.applyLinearAttnDecayMask maskedScores rawScores qDecay kDecay
  convert maskedScoresBf maskedScores
  mma out maskedScoresBf v out

/-- Apply one recurrent `Q_half @ KV_prev_half` contribution. -/
private def accumulateRecurrentHalf
    (out : RT GpuFloat.Float32 chunkSize featDim)
    (qHalf : RT GpuFloat.Float32 chunkSize halfFeat)
    (stateHalf : RT GpuFloat.Float32 halfFeat featDim)
    (qDecay : RV GpuFloat.Float32 chunkSize) : KernelM Unit := do
  let qHalfDecayed : RT GpuFloat.Float32 chunkSize halfFeat ← allocRT .Float32 chunkSize halfFeat
  let qHalfDecayedBf : RT GpuFloat.BFloat16 chunkSize halfFeat ← allocRT .BFloat16 chunkSize halfFeat
  let stateHalfBfCol ← toBf16Col stateHalf
  mulCol qHalfDecayed qHalf qDecay
  convert qHalfDecayedBf qHalfDecayed
  mma out qHalfDecayedBf stateHalfBfCol out

/-- Update one `64x128` recurrent `KV` half with block decay plus `K^T V`. -/
private def updateStateHalf
    (stateHalf : RT GpuFloat.Float32 halfFeat featDim)
    (kHalf : RT GpuFloat.Float32 chunkSize halfFeat)
    (v : RT GpuFloat.BFloat16 chunkSize featDim .Col)
    (kDecay : RV GpuFloat.Float32 chunkSize)
    (stateDecay : RV GpuFloat.Float32 halfFeat) : KernelM Unit := do
  let kHalfDecayed : RT GpuFloat.Float32 chunkSize halfFeat ← allocRT .Float32 chunkSize halfFeat
  let kHalfDecayedBf : RT GpuFloat.BFloat16 chunkSize halfFeat ← allocRT .BFloat16 chunkSize halfFeat
  let kHalfDecayedCol : RT GpuFloat.BFloat16 chunkSize halfFeat .Col ← allocRT .BFloat16 chunkSize halfFeat .Col
  let updated : RT GpuFloat.Float32 halfFeat featDim ← allocRT .Float32 halfFeat featDim

  mulCol stateHalf stateHalf stateDecay
  mulCol kHalfDecayed kHalf kDecay
  convert kHalfDecayedBf kHalfDecayed
  swapLayout kHalfDecayedCol kHalfDecayedBf
  mmaAtB updated kHalfDecayedCol v stateHalf
  copy stateHalf updated

/-- Source-faithful decayed recurrent linear attention forward.

`slope` is the per-head decay parameter used by the vendored ThunderKittens
kernel. The auxiliary `kv_history_*` outputs store the recurrent state before
each chunk update so the backward kernel can replay the exact reverse scan. -/
@[gpu_kernel .SM90]
def linearAttnFwd
    (Q_ptr : GPtr GpuFloat.BFloat16)
    (K_ptr : GPtr GpuFloat.BFloat16)
    (V_ptr : GPtr GpuFloat.BFloat16)
    (O_ptr : GPtr GpuFloat.BFloat16)
    (kv_history_top_ptr : GPtr GpuFloat.Float32)
    (kv_history_bottom_ptr : GPtr GpuFloat.Float32)
    (kv_final_top_ptr : GPtr GpuFloat.Float32)
    (kv_final_bottom_ptr : GPtr GpuFloat.Float32)
    (slope : KVal Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  comment "=== Decayed Linear Attention Forward ==="
  comment "ThunderKittens two-path chunk recurrence: local masked-decayed QK^T V plus recurrent Q @ KV_prev"

  let coord ← blockCoord2D
  let (qDecay, kDecay, stateDecay) ← Support.buildLinearAttnDecayVectors slope

  let q : RT GpuFloat.BFloat16 chunkSize featDim ← allocRT .BFloat16 chunkSize featDim
  let k : RT GpuFloat.BFloat16 chunkSize featDim ← allocRT .BFloat16 chunkSize featDim
  let v : RT GpuFloat.BFloat16 chunkSize featDim .Col ← allocRT .BFloat16 chunkSize featDim .Col
  let qF : RT GpuFloat.Float32 chunkSize featDim ← allocRT .Float32 chunkSize featDim
  let kF : RT GpuFloat.Float32 chunkSize featDim ← allocRT .Float32 chunkSize featDim
  let out : RT GpuFloat.Float32 chunkSize featDim ← zeroRT .Float32 chunkSize featDim
  let outBf : RT GpuFloat.BFloat16 chunkSize featDim ← allocRT .BFloat16 chunkSize featDim
  let stateTop : RT GpuFloat.Float32 halfFeat featDim ← zeroRT .Float32 halfFeat featDim
  let stateBottom : RT GpuFloat.Float32 halfFeat featDim ← zeroRT .Float32 halfFeat featDim

  let qShared : ST GpuFloat.BFloat16 chunkSize featDim ← allocST .BFloat16 chunkSize featDim
  let kShared : ST GpuFloat.BFloat16 chunkSize featDim ← allocST .BFloat16 chunkSize featDim
  let vShared : ST GpuFloat.BFloat16 chunkSize featDim .Col ← allocST .BFloat16 chunkSize featDim .Col
  let outShared : ST GpuFloat.BFloat16 chunkSize featDim ← allocST .BFloat16 chunkSize featDim
  let stateTopShared : ST GpuFloat.Float32 halfFeat featDim ← allocST .Float32 halfFeat featDim
  let stateBottomShared : ST GpuFloat.Float32 halfFeat featDim ← allocST .Float32 halfFeat featDim

  for chunkIdx in krange 0 numChunks do
    let chunkCoord := coord.withRow chunkIdx.id
    loadQKVChunk qShared kShared vShared Q_ptr K_ptr V_ptr chunkCoord
    sync
    load q qShared
    load k kShared
    load v vShared
    convert qF q
    convert kF k

    comment "Checkpoint recurrent KV state before the chunk update"
    store stateTopShared stateTop
    store stateBottomShared stateBottom
    storeGlobal kv_history_top_ptr stateTopShared chunkCoord
    storeGlobal kv_history_bottom_ptr stateBottomShared chunkCoord

    zero out
    accumulateLocalChunk out q k v qDecay kDecay

    let (qLeft, qRight) ← Support.splitCols qF
    let (kLeft, kRight) ← Support.splitCols kF
    accumulateRecurrentHalf out qLeft stateTop qDecay
    accumulateRecurrentHalf out qRight stateBottom qDecay

    convert outBf out
    store outShared outBf
    storeGlobal O_ptr outShared chunkCoord

    updateStateHalf stateTop kLeft v kDecay stateDecay
    updateStateHalf stateBottom kRight v kDecay stateDecay
    sync

  store stateTopShared stateTop
  store stateBottomShared stateBottom
  storeGlobal kv_final_top_ptr stateTopShared coord
  storeGlobal kv_final_bottom_ptr stateBottomShared coord

/-- Canonical ThunderKittens-aligned name. -/
abbrev tkLinearAttnFwd := linearAttnFwd

end Tyr.GPU.Kernels.LinearAttn
