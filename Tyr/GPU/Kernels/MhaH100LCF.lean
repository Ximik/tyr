import Tyr.GPU.Codegen.Macros
import Tyr.GPU.Kernels.Prelude

/-!
# Tyr.GPU.Kernels.MhaH100LCF

ThunderKittens-style load-compute-finish attention kernels based on
`external/git/ThunderKittens/kernels/attention/mha_h100_lcf/mha_h100_lcf.cu`.

The vendored CUDA kernel uses the ThunderKittens `lcf` pipeline template with:

- one stationary 64xD query tile per worker,
- larger streamed KV tiles (192x64 or 128x128),
- online softmax accumulation across the KV stream,
- a multi-worker CTA packing that the current Lean DSL does not model directly.

The Lean surfaces below keep the same tile geometry and benchmark-sized KV
stream lengths from the source (`3072` sequence length, so 16 or 24 KV tiles),
but represent the worker packing as a single logical query tile per kernel
instance. That preserves the source-facing contract without pretending the DSL
already has the full `lcf` template/runtime launch arithmetic.
-/

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen

private def mhaH100LcfFwd
    {headDim kvTileRows numKvTiles : Nat}
    (banner : String)
    (scale : Float)
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64)
    (hHeadDim : headDim % 16 = 0 := by decide)
    (hKvTileRows : kvTileRows % 16 = 0 := by decide)
    : KernelM Unit := do
  let _ := hHeadDim; let _ := hKvTileRows
  comment banner
  comment "ThunderKittens lcf pipeline compressed into a typed single-query-tile shell"
  comment "The exact multi-worker CTA packing is still flattened to one logical query tile per kernel instance."

  let coord ← blockCoord2D

  let qShared : ST GpuFloat.BFloat16 64 headDim ← allocST .BFloat16 64 headDim
  let kShared : ST GpuFloat.BFloat16 kvTileRows headDim ← allocST .BFloat16 kvTileRows headDim
  let vShared : ST GpuFloat.BFloat16 kvTileRows headDim .Col ← allocST .BFloat16 kvTileRows headDim .Col
  let oShared : ST GpuFloat.BFloat16 64 headDim ← allocST .BFloat16 64 headDim

  let q : RT GpuFloat.BFloat16 64 headDim ← allocRT .BFloat16 64 headDim
  let k : RT GpuFloat.BFloat16 kvTileRows headDim ← allocRT .BFloat16 kvTileRows headDim
  let v : RT GpuFloat.BFloat16 kvTileRows headDim .Col ← allocRT .BFloat16 kvTileRows headDim .Col
  let o : RT GpuFloat.Float32 64 headDim ← zeroRT .Float32 64 headDim
  let softmaxState ← allocSoftmaxState .Float32 64

  loadGlobal qShared q_ptr coord
  sync
  load q qShared

  for kvIdx in krange 0 numKvTiles do
    let kvCoord := coord.withRow kvIdx.id
    let scores : RT GpuFloat.Float32 64 kvTileRows ← zeroRT .Float32 64 kvTileRows
    let probs : RT GpuFloat.BFloat16 64 kvTileRows ← allocRT .BFloat16 64 kvTileRows

    loadGlobal kShared k_ptr kvCoord
    loadGlobal vShared v_ptr kvCoord
    sync
    load k kShared
    load v vShared

    mmaT scores q k scores (by decide) hHeadDim hKvTileRows
    scalarMul scores scores scale
    onlineSoftmax scores o softmaxState
    convert probs scores
    mma o probs v o (by decide) hKvTileRows hHeadDim
    sync

  finalizeSoftmax o softmaxState
  let oBf16 : RT GpuFloat.BFloat16 64 headDim ← allocRT .BFloat16 64 headDim
  convert oBf16 o
  store oShared oBf16
  storeGlobal o_ptr oShared coord

/-- ThunderKittens `mha_h100_lcf` benchmark-sized forward surface for `D = 64`.

This matches the source tile geometry:

- `Q`: 64x64
- `K/V`: 192x64
- `3072 / 192 = 16` streamed KV tiles
- scale `1 / sqrt(64) = 0.125`
-/
@[gpu_kernel .SM90]
def tkMhaH100LCFFwd64
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (seq_len : KVal UInt64)
    (head_dim : KVal UInt64) : KernelM Unit := do
  mhaH100LcfFwd
    (headDim := 64)
    (kvTileRows := 192)
    (numKvTiles := 16)
    "=== ThunderKittens mha_h100_lcf forward (D=64, KV tile 192) ==="
    0.125
    q_ptr k_ptr v_ptr o_ptr seq_len head_dim

/-- ThunderKittens `mha_h100_lcf` benchmark-sized forward surface for `D = 128`.

This matches the source tile geometry:

- `Q`: 64x128
- `K/V`: 128x128
- `3072 / 128 = 24` streamed KV tiles
- scale `1 / sqrt(128)`
-/
@[gpu_kernel .SM90]
def tkMhaH100LCFFwd128
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (seq_len : KVal UInt64)
    (head_dim : KVal UInt64) : KernelM Unit := do
  mhaH100LcfFwd
    (headDim := 128)
    (kvTileRows := 128)
    (numKvTiles := 24)
    "=== ThunderKittens mha_h100_lcf forward (D=128, KV tile 128) ==="
    0.08838834764
    q_ptr k_ptr v_ptr o_ptr seq_len head_dim

end Tyr.GPU.Kernels
