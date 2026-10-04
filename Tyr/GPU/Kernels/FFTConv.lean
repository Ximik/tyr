/-
  Tyr/GPU/Kernels/FFTConv.lean

  FFT-based convolution kernels aligned to the structure of
  `external/ThunderKittens/kernels/fftconv/fftconv_pc.cu`.

  This module now has a canonical surface:

  - `Tyr.GPU.Kernels.tkFFTConvPC1024` mirrors the persistent-cache
    producer/consumer flow of the ThunderKittens kernel as closely as the Lean
    DSL can express today.
  - `Tyr.GPU.Kernels.tkFFTConvNonPC64` mirrors the non-persistent
    single-stage flow of `fftconv_non_pc.cu`.

  The legacy names in `Tyr.GPU.Kernels.FFTConv` are retained as compatibility
  aliases to the canonical kernel.

  Current DSL limitation:

  - The vendored CUDA kernel models complex global layouts for `f`, `finv`,
    `tw`, `twinv_t`, and `kf`. The Lean DSL still lacks a convenient complex
    global-layout surface, so those factors remain modeled as long-resident
    shared-memory inputs that are loaded into registers once per head.
-/

import Tyr.GPU.Kernels.Prelude

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen

private def asCol {dtype : GpuFloat} {rows cols : Nat}
    (src : RT dtype rows cols .Row) : KernelM (RT dtype rows cols .Col) := do
  let dst : RT dtype rows cols .Col ← allocRT dtype rows cols .Col
  swapLayout dst src
  pure dst

private def complexAsCol {dtype : GpuFloat} {rows cols : Nat}
    (src : CRT dtype rows cols .Row) : KernelM (CRT dtype rows cols .Col) := do
  let dst : CRT dtype rows cols .Col ← allocCRT dtype rows cols .Col
  complexSwapLayout dst src
  pure dst

private def fftConvTileCore
    (x : RT GpuFloat.BFloat16 64 64 .Row)
    (mmaReg : CRT GpuFloat.Float32 64 64 .Row)
    (accum : CRT GpuFloat.BFloat16 64 64 .Row)
    (tmp : CRT GpuFloat.BFloat16 64 64 .Row)
    (out : RT GpuFloat.BFloat16 64 64 .Row)
    (scratchTmp : CST GpuFloat.BFloat16 64 64 .Row)
    (f : CRT GpuFloat.BFloat16 64 64 .Row)
    (fInv : CRT GpuFloat.BFloat16 64 64 .Row)
    (fCol : CRT GpuFloat.BFloat16 64 64 .Col)
    (fInvCol : CRT GpuFloat.BFloat16 64 64 .Col)
    (tw : CRT GpuFloat.BFloat16 64 64 .Row)
    (twinvT : CRT GpuFloat.BFloat16 64 64 .Row)
    (kf : CRT GpuFloat.BFloat16 64 64 .Row)
    : KernelM Unit := do
  let xCol ← asCol x

  comment "X = F^T @ X for real input"
  let zeroReal : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let zeroImag : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  mma mmaReg.real f.real xCol zeroReal
  mma mmaReg.imag f.imag xCol zeroImag
  mmaCommitGroup
  mmaAsyncWait 0
  convert accum.real mmaReg.real
  convert accum.imag mmaReg.imag

  comment "Apply twiddle, spectral filter, and inverse factors"
  complexMul accum accum tw

  let zeroComplex0 : CRT GpuFloat.Float32 64 64 ← zeroCRT .Float32 64 64
  complexMma mmaReg accum fCol zeroComplex0
  mmaCommitGroup
  mmaAsyncWait 0
  convert accum.real mmaReg.real
  convert accum.imag mmaReg.imag

  complexMul accum accum kf

  let zeroComplex1 : CRT GpuFloat.Float32 64 64 ← zeroCRT .Float32 64 64
  complexMma mmaReg accum fInvCol zeroComplex1
  mmaCommitGroup
  mmaAsyncWait 0
  convert accum.real mmaReg.real
  convert accum.imag mmaReg.imag

  complexMul accum accum twinvT

  comment "Store/reload through scratch for the final AtB-style inverse transform"
  storeComplex scratchTmp accum
  sync
  loadComplex tmp scratchTmp
  let tmpCol ← complexAsCol tmp

  let zeroComplex2 : CRT GpuFloat.Float32 64 64 ← zeroCRT .Float32 64 64
  complexMma mmaReg fInv tmpCol zeroComplex2
  mmaCommitGroup
  mmaAsyncWait 0

  convert out mmaReg.real

/-- Canonical ThunderKittens-aligned FFTConv surface for the 1024-sequence path.

This keeps the source-backed shape of `fftconv_pc.cu`:

- long-resident `f`, `finv`, `tw`, `twinv_t`
- per-head `kf` reload
- producer/consumer split across warp groups
- persistent looping over batch tiles for one logical head

The input/output tiles are modeled explicitly as global-memory pointers; the
FFT factors remain abstract shared-memory inputs until the DSL grows complex
global layout descriptors. -/
@[gpu_kernel .SM90]
def tkFFTConvPC1024
    (x_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16) : KernelM Unit := do
  comment "ThunderKittens fftconv_pc.cu: persistent producer/consumer FFTConv"

  let batchTilesPerHead : Nat := 8

  let zeroIdx ← freshVar
  emit (.constInt zeroIdx 0)
  let headIdx ← getBlockIdxX

  let xStage : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let oStage : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  let fShared : CST GpuFloat.BFloat16 64 64 ← allocCST .BFloat16 64 64
  let fInvShared : CST GpuFloat.BFloat16 64 64 ← allocCST .BFloat16 64 64
  let twShared : CST GpuFloat.BFloat16 64 64 ← allocCST .BFloat16 64 64
  let twinvTShared : CST GpuFloat.BFloat16 64 64 ← allocCST .BFloat16 64 64
  let kfShared : CST GpuFloat.BFloat16 64 64 ← allocCST .BFloat16 64 64
  let scratchTmp : CST GpuFloat.BFloat16 64 64 ← allocCST .BFloat16 64 64

  let f : CRT GpuFloat.BFloat16 64 64 ← allocCRT .BFloat16 64 64
  let fInv : CRT GpuFloat.BFloat16 64 64 ← allocCRT .BFloat16 64 64
  let tw : CRT GpuFloat.BFloat16 64 64 ← allocCRT .BFloat16 64 64
  let twinvT : CRT GpuFloat.BFloat16 64 64 ← allocCRT .BFloat16 64 64
  let kf : CRT GpuFloat.BFloat16 64 64 ← allocCRT .BFloat16 64 64

  let x : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let mmaReg : CRT GpuFloat.Float32 64 64 ← zeroCRT .Float32 64 64
  let accum : CRT GpuFloat.BFloat16 64 64 ← allocCRT .BFloat16 64 64
  let tmp : CRT GpuFloat.BFloat16 64 64 ← allocCRT .BFloat16 64 64

  let inputConsumed ← allocSemaphore
  let inputReady ← allocSemaphore
  initSemaphore inputConsumed 1
  initSemaphore inputReady 0

  comment "Long-resident FFT factors stay live across all batch tiles for this head"
  loadComplex f fShared
  loadComplex fInv fInvShared
  loadComplex tw twShared
  loadComplex twinvT twinvTShared
  let fCol ← complexAsCol f
  let fInvCol ← complexAsCol fInv

  comment "Reload only the per-head frequency-domain filter"
  loadComplex kf kfShared

  ifWarpGroup 0 do
    comment "Producer warp group: stream batch tiles for the current head"
    for batchIdx in krange 0 batchTilesPerHead do
      let tileCoord : RTileCoord :=
        { b := batchIdx.id, d := headIdx, r := zeroIdx, c := zeroIdx }
      waitSemaphore inputConsumed
      loadGlobal xStage x_ptr tileCoord
      sync
      arriveSemaphore inputReady 1

  ifWarpGroup 1 do
    comment "Consumer warp group: FFT pipeline with persistent factors"
    for batchIdx in krange 0 batchTilesPerHead do
      let tileCoord : RTileCoord :=
        { b := batchIdx.id, d := headIdx, r := zeroIdx, c := zeroIdx }
      waitSemaphore inputReady
      load x xStage
      fftConvTileCore x mmaReg accum tmp out scratchTmp f fInv fCol fInvCol tw twinvT kf
      store oStage out
      sync
      storeGlobal o_ptr oStage tileCoord
      arriveSemaphore inputConsumed 1

/-- Canonical ThunderKittens-aligned non-persistent FFTConv surface.

This mirrors the structure of `fftconv_non_pc.cu`: one logical tile per launch,
no persistent producer/consumer loop, and direct global-memory staging for the
input/output tile while reusing the same FFT/filter core as the persistent
variant. The complex factor globals are still modeled as long-resident shared
inputs because the Lean DSL does not yet expose complex global layouts. -/
@[gpu_kernel .SM90]
def tkFFTConvNonPC64
    (x_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16) : KernelM Unit := do
  comment "ThunderKittens fftconv_non_pc.cu: direct non-persistent FFTConv"

  let coord ← blockCoord2D

  let xStage : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let oStage : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  let fShared : CST GpuFloat.BFloat16 64 64 ← allocCST .BFloat16 64 64
  let fInvShared : CST GpuFloat.BFloat16 64 64 ← allocCST .BFloat16 64 64
  let twShared : CST GpuFloat.BFloat16 64 64 ← allocCST .BFloat16 64 64
  let twinvTShared : CST GpuFloat.BFloat16 64 64 ← allocCST .BFloat16 64 64
  let kfShared : CST GpuFloat.BFloat16 64 64 ← allocCST .BFloat16 64 64
  let scratchTmp : CST GpuFloat.BFloat16 64 64 ← allocCST .BFloat16 64 64

  let f : CRT GpuFloat.BFloat16 64 64 ← allocCRT .BFloat16 64 64
  let fInv : CRT GpuFloat.BFloat16 64 64 ← allocCRT .BFloat16 64 64
  let tw : CRT GpuFloat.BFloat16 64 64 ← allocCRT .BFloat16 64 64
  let twinvT : CRT GpuFloat.BFloat16 64 64 ← allocCRT .BFloat16 64 64
  let kf : CRT GpuFloat.BFloat16 64 64 ← allocCRT .BFloat16 64 64

  let x : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let mmaReg : CRT GpuFloat.Float32 64 64 ← zeroCRT .Float32 64 64
  let accum : CRT GpuFloat.BFloat16 64 64 ← allocCRT .BFloat16 64 64
  let tmp : CRT GpuFloat.BFloat16 64 64 ← allocCRT .BFloat16 64 64

  comment "Non-persistent path: load factors and the current input tile directly"
  loadComplex f fShared
  loadComplex fInv fInvShared
  loadComplex tw twShared
  loadComplex twinvT twinvTShared
  let fCol ← complexAsCol f
  let fInvCol ← complexAsCol fInv
  loadComplex kf kfShared

  loadGlobal xStage x_ptr coord
  sync
  load x xStage
  fftConvTileCore x mmaReg accum tmp out scratchTmp f fInv fCol fInvCol tw twinvT kf
  store oStage out
  sync
  storeGlobal o_ptr oStage coord

end Tyr.GPU.Kernels
