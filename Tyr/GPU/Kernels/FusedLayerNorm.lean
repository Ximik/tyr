/-
  Tyr/GPU/Kernels/FusedLayerNorm.lean

  Fused LayerNorm kernels.

  `tkFusedLayerNormResidual1024` is the canonical ThunderKittens-aligned port of
  `external/ThunderKittens/kernels/layernorm/layernorm.cu`.
-/

import Tyr.GPU.Kernels.Prelude

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen
private abbrev rowTileDim : Nat := 16

private def fusedLayerNormResidual1024Bf16Body
    (x_ptr : GPtr GpuFloat.BFloat16)
    (residual_ptr : GPtr GpuFloat.BFloat16)
    (weight_ptr : GPtr GpuFloat.BFloat16)
    (bias_ptr : GPtr GpuFloat.BFloat16)
    (out_ptr : GPtr GpuFloat.BFloat16)
    (out_resid_ptr : GPtr GpuFloat.BFloat16) : KernelM Unit := do
  comment "ThunderKittens-style layernorm + residual on 64x1024 tiles"

  let hiddenTiles : Nat := 16
  let invHidden : Float := 0.0009765625 -- 1 / 1024
  let eps : Float := 1.0e-5

  let coord ← blockCoord2D

  let x ← allocRT .BFloat16 rowTileDim 64
  let residual ← allocRT .BFloat16 rowTileDim 64
  let outResid ← allocRT .BFloat16 rowTileDim 64
  let outResidF ← allocRT .Float32 rowTileDim 64
  let centered ← allocRT .Float32 rowTileDim 64
  let sq ← allocRT .Float32 rowTileDim 64

  let tileShared ← allocST .BFloat16 rowTileDim 64

  let sum ← zeroRV .Float32 rowTileDim
  let squareSum ← zeroRV .Float32 rowTileDim
  let tmpSum ← allocRV .Float32 rowTileDim
  let tmpSquareSum ← allocRV .Float32 rowTileDim
  let mean ← allocRV .Float32 rowTileDim
  let var ← allocRV .Float32 rowTileDim
  let invStd ← allocRV .Float32 rowTileDim

  let weightShared ← allocSV .BFloat16 64
  let biasShared ← allocSV .BFloat16 64
  let weightRV ← allocRV .BFloat16 64
  let biasRV ← allocRV .BFloat16 64
  let weightF ← allocRV .Float32 64
  let biasF ← allocRV .Float32 64

  comment "Pass 1: out_resid and row statistics across full hidden dimension"
  for hiddenIdx in krange 0 hiddenTiles do
    let tileCoord := coord.withCol hiddenIdx.id

    loadGlobal tileShared x_ptr tileCoord
    sync
    load x tileShared

    loadGlobal tileShared residual_ptr tileCoord
    sync
    load residual tileShared

    add outResid x residual
    store tileShared outResid
    sync
    storeGlobal out_resid_ptr tileShared tileCoord

    convert outResidF outResid
    rowSum tmpSum outResidF
    addVec sum sum tmpSum
    mul sq outResidF outResidF
    rowSum tmpSquareSum sq
    addVec squareSum squareSum tmpSquareSum

  scalarMulVec mean sum invHidden
  scalarMulVec var squareSum invHidden
  mulVec tmpSquareSum mean mean
  subVec var var tmpSquareSum
  scalarAddVec var var eps
  rsqrtVec invStd var

  comment "Pass 2: normalize + affine (weight, bias)"
  for hiddenIdx in krange 0 hiddenTiles do
    let tileCoord := coord.withCol hiddenIdx.id

    loadGlobal tileShared out_resid_ptr tileCoord
    sync
    load outResid tileShared
    convert outResidF outResid

    loadVecGlobalCol weightShared weight_ptr tileCoord
    loadVec weightRV weightShared
    convertVec weightF weightRV

    loadVecGlobalCol biasShared bias_ptr tileCoord
    loadVec biasRV biasShared
    convertVec biasF biasRV

    subCol centered outResidF mean
    mulCol centered centered invStd
    mulRow centered centered weightF
    addRow centered centered biasF

    convert outResid centered
    store tileShared outResid
    sync
    storeGlobal out_ptr tileShared tileCoord

private def fusedLayerNormResidual1024F32Body
    (x_ptr : GPtr GpuFloat.Float32)
    (residual_ptr : GPtr GpuFloat.Float32)
    (weight_ptr : GPtr GpuFloat.Float32)
    (bias_ptr : GPtr GpuFloat.Float32)
    (out_ptr : GPtr GpuFloat.Float32)
    (out_resid_ptr : GPtr GpuFloat.Float32) : KernelM Unit := do
  comment "Float32 fused layernorm + residual on 64x1024 tiles"

  let hiddenTiles : Nat := 16
  let invHidden : Float := 0.0009765625 -- 1 / 1024
  let eps : Float := 1.0e-5

  let coord ← blockCoord2D

  let x ← allocRT .Float32 rowTileDim 64
  let residual ← allocRT .Float32 rowTileDim 64
  let outResid ← allocRT .Float32 rowTileDim 64
  let centered ← allocRT .Float32 rowTileDim 64
  let sq ← allocRT .Float32 rowTileDim 64

  let tileShared ← allocST .Float32 rowTileDim 64

  let sum ← zeroRV .Float32 rowTileDim
  let squareSum ← zeroRV .Float32 rowTileDim
  let tmpSum ← allocRV .Float32 rowTileDim
  let tmpSquareSum ← allocRV .Float32 rowTileDim
  let mean ← allocRV .Float32 rowTileDim
  let var ← allocRV .Float32 rowTileDim
  let invStd ← allocRV .Float32 rowTileDim

  let weightShared ← allocSV .Float32 64
  let biasShared ← allocSV .Float32 64
  let weightRV ← allocRV .Float32 64
  let biasRV ← allocRV .Float32 64

  comment "Pass 1: out_resid and row statistics across full hidden dimension"
  for hiddenIdx in krange 0 hiddenTiles do
    let tileCoord := coord.withCol hiddenIdx.id

    loadGlobal tileShared x_ptr tileCoord
    sync
    load x tileShared

    loadGlobal tileShared residual_ptr tileCoord
    sync
    load residual tileShared

    add outResid x residual
    store tileShared outResid
    sync
    storeGlobal out_resid_ptr tileShared tileCoord

    rowSum tmpSum outResid
    addVec sum sum tmpSum
    mul sq outResid outResid
    rowSum tmpSquareSum sq
    addVec squareSum squareSum tmpSquareSum

  scalarMulVec mean sum invHidden
  scalarMulVec var squareSum invHidden
  mulVec tmpSquareSum mean mean
  subVec var var tmpSquareSum
  scalarAddVec var var eps
  rsqrtVec invStd var

  comment "Pass 2: normalize + affine (weight, bias)"
  for hiddenIdx in krange 0 hiddenTiles do
    let tileCoord := coord.withCol hiddenIdx.id

    loadGlobal tileShared out_resid_ptr tileCoord
    sync
    load outResid tileShared

    loadVecGlobalCol weightShared weight_ptr tileCoord
    loadVec weightRV weightShared

    loadVecGlobalCol biasShared bias_ptr tileCoord
    loadVec biasRV biasShared

    subCol centered outResid mean
    mulCol centered centered invStd
    mulRow centered centered weightRV
    addRow centered centered biasRV

    store tileShared centered
    sync
    storeGlobal out_ptr tileShared tileCoord

/-- ThunderKittens-aligned fused residual + layernorm kernel for `d_model = 1024`.

This is the canonical porting surface for `kernels/layernorm/layernorm.cu`.
It keeps the same two-pass structure as the ThunderKittens kernel:

- pass 1 computes `out_resid = x + residual` and full-row statistics
- pass 2 reloads `out_resid`, applies normalization, then affine parameters

The older kernels below remain useful as DSL sketches, but this kernel should be
treated as the authoritative fused-layernorm implementation in Tyr. -/
@[gpu_kernel .SM90]
def tkFusedLayerNormResidual1024
    (x_ptr : GPtr GpuFloat.BFloat16)
    (residual_ptr : GPtr GpuFloat.BFloat16)
    (weight_ptr : GPtr GpuFloat.BFloat16)
    (bias_ptr : GPtr GpuFloat.BFloat16)
    (out_ptr : GPtr GpuFloat.BFloat16)
    (out_resid_ptr : GPtr GpuFloat.BFloat16) : KernelM Unit := do
  fusedLayerNormResidual1024Bf16Body
    x_ptr residual_ptr weight_ptr bias_ptr out_ptr out_resid_ptr

/-- Blackwell-family variant of the canonical bf16 fused layernorm kernel. -/
@[gpu_kernel .SM90]
def tkFusedLayerNormResidual1024Blackwell
    (x_ptr : GPtr GpuFloat.BFloat16)
    (residual_ptr : GPtr GpuFloat.BFloat16)
    (weight_ptr : GPtr GpuFloat.BFloat16)
    (bias_ptr : GPtr GpuFloat.BFloat16)
    (out_ptr : GPtr GpuFloat.BFloat16)
    (out_resid_ptr : GPtr GpuFloat.BFloat16) : KernelM Unit := do
  setFamily .Blackwell
  fusedLayerNormResidual1024Bf16Body
    x_ptr residual_ptr weight_ptr bias_ptr out_ptr out_resid_ptr

/-- Float32 variant of `tkFusedLayerNormResidual1024`.

This keeps the same two-pass structure and launch geometry, but operates on
`f32` inputs and outputs directly instead of silently quantizing at the host
boundary. -/
@[gpu_kernel .SM90]
def tkFusedLayerNormResidual1024F32
    (x_ptr : GPtr GpuFloat.Float32)
    (residual_ptr : GPtr GpuFloat.Float32)
    (weight_ptr : GPtr GpuFloat.Float32)
    (bias_ptr : GPtr GpuFloat.Float32)
    (out_ptr : GPtr GpuFloat.Float32)
    (out_resid_ptr : GPtr GpuFloat.Float32) : KernelM Unit := do
  fusedLayerNormResidual1024F32Body
    x_ptr residual_ptr weight_ptr bias_ptr out_ptr out_resid_ptr

/-- Blackwell-family variant of the f32 fused layernorm kernel. -/
@[gpu_kernel .SM90]
def tkFusedLayerNormResidual1024F32Blackwell
    (x_ptr : GPtr GpuFloat.Float32)
    (residual_ptr : GPtr GpuFloat.Float32)
    (weight_ptr : GPtr GpuFloat.Float32)
    (bias_ptr : GPtr GpuFloat.Float32)
    (out_ptr : GPtr GpuFloat.Float32)
    (out_resid_ptr : GPtr GpuFloat.Float32) : KernelM Unit := do
  setFamily .Blackwell
  fusedLayerNormResidual1024F32Body
    x_ptr residual_ptr weight_ptr bias_ptr out_ptr out_resid_ptr

/-- Row-parallel fixed-shape FP32 route for `[64, 1024]`. One 256-thread CTA
    owns one row and each thread retains four residual values across the block
    reduction, avoiding the canonical route's tile staging and second global
    residual load. -/
@[gpu_kernel .SM90]
def fusedLayerNormResidual64x1024F32Direct
    (x_ptr : GPtr GpuFloat.Float32)
    (residual_ptr : GPtr GpuFloat.Float32)
    (weight_ptr : GPtr GpuFloat.Float32)
    (bias_ptr : GPtr GpuFloat.Float32)
    (out_ptr : GPtr GpuFloat.Float32)
    (out_resid_ptr : GPtr GpuFloat.Float32) : KernelM Unit := do
  setFamily .Blackwell
  emitRaw "__shared__ float row_sum[8];"
  emitRaw "__shared__ float row_sq_sum[8];"
  emitRaw "__shared__ float row_mean;"
  emitRaw "__shared__ float row_inv_std;"
  emitRaw s!"const float* x_raw = reinterpret_cast<const float*>({x_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const float* residual_raw = reinterpret_cast<const float*>({residual_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const float* weight_raw = reinterpret_cast<const float*>({weight_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const float* bias_raw = reinterpret_cast<const float*>({bias_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"float* out_raw = reinterpret_cast<float*>({out_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"float* out_residual_raw = reinterpret_cast<float*>({out_resid_ptr.id.toIdent}.raw_ptr);"
  emitRaw "const int row_base = static_cast<int>(blockIdx.x) * 1024;"
  emitRaw "const int lane = static_cast<int>(threadIdx.x) & 31;"
  emitRaw "const int warp_id = static_cast<int>(threadIdx.x) >> 5;"
  emitRaw "float values[4];"
  emitRaw "float sum = 0.0f;"
  emitRaw "float square_sum = 0.0f;"
  emitRaw "#pragma unroll"
  emitRaw "for (int item = 0; item < 4; ++item) {"
  emitRaw "  const int col = static_cast<int>(threadIdx.x) + item * 256;"
  emitRaw "  const int idx = row_base + col;"
  emitRaw "  const float value = x_raw[idx] + residual_raw[idx];"
  emitRaw "  values[item] = value;"
  emitRaw "  out_residual_raw[idx] = value;"
  emitRaw "  sum += value;"
  emitRaw "  square_sum = fmaf(value, value, square_sum);"
  emitRaw "}"
  emitRaw "#pragma unroll"
  emitRaw "for (int offset = 16; offset > 0; offset >>= 1) {"
  emitRaw "  sum += __shfl_down_sync(0xffffffffu, sum, offset);"
  emitRaw "  square_sum += __shfl_down_sync(0xffffffffu, square_sum, offset);"
  emitRaw "}"
  emitRaw "if (lane == 0) { row_sum[warp_id] = sum; row_sq_sum[warp_id] = square_sum; }"
  emitRaw "__syncthreads();"
  emitRaw "if (warp_id == 0) {"
  emitRaw "  sum = lane < 8 ? row_sum[lane] : 0.0f;"
  emitRaw "  square_sum = lane < 8 ? row_sq_sum[lane] : 0.0f;"
  emitRaw "  #pragma unroll"
  emitRaw "  for (int offset = 16; offset > 0; offset >>= 1) {"
  emitRaw "    sum += __shfl_down_sync(0xffffffffu, sum, offset);"
  emitRaw "    square_sum += __shfl_down_sync(0xffffffffu, square_sum, offset);"
  emitRaw "  }"
  emitRaw "  if (lane == 0) {"
  emitRaw "    const float mean = sum * 0.0009765625f;"
  emitRaw "    row_mean = mean;"
  emitRaw "    row_inv_std = rsqrtf(fmaf(-mean, mean, square_sum * 0.0009765625f) + 1.0e-5f);"
  emitRaw "  }"
  emitRaw "}"
  emitRaw "__syncthreads();"
  emitRaw "#pragma unroll"
  emitRaw "for (int item = 0; item < 4; ++item) {"
  emitRaw "  const int col = static_cast<int>(threadIdx.x) + item * 256;"
  emitRaw "  out_raw[row_base + col] = (values[item] - row_mean) * row_inv_std * weight_raw[col] + bias_raw[col];"
  emitRaw "}"

/-- BF16 counterpart of the row-parallel fixed-shape route. Reductions and the
    affine expression accumulate in FP32; residual and normalized outputs are
    rounded to BF16 at their stores. -/
@[gpu_kernel .SM90]
def fusedLayerNormResidual64x1024Bf16Direct
    (x_ptr : GPtr GpuFloat.BFloat16)
    (residual_ptr : GPtr GpuFloat.BFloat16)
    (weight_ptr : GPtr GpuFloat.BFloat16)
    (bias_ptr : GPtr GpuFloat.BFloat16)
    (out_ptr : GPtr GpuFloat.BFloat16)
    (out_resid_ptr : GPtr GpuFloat.BFloat16) : KernelM Unit := do
  setFamily .Blackwell
  emitRaw "__shared__ float row_sum[8];"
  emitRaw "__shared__ float row_sq_sum[8];"
  emitRaw "__shared__ float row_mean;"
  emitRaw "__shared__ float row_inv_std;"
  emitRaw s!"const __nv_bfloat16* x_raw = reinterpret_cast<const __nv_bfloat16*>({x_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const __nv_bfloat16* residual_raw = reinterpret_cast<const __nv_bfloat16*>({residual_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const __nv_bfloat16* weight_raw = reinterpret_cast<const __nv_bfloat16*>({weight_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const __nv_bfloat16* bias_raw = reinterpret_cast<const __nv_bfloat16*>({bias_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"__nv_bfloat16* out_raw = reinterpret_cast<__nv_bfloat16*>({out_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"__nv_bfloat16* out_residual_raw = reinterpret_cast<__nv_bfloat16*>({out_resid_ptr.id.toIdent}.raw_ptr);"
  emitRaw "const int row_base = static_cast<int>(blockIdx.x) * 1024;"
  emitRaw "const int lane = static_cast<int>(threadIdx.x) & 31;"
  emitRaw "const int warp_id = static_cast<int>(threadIdx.x) >> 5;"
  emitRaw "float values[4];"
  emitRaw "float sum = 0.0f;"
  emitRaw "float square_sum = 0.0f;"
  emitRaw "#pragma unroll"
  emitRaw "for (int item = 0; item < 4; ++item) {"
  emitRaw "  const int col = static_cast<int>(threadIdx.x) + item * 256;"
  emitRaw "  const int idx = row_base + col;"
  emitRaw "  const float value = __bfloat162float(x_raw[idx]) + __bfloat162float(residual_raw[idx]);"
  emitRaw "  values[item] = value;"
  emitRaw "  out_residual_raw[idx] = __float2bfloat16_rn(value);"
  emitRaw "  sum += value;"
  emitRaw "  square_sum = fmaf(value, value, square_sum);"
  emitRaw "}"
  emitRaw "#pragma unroll"
  emitRaw "for (int offset = 16; offset > 0; offset >>= 1) {"
  emitRaw "  sum += __shfl_down_sync(0xffffffffu, sum, offset);"
  emitRaw "  square_sum += __shfl_down_sync(0xffffffffu, square_sum, offset);"
  emitRaw "}"
  emitRaw "if (lane == 0) { row_sum[warp_id] = sum; row_sq_sum[warp_id] = square_sum; }"
  emitRaw "__syncthreads();"
  emitRaw "if (warp_id == 0) {"
  emitRaw "  sum = lane < 8 ? row_sum[lane] : 0.0f;"
  emitRaw "  square_sum = lane < 8 ? row_sq_sum[lane] : 0.0f;"
  emitRaw "  #pragma unroll"
  emitRaw "  for (int offset = 16; offset > 0; offset >>= 1) {"
  emitRaw "    sum += __shfl_down_sync(0xffffffffu, sum, offset);"
  emitRaw "    square_sum += __shfl_down_sync(0xffffffffu, square_sum, offset);"
  emitRaw "  }"
  emitRaw "  if (lane == 0) {"
  emitRaw "    const float mean = sum * 0.0009765625f;"
  emitRaw "    row_mean = mean;"
  emitRaw "    row_inv_std = rsqrtf(fmaf(-mean, mean, square_sum * 0.0009765625f) + 1.0e-5f);"
  emitRaw "  }"
  emitRaw "}"
  emitRaw "__syncthreads();"
  emitRaw "#pragma unroll"
  emitRaw "for (int item = 0; item < 4; ++item) {"
  emitRaw "  const int col = static_cast<int>(threadIdx.x) + item * 256;"
  emitRaw "  const float normalized = (values[item] - row_mean) * row_inv_std;"
  emitRaw "  const float affine = normalized * __bfloat162float(weight_raw[col]) + __bfloat162float(bias_raw[col]);"
  emitRaw "  out_raw[row_base + col] = __float2bfloat16_rn(affine);"
  emitRaw "}"

end Tyr.GPU.Kernels
