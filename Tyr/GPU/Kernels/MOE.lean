/-
  Tyr/GPU/Kernels/MOE.lean

  Mixture-of-Experts dispatch + GEMM kernels aligned to
  `external/ThunderKittens/kernels/parallel/moe_dispatch_gemm/moe_dispatch_gemm_h100.cu`.

  The vendored ThunderKittens kernel is a fused multi-GPU dispatch and grouped
  GEMM pipeline. The Lean DSL still lacks integer global layouts and the exact
  producer/consumer warpgroup protocol, so the canonical surface below models:

  - dispatch staging through multimem/shared-memory publication,
  - an explicit cross-device barrier between dispatch and compute,
  - grouped expert GEMM over already-dispatched local tiles.

  Routing metadata buffers are represented as Float32 compatibility layouts
  until integer global layouts are added to the DSL.
-/

import Tyr.GPU.Kernels.Prelude
import Tyr.GPU.Kernels.Support

namespace Tyr.GPU.Kernels.MOE

open Tyr.GPU
open Tyr.GPU.Codegen

private def groupedExpertMma
    (tokens : RT GpuFloat.BFloat16 128 64 .Row)
    (weights : RT GpuFloat.BFloat16 64 128 .Col)
    (accum : RT GpuFloat.Float32 128 128 .Row) : KernelM Unit := do
  mma accum tokens weights accum

/-- Dispatch-only phase of the ThunderKittens MOE kernel.

The caller provides already materialized per-device token blocks and a metadata
tile describing which local block should be published next. The actual indexed
pull from `pull_dispatch_indices` is represented here as a multimem publish plus
barriered local reload, which preserves the source phase structure. -/
@[gpu_kernel .SM90]
def tkMoeDispatch
    (pre_tokens_ptr : GPtr GpuFloat.BFloat16)
    (post_tokens_ptr : GPtr GpuFloat.BFloat16)
    (dispatch_meta_ptr : GPtr GpuFloat.Float32)
    (_num_padded_local_tokens : KVal UInt32)
    (_num_devices : KVal UInt32) : KernelM Unit := do
  comment "ThunderKittens moe_dispatch_gemm: dispatch phase"

  let coord ← blockCoord2D
  let tokenRows : Nat := 128
  let hidden : Nat := 64

  let preTokens : RT GpuFloat.BFloat16 tokenRows hidden ← allocRT .BFloat16 tokenRows hidden
  let dispatched : RT GpuFloat.BFloat16 tokenRows hidden ← allocRT .BFloat16 tokenRows hidden
  let dispatchMeta : RT GpuFloat.Float32 1 64 ← allocRT .Float32 1 64

  let preShared : ST GpuFloat.BFloat16 tokenRows hidden ← allocST .BFloat16 tokenRows hidden
  let publishShared : ST GpuFloat.BFloat16 tokenRows hidden ← allocST .BFloat16 tokenRows hidden
  let outShared : ST GpuFloat.BFloat16 tokenRows hidden ← allocST .BFloat16 tokenRows hidden
  let metaShared : ST GpuFloat.Float32 1 64 ← allocST .Float32 1 64

  Support.asyncTileLoad preShared pre_tokens_ptr coord (tokenRows * hidden * 2)
  Support.asyncTileLoad metaShared dispatch_meta_ptr coord (64 * 4)
  load preTokens preShared
  load dispatchMeta metaShared

  comment "Publish dispatched token block through the cluster-visible multimem stage"
  multimemStore publishShared preTokens
  Support.barrierAllDevices "moe dispatch publish complete" 0

  comment "Consume the local post-dispatch view after the barrier"
  load dispatched publishShared
  store outShared dispatched
  storeGlobal post_tokens_ptr outShared coord
  Support.barrierAllDevices "moe dispatch epilogue" 1

/-- Grouped expert GEMM over already-dispatched local tokens.

This mirrors the second half of the ThunderKittens fused kernel: after dispatch,
each device iterates over its local expert blocks and computes a grouped GEMM
against the expert weights. -/
@[gpu_kernel .SM90]
def tkMoeGroupedGemm
    (post_tokens_ptr : GPtr GpuFloat.BFloat16)
    (weights_ptr : GPtr GpuFloat.BFloat16)
    (expert_counts_ptr : GPtr GpuFloat.Float32)
    (outputs_ptr : GPtr GpuFloat.BFloat16)
    (_num_experts_per_device : KVal UInt32) : KernelM Unit := do
  comment "ThunderKittens moe_dispatch_gemm: grouped expert GEMM phase"

  let coord ← blockCoord2D
  let rowBlock : Nat := 128
  let redBlock : Nat := 64
  let colBlock : Nat := 128
  let pipelineStages : Nat := 4

  let tokens : RT GpuFloat.BFloat16 rowBlock redBlock ← allocRT .BFloat16 rowBlock redBlock
  let weights : RT GpuFloat.BFloat16 redBlock colBlock .Col ← allocRT .BFloat16 redBlock colBlock .Col
  let accum : RT GpuFloat.Float32 rowBlock colBlock ← zeroRT .Float32 rowBlock colBlock
  let out : RT GpuFloat.BFloat16 rowBlock colBlock ← allocRT .BFloat16 rowBlock colBlock
  let expertCounts : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  let tokenShared : ST GpuFloat.BFloat16 rowBlock redBlock ← allocST .BFloat16 rowBlock redBlock
  let weightShared : ST GpuFloat.BFloat16 redBlock colBlock .Col ← allocST .BFloat16 redBlock colBlock .Col
  let outShared : ST GpuFloat.BFloat16 rowBlock colBlock ← allocST .BFloat16 rowBlock colBlock
  let countsShared : SV GpuFloat.Float32 64 ← allocSV .Float32 64

  loadVecGlobalCoord countsShared expert_counts_ptr coord.c
  loadVec expertCounts countsShared

  comment "Producer/consumer GEMM pipeline over local expert blocks"
  for stageIdx in krange 0 pipelineStages do
    let tokenCoord := coord.withCol stageIdx.id
    let weightCoord := coord.withRow stageIdx.id
    Support.asyncTileLoad tokenShared post_tokens_ptr tokenCoord (rowBlock * redBlock * 2)
    Support.asyncTileLoad weightShared weights_ptr weightCoord (redBlock * colBlock * 2)
    load tokens tokenShared
    load weights weightShared
    groupedExpertMma tokens weights accum
    sync

  convert out accum
  store outShared out
  storeGlobal outputs_ptr outShared coord

/-- Canonical fused dispatch + grouped GEMM surface.

This keeps the same phase split as the source, but composes it inside one Lean
kernel entrypoint so the catalog has a concrete MOE surface today. -/
@[gpu_kernel .SM90]
def tkMoeDispatchGemm
    (pre_tokens_ptr : GPtr GpuFloat.BFloat16)
    (post_tokens_ptr : GPtr GpuFloat.BFloat16)
    (weights_ptr : GPtr GpuFloat.BFloat16)
    (dispatch_meta_ptr : GPtr GpuFloat.Float32)
    (expert_counts_ptr : GPtr GpuFloat.Float32)
    (outputs_ptr : GPtr GpuFloat.BFloat16)
    (num_padded_local_tokens : KVal UInt32)
    (num_devices : KVal UInt32)
    (num_experts_per_device : KVal UInt32) : KernelM Unit := do
  comment "ThunderKittens moe_dispatch_gemm: fused dispatch + grouped GEMM"
  tkMoeDispatch pre_tokens_ptr post_tokens_ptr dispatch_meta_ptr num_padded_local_tokens num_devices
  tkMoeGroupedGemm post_tokens_ptr weights_ptr expert_counts_ptr outputs_ptr num_experts_per_device

end Tyr.GPU.Kernels.MOE
