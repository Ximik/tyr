/*
 * tyr_laguna_moe.cu - Fused NVFP4 MoE expert forward for Laguna inference.
 *
 * Computes the routed expert sum for a batch of (token, slot) pairs directly
 * from the NVFP4-packed expert banks, without materializing BF16 weights:
 *
 *   routed[t] = Σ_p w[t,p] · expert_{idx[t,p]}(x[t])
 *   expert_e(x) = down( silu(gate x) * up x )
 *   W[i,j]      = e2m1(nibble) * (f8_scale[i, j/16] / globalScale)
 *
 * Bank layouts (row-major, contiguous):
 *   gate/up: packed U8 [E, moeInt, hidden/2], scales F8_E4M3 [E, moeInt, hidden/16]
 *   down:    packed U8 [E, hidden, moeInt/2], scales F8_E4M3 [E, hidden, moeInt/16]
 *   globals: F32 [E]
 *
 * The portable path uses two weight-stationary streaming GEMV kernels (decode
 * is bandwidth bound; every selected packed weight byte is read once):
 *
 *   Stage A: one block per (pair, 64-row tile of moeInt). The token vector x[t]
 *     is cached in shared memory (FP32). Each warp owns 8 rows and iterates
 *     over 512-byte row strips with vectorized uint4 loads; E2M1 nibbles are
 *     decoded through a 256-entry float2 shared LUT (byte -> (lo, hi)) and
 *     F8_E4M3 group scales through a 256-entry float shared LUT. Gate and up
 *     are accumulated together so hid[i] = silu(g[i]) * u[i] is written in one
 *     pass (FP32 [pairs, moeInt]).
 *   Stage B: one block per (pair, 64-row tile of hidden). hid is cached in
 *     shared memory; the same strip loop computes out[o] and atomically
 *     accumulates w·out[o] into the FP32 routed[T, hidden] buffer.
 * The host wrapper then casts routed to BF16 and returns it; the
 * moe_routed_scaling_factor and the shared expert stay on the Lean side.
 *
 * On SM12x, sufficiently large prefills instead use cuBLASLt's block-scaled
 * NVFP4 tensor-core GEMM. Activations are dynamically quantized per 16 values,
 * checkpoint weight scales are losslessly repacked to the hardware 128x4
 * layout once, and routed rows are compacted to the exact maximum expert
 * occupancy. Decode, small prefills, non-SM12x devices, older CUDA toolkits,
 * and TYR_LAGUNA_DISABLE_NATIVE_FP4=1 retain the portable path. This keeps the
 * public operation independent of the physical Blackwell target: a future
 * server-Blackwell GEMM backend can reuse the same routing and Lean API.
 *
 * Constraints (validated by the Lean caller and re-checked here):
 *   hidden % 32 == 0 and moeInt % 32 == 0 (so packed rows are 16-byte aligned
 *   and every 16-byte strip spans exactly two 16-element scale groups),
 *   tokens*k <= 65535 (grid.y limit).
 *
 * Expert indices are clamped into [0, E) on the device: the router guarantees
 * the range, and a clamp keeps a bad index from corrupting memory.
 *
 * The translation unit contains no inline Hopper/Blackwell-specific
 * instructions; the SM12x acceleration is selected through cuBLASLt at
 * runtime. The Makefile therefore compiles it for every CUDA target. When no
 * CUDA toolkit is present, the same file is compiled as plain C++ (`-x c++`,
 * no __CUDACC__) and only the stub extern at the bottom is emitted, keeping
 * the Lean symbol resolvable.
 */

#include "tyr_laguna_abi.h"

#ifndef __CUDACC__

/* ---------------- CPU stub (no CUDA toolkit): satisfy the Lean extern. --- */
#include <lean/lean.h>

extern "C" lean_object* lean_torch_laguna_moe_fp4_forward(
    lean_object* /*x*/, lean_object* /*topIdx*/, lean_object* /*topW*/,
    lean_object* /*gatePacked*/, lean_object* /*gateScale*/, lean_object* /*gateGlobal*/,
    lean_object* /*upPacked*/, lean_object* /*upScale*/, lean_object* /*upGlobal*/,
    lean_object* /*downPacked*/, lean_object* /*downScale*/, lean_object* /*downGlobal*/,
    uint64_t /*numExperts*/, uint64_t /*moeInt*/, uint64_t /*hidden*/) {
  return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(
      "laguna_moe_fp4_forward: libTyrC was built without CUDA support")));
}

#else

/* ------------------------------ CUDA build ------------------------------ */
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12080
#include <cublasLt.h>
#include <cuda_fp4.h>
#endif
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <lean/lean.h>

/* Defined in tyr.cpp (C++ linkage, global namespace; torch::Tensor is an
   alias of at::Tensor, so these declarations match its definitions). */
at::Tensor borrowTensor(b_lean_obj_arg o);
lean_object* giveTensor(at::Tensor t);

namespace {

constexpr int kThreads = 256;            // 8 warps per block
constexpr int kRowsPerBlock = 64;        // output rows per block (8 per warp)
constexpr int kWarpRows = kRowsPerBlock / (kThreads / 32);
constexpr int kStripBytes = 32 * 16;     // one uint4 per lane = 512B per warp strip
constexpr int kPrefillGroup = 4;         // routed pairs sharing one expert load
constexpr int kGroupedRowsPerBlock = 32; // four output rows per warp
constexpr int kGroupedWarpRows = kGroupedRowsPerBlock / (kThreads / 32);

struct ExpertTask {
  int expert;
  int pairStart;
  int pairCount;
  int reserved;
};

/* E2M1 magnitudes indexed by the low 3 bits; bit 3 is the sign. */
__device__ __forceinline__ float e2m1Mag(int nibble3) {
  const float mags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  return mags[nibble3 & 7];
}

__device__ __forceinline__ float warpSum(float v) {
  for (int m = 16; m > 0; m >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, m);
  }
  return v;
}

/* Fill the shared lookup tables: f8lut = F8_E4M3 byte -> float,
   pairlut = packed byte -> (e2m1(low nibble), e2m1(high nibble)). */
__device__ __forceinline__ void initLuts(float* f8lut, float2* pairlut) {
  const int tid = threadIdx.x;
  if (tid < 256) {
    f8lut[tid] = __half2float(__nv_cvt_fp8_to_halfraw(
        static_cast<uint8_t>(tid), __NV_E4M3));
    const float lo = (tid & 8) ? -e2m1Mag(tid & 7) : e2m1Mag(tid & 7);
    const int hi3 = (tid >> 4) & 0xF;
    const float hi = (hi3 & 8) ? -e2m1Mag(hi3 & 7) : e2m1Mag(hi3 & 7);
    pairlut[tid] = make_float2(lo, hi);
  }
}

/* Accumulate one 16-byte strip (32 elements = two 16-element scale groups)
   of one packed matrix row against xv[32]: acc += Σ_j scale·e2m1(nibble)·x[j]. */
__device__ __forceinline__ float dot16(
    uint4 q, const float* xv, float s0, float s1, const float2* pairlut) {
  float acc = 0.0f;
  const uint32_t words[4] = {q.x, q.y, q.z, q.w};
  #pragma unroll
  for (int w = 0; w < 4; ++w) {
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
      const uint32_t byte = (words[w] >> (8 * b)) & 0xFFu;
      const int elem = 2 * (4 * w + b);
      const float2 lh = pairlut[byte];
      const float sc = (elem < 16) ? s0 : s1;
      acc += sc * (lh.x * xv[elem] + lh.y * xv[elem + 1]);
    }
  }
  return acc;
}

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12080

/*
 * Stage A: gate+up GEMVs with fused SwiGLU.
 * Grid: (ceil(moeInt / 64), pairs). Block: 256 threads.
 * Shared: [hidden] floats of x, [256] f8 lut, [256] float2 nibble-pair lut.
 */
__global__ __launch_bounds__(kThreads) void laguna_moe_fp4_stage_a(
    const uint8_t* __restrict__ gateP, const uint8_t* __restrict__ gateS,
    const uint8_t* __restrict__ upP, const uint8_t* __restrict__ upS,
    const float* __restrict__ gateG, const float* __restrict__ upG,
    const __nv_bfloat16* __restrict__ x, const int64_t* __restrict__ topIdx,
    float* __restrict__ hid,
    int k, int moeInt, int hidden, int numExperts) {
  extern __shared__ float smem[];
  float* xs = smem;                       // [hidden]
  float* f8lut = smem + hidden;           // [256]
  float2* pairlut = reinterpret_cast<float2*>(smem + hidden + 256);  // [256]

  const int pair = blockIdx.y;
  const int t = pair / k;
  int64_t e = topIdx[pair];
  e = e < 0 ? 0 : (e >= (int64_t)numExperts ? (int64_t)numExperts - 1 : e);

  const __nv_bfloat16* xRow = x + (size_t)t * hidden;
  for (int j = threadIdx.x; j < hidden; j += kThreads) {
    xs[j] = __bfloat162float(xRow[j]);
  }
  initLuts(f8lut, pairlut);
  __syncthreads();

  const int hb = hidden >> 1;             // packed bytes per row
  const int sg = hidden >> 4;             // scale groups per row
  const size_t rowBase = (size_t)e * moeInt;
  const float invGateG = 1.0f / gateG[e];
  const float invUpG = 1.0f / upG[e];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row0 = blockIdx.x * kRowsPerBlock;

  float gAcc[kWarpRows];
  float uAcc[kWarpRows];
  #pragma unroll
  for (int r = 0; r < kWarpRows; ++r) { gAcc[r] = 0.0f; uAcc[r] = 0.0f; }

  for (int off = lane * 16; off < hb; off += kStripBytes) {
    /* Cache this strip's 32 activations once; reuse across the warp's rows. */
    float xv[32];
    const float4* x4 = reinterpret_cast<const float4*>(xs + 2 * off);
    #pragma unroll
    for (int v = 0; v < 8; ++v) {
      const float4 f = x4[v];
      xv[4 * v] = f.x; xv[4 * v + 1] = f.y; xv[4 * v + 2] = f.z; xv[4 * v + 3] = f.w;
    }
    const int s0 = off >> 3;
    #pragma unroll
    for (int r = 0; r < kWarpRows; ++r) {
      const int row = row0 + warp + r * (kThreads / 32);
      if (row >= moeInt) break;
      const size_t rOff = (rowBase + row) * (size_t)hb;
      const size_t sOff = (rowBase + row) * (size_t)sg;
      const uint4 gq = *reinterpret_cast<const uint4*>(gateP + rOff + off);
      const uint4 uq = *reinterpret_cast<const uint4*>(upP + rOff + off);
      const float gs0 = f8lut[gateS[sOff + s0]];
      const float gs1 = f8lut[gateS[sOff + s0 + 1]];
      const float us0 = f8lut[upS[sOff + s0]];
      const float us1 = f8lut[upS[sOff + s0 + 1]];
      gAcc[r] += dot16(gq, xv, gs0, gs1, pairlut);
      uAcc[r] += dot16(uq, xv, us0, us1, pairlut);
    }
  }

  #pragma unroll
  for (int r = 0; r < kWarpRows; ++r) {
    const int row = row0 + warp + r * (kThreads / 32);
    if (row >= moeInt) break;
    const float g = warpSum(gAcc[r]) * invGateG;
    const float u = warpSum(uAcc[r]) * invUpG;
    if (lane == 0) {
      const float hidv = (g / (1.0f + expf(-g))) * u;   // silu(g) * u
      hid[(size_t)pair * moeInt + row] = hidv;
    }
  }
}

/*
 * Stage B: down GEMV + routed weighted accumulation.
 * Grid: (ceil(hidden / 64), pairs). Block: 256 threads.
 * Shared: [moeInt] floats of hid, [256] f8 lut, [256] float2 nibble-pair lut.
 */
__global__ __launch_bounds__(kThreads) void laguna_moe_fp4_stage_b(
    const uint8_t* __restrict__ downP, const uint8_t* __restrict__ downS,
    const float* __restrict__ downG,
    const float* __restrict__ hid,
    const __nv_bfloat16* __restrict__ topW, const int64_t* __restrict__ topIdx,
    float* __restrict__ routed,
    int k, int moeInt, int hidden, int numExperts) {
  extern __shared__ float smem[];
  float* hs = smem;                       // [moeInt]
  float* f8lut = smem + moeInt;           // [256]
  float2* pairlut = reinterpret_cast<float2*>(smem + moeInt + 256);  // [256]

  const int pair = blockIdx.y;
  const int t = pair / k;
  int64_t e = topIdx[pair];
  e = e < 0 ? 0 : (e >= (int64_t)numExperts ? (int64_t)numExperts - 1 : e);
  const float w = __bfloat162float(topW[pair]);

  const float* hidRow = hid + (size_t)pair * moeInt;
  for (int j = threadIdx.x; j < moeInt; j += kThreads) {
    hs[j] = hidRow[j];
  }
  initLuts(f8lut, pairlut);
  __syncthreads();

  const int mb = moeInt >> 1;             // packed bytes per row
  const int sg = moeInt >> 4;             // scale groups per row
  const size_t rowBase = (size_t)e * hidden;
  const float invDownG = 1.0f / downG[e];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row0 = blockIdx.x * kRowsPerBlock;

  float dAcc[kWarpRows];
  #pragma unroll
  for (int r = 0; r < kWarpRows; ++r) { dAcc[r] = 0.0f; }

  for (int off = lane * 16; off < mb; off += kStripBytes) {
    float hv[32];
    const float4* h4 = reinterpret_cast<const float4*>(hs + 2 * off);
    #pragma unroll
    for (int v = 0; v < 8; ++v) {
      const float4 f = h4[v];
      hv[4 * v] = f.x; hv[4 * v + 1] = f.y; hv[4 * v + 2] = f.z; hv[4 * v + 3] = f.w;
    }
    const int s0 = off >> 3;
    #pragma unroll
    for (int r = 0; r < kWarpRows; ++r) {
      const int row = row0 + warp + r * (kThreads / 32);
      if (row >= hidden) break;
      const size_t rOff = (rowBase + row) * (size_t)mb;
      const size_t sOff = (rowBase + row) * (size_t)sg;
      const uint4 dq = *reinterpret_cast<const uint4*>(downP + rOff + off);
      const float ds0 = f8lut[downS[sOff + s0]];
      const float ds1 = f8lut[downS[sOff + s0 + 1]];
      dAcc[r] += dot16(dq, hv, ds0, ds1, pairlut);
    }
  }

  #pragma unroll
  for (int r = 0; r < kWarpRows; ++r) {
    const int row = row0 + warp + r * (kThreads / 32);
    if (row >= hidden) break;
    const float out = warpSum(dAcc[r]) * invDownG;
    if (lane == 0) {
      atomicAdd(&routed[(size_t)t * hidden + row], w * out);
    }
  }
}

/* Build an expert-major dispatch without a host synchronization. The task
 * list contains consecutive chunks of at most kPrefillGroup pairs for one
 * expert; the grouped GEMV kernels use it to read that expert's packed rows
 * once and apply them to every routed token in the chunk. A four-pair group
 * captures almost all reuse for short prompts without forcing the 101 KiB
 * shared-memory footprint and single-block occupancy of a 16-pair group. */
__global__ void laguna_moe_count_experts(
    const int64_t* __restrict__ topIdx, int* __restrict__ counts,
    int pairs, int numExperts) {
  for (int pair = blockIdx.x * blockDim.x + threadIdx.x;
       pair < pairs; pair += blockDim.x * gridDim.x) {
    int64_t e = topIdx[pair];
    e = e < 0 ? 0 : (e >= (int64_t)numExperts ? (int64_t)numExperts - 1 : e);
    atomicAdd(&counts[e], 1);
  }
}

__global__ void laguna_moe_build_layout(
    const int* __restrict__ counts, int* __restrict__ offsets,
    ExpertTask* __restrict__ tasks, int* __restrict__ taskCount,
    int numExperts) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int pairOffset = 0;
  int nextTask = 0;
  for (int e = 0; e < numExperts; ++e) {
    offsets[e] = pairOffset;
    const int count = counts[e];
    for (int start = 0; start < count; start += kPrefillGroup) {
      const int n = min(kPrefillGroup, count - start);
      tasks[nextTask++] = ExpertTask{e, pairOffset + start, n, 0};
    }
    pairOffset += count;
  }
  offsets[numExperts] = pairOffset;
  *taskCount = nextTask;
}

__global__ void laguna_moe_scatter_pairs(
    const int64_t* __restrict__ topIdx, const int* __restrict__ offsets,
    int* __restrict__ cursors, int* __restrict__ pairOrder,
    int pairs, int numExperts) {
  for (int pair = blockIdx.x * blockDim.x + threadIdx.x;
       pair < pairs; pair += blockDim.x * gridDim.x) {
    int64_t e = topIdx[pair];
    e = e < 0 ? 0 : (e >= (int64_t)numExperts ? (int64_t)numExperts - 1 : e);
    const int slot = atomicAdd(&cursors[e], 1);
    pairOrder[offsets[e] + slot] = pair;
  }
}

/* Expert-grouped prefill Stage A. Work is a persistent one-dimensional grid
 * over (task, output-row-tile), so the host need not read taskCount back from
 * the device. Each block caches up to four BF16 token rows, loads every packed
 * gate/up weight row once, and reuses it for all pairs in the expert task. */
__global__ __launch_bounds__(kThreads) void laguna_moe_fp4_stage_a_grouped(
    const uint8_t* __restrict__ gateP, const uint8_t* __restrict__ gateS,
    const uint8_t* __restrict__ upP, const uint8_t* __restrict__ upS,
    const float* __restrict__ gateG, const float* __restrict__ upG,
    const __nv_bfloat16* __restrict__ x,
    const ExpertTask* __restrict__ tasks, const int* __restrict__ taskCount,
    const int* __restrict__ pairOrder, float* __restrict__ hid,
    int k, int moeInt, int hidden) {
  extern __shared__ unsigned char smemRaw[];
  auto* xs = reinterpret_cast<__nv_bfloat16*>(smemRaw);
  auto* f8lut = reinterpret_cast<float*>(xs + (size_t)kPrefillGroup * hidden);
  auto* pairlut = reinterpret_cast<float2*>(f8lut + 256);

  initLuts(f8lut, pairlut);
  __syncthreads();

  const int rowTiles = (moeInt + kGroupedRowsPerBlock - 1) /
                       kGroupedRowsPerBlock;
  const int totalWork = (*taskCount) * rowTiles;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int hb = hidden >> 1;
  const int sg = hidden >> 4;

  for (int work = blockIdx.x; work < totalWork; work += gridDim.x) {
    const int taskId = work / rowTiles;
    const int rowTile = work - taskId * rowTiles;
    const ExpertTask task = tasks[taskId];

    for (int j = threadIdx.x; j < task.pairCount * hidden; j += kThreads) {
      const int q = j / hidden;
      const int col = j - q * hidden;
      const int pair = pairOrder[task.pairStart + q];
      xs[(size_t)q * hidden + col] = x[(size_t)(pair / k) * hidden + col];
    }
    __syncthreads();

    const size_t rowBase = (size_t)task.expert * moeInt;
    const float invGateG = 1.0f / gateG[task.expert];
    const float invUpG = 1.0f / upG[task.expert];
    const int row0 = rowTile * kGroupedRowsPerBlock;

    float gAcc[kGroupedWarpRows][kPrefillGroup];
    float uAcc[kGroupedWarpRows][kPrefillGroup];
    #pragma unroll
    for (int r = 0; r < kGroupedWarpRows; ++r) {
      #pragma unroll
      for (int q = 0; q < kPrefillGroup; ++q) {
        gAcc[r][q] = 0.0f;
        uAcc[r][q] = 0.0f;
      }
    }

    for (int off = lane * 16; off < hb; off += kStripBytes) {
      const int s0 = off >> 3;
      #pragma unroll
      for (int r = 0; r < kGroupedWarpRows; ++r) {
        const int row = row0 + warp + r * (kThreads / 32);
        if (row >= moeInt) break;
        const size_t rOff = (rowBase + row) * (size_t)hb;
        const size_t sOff = (rowBase + row) * (size_t)sg;
        const uint4 gq = *reinterpret_cast<const uint4*>(gateP + rOff + off);
        const uint4 uq = *reinterpret_cast<const uint4*>(upP + rOff + off);
        const float gs0 = f8lut[gateS[sOff + s0]];
        const float gs1 = f8lut[gateS[sOff + s0 + 1]];
        const float us0 = f8lut[upS[sOff + s0]];
        const float us1 = f8lut[upS[sOff + s0 + 1]];
        #pragma unroll
        for (int q = 0; q < kPrefillGroup; ++q) {
          if (q >= task.pairCount) break;
          float xv[32];
          const __nv_bfloat16* xStrip = xs + (size_t)q * hidden + 2 * off;
          #pragma unroll
          for (int v = 0; v < 32; ++v) xv[v] = __bfloat162float(xStrip[v]);
          gAcc[r][q] += dot16(gq, xv, gs0, gs1, pairlut);
          uAcc[r][q] += dot16(uq, xv, us0, us1, pairlut);
        }
      }
    }

    #pragma unroll
    for (int r = 0; r < kGroupedWarpRows; ++r) {
      const int row = row0 + warp + r * (kThreads / 32);
      if (row >= moeInt) break;
      #pragma unroll
      for (int q = 0; q < kPrefillGroup; ++q) {
        if (q >= task.pairCount) break;
        const float g = warpSum(gAcc[r][q]) * invGateG;
        const float u = warpSum(uAcc[r][q]) * invUpG;
        if (lane == 0) {
          const int pair = pairOrder[task.pairStart + q];
          hid[(size_t)pair * moeInt + row] =
              (g / (1.0f + expf(-g))) * u;
        }
      }
    }
    __syncthreads();
  }
}

/* Expert-grouped prefill Stage B. The FP32 intermediate rows are cached for
 * the whole task; one packed down-projection row then feeds every routed pair
 * before the weighted outputs are accumulated into their source tokens. */
__global__ __launch_bounds__(kThreads) void laguna_moe_fp4_stage_b_grouped(
    const uint8_t* __restrict__ downP, const uint8_t* __restrict__ downS,
    const float* __restrict__ downG, const float* __restrict__ hid,
    const __nv_bfloat16* __restrict__ topW,
    const ExpertTask* __restrict__ tasks, const int* __restrict__ taskCount,
    const int* __restrict__ pairOrder, float* __restrict__ routed,
    int k, int moeInt, int hidden) {
  extern __shared__ float smem[];
  float* hs = smem;
  float* f8lut = hs + (size_t)kPrefillGroup * moeInt;
  float2* pairlut = reinterpret_cast<float2*>(f8lut + 256);

  initLuts(f8lut, pairlut);
  __syncthreads();

  const int rowTiles = (hidden + kGroupedRowsPerBlock - 1) /
                       kGroupedRowsPerBlock;
  const int totalWork = (*taskCount) * rowTiles;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int mb = moeInt >> 1;
  const int sg = moeInt >> 4;

  for (int work = blockIdx.x; work < totalWork; work += gridDim.x) {
    const int taskId = work / rowTiles;
    const int rowTile = work - taskId * rowTiles;
    const ExpertTask task = tasks[taskId];

    for (int j = threadIdx.x; j < task.pairCount * moeInt; j += kThreads) {
      const int q = j / moeInt;
      const int col = j - q * moeInt;
      const int pair = pairOrder[task.pairStart + q];
      hs[(size_t)q * moeInt + col] = hid[(size_t)pair * moeInt + col];
    }
    __syncthreads();

    const size_t rowBase = (size_t)task.expert * hidden;
    const float invDownG = 1.0f / downG[task.expert];
    const int row0 = rowTile * kGroupedRowsPerBlock;

    float dAcc[kGroupedWarpRows][kPrefillGroup];
    #pragma unroll
    for (int r = 0; r < kGroupedWarpRows; ++r) {
      #pragma unroll
      for (int q = 0; q < kPrefillGroup; ++q) dAcc[r][q] = 0.0f;
    }

    for (int off = lane * 16; off < mb; off += kStripBytes) {
      const int s0 = off >> 3;
      #pragma unroll
      for (int r = 0; r < kGroupedWarpRows; ++r) {
        const int row = row0 + warp + r * (kThreads / 32);
        if (row >= hidden) break;
        const size_t rOff = (rowBase + row) * (size_t)mb;
        const size_t sOff = (rowBase + row) * (size_t)sg;
        const uint4 dq = *reinterpret_cast<const uint4*>(downP + rOff + off);
        const float ds0 = f8lut[downS[sOff + s0]];
        const float ds1 = f8lut[downS[sOff + s0 + 1]];
        #pragma unroll
        for (int q = 0; q < kPrefillGroup; ++q) {
          if (q >= task.pairCount) break;
          float hv[32];
          const float* hStrip = hs + (size_t)q * moeInt + 2 * off;
          const float4* h4 = reinterpret_cast<const float4*>(hStrip);
          #pragma unroll
          for (int v = 0; v < 8; ++v) {
            const float4 f = h4[v];
            hv[4 * v] = f.x; hv[4 * v + 1] = f.y;
            hv[4 * v + 2] = f.z; hv[4 * v + 3] = f.w;
          }
          dAcc[r][q] += dot16(dq, hv, ds0, ds1, pairlut);
        }
      }
    }

    #pragma unroll
    for (int r = 0; r < kGroupedWarpRows; ++r) {
      const int row = row0 + warp + r * (kThreads / 32);
      if (row >= hidden) break;
      #pragma unroll
      for (int q = 0; q < kPrefillGroup; ++q) {
        if (q >= task.pairCount) break;
        const float out = warpSum(dAcc[r][q]) * invDownG;
        if (lane == 0) {
          const int pair = pairOrder[task.pairStart + q];
          const int token = pair / k;
          const float w = __bfloat162float(topW[pair]);
          atomicAdd(&routed[(size_t)token * hidden + row], w * out);
        }
      }
    }
    __syncthreads();
  }
}

/*
 * Native Blackwell prefill path.
 *
 * cuBLASLt's NVFP4 batched GEMM reads each expert bank once and executes the
 * block-scaled tensor-core MMA.  Its batched kernels do not support a
 * different alpha for every batch item, so the checkpoint's per-expert global
 * scales are applied in the adjacent SwiGLU/scatter kernels instead:
 *
 *   gateRaw = q(x)·sx · q(Wg)·sg
 *   gate    = gateRaw / gateGlobal[e]
 *
 * Activations use a global scale of one and dynamic UE4M3 scales per
 * 16-element block.  The routed rows are stored as a fixed-stride
 * [expert, roundUp(maxExpertRows, 4), K] batch.  A short device count pass and
 * one four-byte stream readback determine the exact maximum.  This avoids
 * padding every expert to the full token count (which is mostly empty work for
 * a balanced router) while preserving arbitrary, even maximally skewed, top-k
 * assignments without a capacity assumption.
 *
 * Weight block scales arrive in logical row-major [E, N, K/16] order.  The
 * tensor-core API consumes the hardware 128x4 swizzle.  prepackScale() caches
 * that lossless permutation once per source tensor; it intentionally retains
 * the source tensor alongside the packed copy so allocator pointer reuse
 * cannot alias a stale cache entry.
 */
constexpr int kNvfp4Block = 16;
constexpr int kNvfp4ScaleRows = 128;
constexpr int kNvfp4ScaleCols = 4;
constexpr size_t kCublasWorkspaceBytes = 128ULL << 20;

__host__ __device__ __forceinline__ size_t nativeScaleIndex(
    int matrix, int row, int kBlock, int rows, int kBlocks) {
  const int paddedRows =
      (rows + kNvfp4ScaleRows - 1) / kNvfp4ScaleRows * kNvfp4ScaleRows;
  const int kTiles =
      (kBlocks + kNvfp4ScaleCols - 1) / kNvfp4ScaleCols;
  const size_t matrixStride = (size_t)paddedRows * kTiles * kNvfp4ScaleCols;
  const size_t tileBase =
      ((size_t)(row / kNvfp4ScaleRows) * kTiles +
       (size_t)(kBlock / kNvfp4ScaleCols)) * 512;
  const size_t tileOffset =
      (size_t)(row & 31) * 16 +
      (size_t)((row >> 5) & 3) * 4 +
      (size_t)(kBlock & 3);
  return (size_t)matrix * matrixStride + tileBase + tileOffset;
}

__global__ void laguna_nvfp4_prepack_scales(
    const uint8_t* __restrict__ input, uint8_t* __restrict__ output,
    int matrices, int rows, int kBlocks) {
  const int64_t total = (int64_t)matrices * rows * kBlocks;
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       idx < total; idx += (int64_t)blockDim.x * gridDim.x) {
    const int matrix = (int)(idx / ((int64_t)rows * kBlocks));
    const int rem = (int)(idx - (int64_t)matrix * rows * kBlocks);
    const int row = rem / kBlocks;
    const int kBlock = rem - row * kBlocks;
    output[nativeScaleIndex(matrix, row, kBlock, rows, kBlocks)] = input[idx];
  }
}

__global__ void laguna_nvfp4_count_rows(
    const int64_t* __restrict__ topIdx, int* __restrict__ counts,
    int* __restrict__ pairRow, int* __restrict__ maxRows,
    int pairs, int numExperts) {
  for (int pair = blockIdx.x * blockDim.x + threadIdx.x;
       pair < pairs; pair += blockDim.x * gridDim.x) {
    int64_t expert = topIdx[pair];
    expert = expert < 0 ? 0 :
        (expert >= (int64_t)numExperts ? (int64_t)numExperts - 1 : expert);
    const int row = atomicAdd(&counts[expert], 1);
    pairRow[pair] = row;
    atomicMax(maxRows, row + 1);
  }
}

__global__ void laguna_nvfp4_finalize_rows(
    const int64_t* __restrict__ topIdx, int* __restrict__ pairSlot,
    int pairs, int rowsPerExpert, int numExperts) {
  for (int pair = blockIdx.x * blockDim.x + threadIdx.x;
       pair < pairs; pair += blockDim.x * gridDim.x) {
    int64_t expert = topIdx[pair];
    expert = expert < 0 ? 0 :
        (expert >= (int64_t)numExperts ? (int64_t)numExperts - 1 : expert);
    pairSlot[pair] = (int)expert * rowsPerExpert + pairSlot[pair];
  }
}

__device__ __forceinline__ uint8_t fp8E4m3(float value) {
  return __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ float fp8E4m3ToFloat(uint8_t value) {
  return __half2float(__nv_cvt_fp8_to_halfraw(value, __NV_E4M3));
}

__device__ __forceinline__ void quantizeNvfp4Block(
    const float (&values)[kNvfp4Block], uint8_t* packed, uint8_t* scale) {
  float amax = 0.0f;
  #pragma unroll
  for (int i = 0; i < kNvfp4Block; ++i) {
    amax = fmaxf(amax, fabsf(values[i]));
  }
  const uint8_t scaleByte = fp8E4m3(amax * (1.0f / 6.0f));
  *scale = scaleByte;
  const float quantScale = fp8E4m3ToFloat(scaleByte);
  const float invScale = quantScale > 0.0f ? 1.0f / quantScale : 0.0f;
  #pragma unroll
  for (int i = 0; i < kNvfp4Block; i += 2) {
    const float2 pair = make_float2(
        values[i] * invScale, values[i + 1] * invScale);
    packed[i / 2] = __nv_cvt_float2_to_fp4x2(
        pair, __NV_E2M1, cudaRoundNearest);
  }
}

__global__ void laguna_nvfp4_quantize_routed(
    const __nv_bfloat16* __restrict__ x,
    const int* __restrict__ pairSlot,
    uint8_t* __restrict__ packed, uint8_t* __restrict__ scales,
    int pairs, int topK, int rowsPerExpert, int features) {
  const int kBlocks = features / kNvfp4Block;
  const int64_t total = (int64_t)pairs * kBlocks;
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       idx < total; idx += (int64_t)blockDim.x * gridDim.x) {
    const int pair = (int)(idx / kBlocks);
    const int kBlock = (int)(idx - (int64_t)pair * kBlocks);
    const int slot = pairSlot[pair];
    if (slot < 0) continue;
    const int expert = slot / rowsPerExpert;
    const int row = slot - expert * rowsPerExpert;
    const __nv_bfloat16* input =
        x + (size_t)(pair / topK) * features +
        (size_t)kBlock * kNvfp4Block;
    float values[kNvfp4Block];
    #pragma unroll
    for (int i = 0; i < kNvfp4Block; ++i) {
      values[i] = __bfloat162float(input[i]);
    }
    uint8_t* packedBlock =
        packed + (size_t)slot * (features / 2) +
        (size_t)kBlock * (kNvfp4Block / 2);
    uint8_t* scale = scales +
        nativeScaleIndex(expert, row, kBlock, rowsPerExpert, kBlocks);
    quantizeNvfp4Block(values, packedBlock, scale);
  }
}

__global__ void laguna_nvfp4_swiglu_quantize(
    const __nv_bfloat16* __restrict__ gateRaw,
    const __nv_bfloat16* __restrict__ upRaw,
    const float* __restrict__ gateGlobal,
    const float* __restrict__ upGlobal,
    const int* __restrict__ pairSlot,
    uint8_t* __restrict__ packed, uint8_t* __restrict__ scales,
    int pairs, int rowsPerExpert, int features) {
  const int kBlocks = features / kNvfp4Block;
  const int64_t total = (int64_t)pairs * kBlocks;
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       idx < total; idx += (int64_t)blockDim.x * gridDim.x) {
    const int pair = (int)(idx / kBlocks);
    const int kBlock = (int)(idx - (int64_t)pair * kBlocks);
    const int slot = pairSlot[pair];
    if (slot < 0) continue;
    const int expert = slot / rowsPerExpert;
    const int row = slot - expert * rowsPerExpert;
    const float invGate = 1.0f / gateGlobal[expert];
    const float invUp = 1.0f / upGlobal[expert];
    const size_t base =
        (size_t)slot * features + (size_t)kBlock * kNvfp4Block;
    float values[kNvfp4Block];
    #pragma unroll
    for (int i = 0; i < kNvfp4Block; ++i) {
      const float gate = __bfloat162float(gateRaw[base + i]) * invGate;
      const float up = __bfloat162float(upRaw[base + i]) * invUp;
      values[i] = (gate / (1.0f + expf(-gate))) * up;
    }
    uint8_t* packedBlock =
        packed + (size_t)slot * (features / 2) +
        (size_t)kBlock * (kNvfp4Block / 2);
    uint8_t* scale = scales +
        nativeScaleIndex(expert, row, kBlock, rowsPerExpert, kBlocks);
    quantizeNvfp4Block(values, packedBlock, scale);
  }
}

__global__ void laguna_nvfp4_scatter_down(
    const __nv_bfloat16* __restrict__ downRaw,
    const float* __restrict__ downGlobal,
    const __nv_bfloat16* __restrict__ topW,
    const int* __restrict__ pairSlot,
    float* __restrict__ routed,
    int pairs, int topK, int rowsPerExpert, int hidden) {
  const int64_t total = (int64_t)pairs * hidden;
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       idx < total; idx += (int64_t)blockDim.x * gridDim.x) {
    const int pair = (int)(idx / hidden);
    const int col = (int)(idx - (int64_t)pair * hidden);
    const int slot = pairSlot[pair];
    if (slot < 0) continue;
    const int expert = slot / rowsPerExpert;
    const float weight =
        __bfloat162float(topW[pair]) / downGlobal[expert];
    const float value = __bfloat162float(
        downRaw[(size_t)slot * hidden + col]);
    atomicAdd(&routed[(size_t)(pair / topK) * hidden + col],
              weight * value);
  }
}

struct NativeScaleCacheEntry {
  at::Tensor source;
  at::Tensor packed;
  int64_t rows;
  int64_t features;
};

thread_local std::unordered_map<const c10::TensorImpl*, NativeScaleCacheEntry>
    nativeScaleCache;

static bool prepackScale(
    const at::Tensor& source, int64_t matrices, int64_t rows,
    int64_t features, cudaStream_t stream, at::Tensor& packed,
    std::string& err) {
  const auto* key = source.unsafeGetTensorImpl();
  auto it = nativeScaleCache.find(key);
  if (it != nativeScaleCache.end() &&
      it->second.source.is_same(source) &&
      it->second.rows == rows && it->second.features == features) {
    packed = it->second.packed;
    return true;
  }
  const int64_t kBlocks = features / kNvfp4Block;
  const int64_t paddedRows =
      (rows + kNvfp4ScaleRows - 1) / kNvfp4ScaleRows *
      kNvfp4ScaleRows;
  const int64_t storage = matrices * paddedRows * kBlocks;
  at::Tensor output = at::empty({storage}, source.options());
  const int64_t elements = matrices * rows * kBlocks;
  const int blocks = (int)std::min<int64_t>(
      (elements + kThreads - 1) / kThreads, 65535);
  laguna_nvfp4_prepack_scales<<<blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(source.data_ptr()),
      reinterpret_cast<uint8_t*>(output.data_ptr()),
      (int)matrices, (int)rows, (int)kBlocks);
  const cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    err = std::string("scale prepack launch failed: ") +
          cudaGetErrorString(status);
    return false;
  }
  auto [inserted, _] = nativeScaleCache.insert_or_assign(
      key, NativeScaleCacheEntry{source, output, rows, features});
  packed = inserted->second.packed;
  return true;
}

struct LagunaNvfp4BatchedGemm {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t matmul = nullptr;
  cublasLtMatrixLayout_t layoutWeight = nullptr;
  cublasLtMatrixLayout_t layoutAct = nullptr;
  cublasLtMatrixLayout_t layoutC = nullptr;
  cublasLtMatrixLayout_t layoutD = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic{};
  int device = -1;
  int batch = 0;
  int m = 0;
  int n = 0;
  int k = 0;

  ~LagunaNvfp4BatchedGemm() {
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (layoutD) cublasLtMatrixLayoutDestroy(layoutD);
    if (layoutC) cublasLtMatrixLayoutDestroy(layoutC);
    if (layoutAct) cublasLtMatrixLayoutDestroy(layoutAct);
    if (layoutWeight) cublasLtMatrixLayoutDestroy(layoutWeight);
    if (matmul) cublasLtMatmulDescDestroy(matmul);
    if (handle) cublasLtDestroy(handle);
  }

  static bool ok(cublasStatus_t status, const char* what, std::string& err) {
    if (status == CUBLAS_STATUS_SUCCESS) return true;
    err = std::string(what) + " failed with cuBLAS status " +
          std::to_string((int)status);
    return false;
  }

  bool init(
      int device_, int batch_, int m_, int n_, int k_,
      const void* actScale, const void* weightScale, std::string& err) {
    device = device_;
    batch = batch_;
    m = m_;
    n = n_;
    k = k_;
    if (!ok(cublasLtCreate(&handle), "cublasLtCreate", err)) return false;
    if (!ok(cublasLtMatmulDescCreate(
                &matmul, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "cublasLtMatmulDescCreate", err)) return false;
    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_N;
    if (!ok(cublasLtMatmulDescSetAttribute(
                matmul, CUBLASLT_MATMUL_DESC_TRANSA,
                &transA, sizeof(transA)),
            "set TRANSA", err) ||
        !ok(cublasLtMatmulDescSetAttribute(
                matmul, CUBLASLT_MATMUL_DESC_TRANSB,
                &transB, sizeof(transB)),
            "set TRANSB", err)) return false;
    cublasLtMatmulMatrixScale_t scaleMode =
        CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    if (!ok(cublasLtMatmulDescSetAttribute(
                matmul, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                &scaleMode, sizeof(scaleMode)),
            "set weight scale mode", err) ||
        !ok(cublasLtMatmulDescSetAttribute(
                matmul, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                &scaleMode, sizeof(scaleMode)),
            "set activation scale mode", err)) return false;
    if (!ok(cublasLtMatrixLayoutCreate(
                &layoutWeight, CUDA_R_4F_E2M1, k, n, k),
            "create weight layout", err) ||
        !ok(cublasLtMatrixLayoutCreate(
                &layoutAct, CUDA_R_4F_E2M1, k, m, k),
            "create activation layout", err) ||
        !ok(cublasLtMatrixLayoutCreate(
                &layoutC, CUDA_R_16BF, n, m, n),
            "create C layout", err) ||
        !ok(cublasLtMatrixLayoutCreate(
                &layoutD, CUDA_R_16BF, n, m, n),
            "create D layout", err)) return false;
    const int64_t weightStride = (int64_t)n * k;
    const int64_t actStride = (int64_t)m * k;
    const int64_t outputStride = (int64_t)m * n;
    auto setBatch = [&](cublasLtMatrixLayout_t layout, int64_t stride) {
      return ok(cublasLtMatrixLayoutSetAttribute(
                    layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                    &batch, sizeof(batch)),
                "set batch count", err) &&
             ok(cublasLtMatrixLayoutSetAttribute(
                    layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                    &stride, sizeof(stride)),
                "set batch stride", err);
    };
    if (!setBatch(layoutWeight, weightStride) ||
        !setBatch(layoutAct, actStride) ||
        !setBatch(layoutC, outputStride) ||
        !setBatch(layoutD, outputStride)) return false;
    if (!ok(cublasLtMatmulDescSetAttribute(
                matmul, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                &weightScale, sizeof(weightScale)),
            "set weight scale pointer", err) ||
        !ok(cublasLtMatmulDescSetAttribute(
                matmul, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                &actScale, sizeof(actScale)),
            "set activation scale pointer", err)) return false;
    if (!ok(cublasLtMatmulPreferenceCreate(&preference),
            "cublasLtMatmulPreferenceCreate", err)) return false;
    size_t workspaceBytes = kCublasWorkspaceBytes;
    if (!ok(cublasLtMatmulPreferenceSetAttribute(
                preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &workspaceBytes, sizeof(workspaceBytes)),
            "set workspace preference", err)) return false;
    int returned = 0;
    if (!ok(cublasLtMatmulAlgoGetHeuristic(
                handle, matmul, layoutWeight, layoutAct, layoutC, layoutD,
                preference, 1, &heuristic, &returned),
            "cublasLtMatmulAlgoGetHeuristic", err)) return false;
    if (returned == 0) {
      err = "cuBLASLt found no batched NVFP4 algorithm";
      return false;
    }
    return true;
  }

  bool run(
      const void* act, const void* weight,
      const void* actScale, const void* weightScale,
      void* output, void* workspace, cudaStream_t stream,
      std::string& err) {
    if (!ok(cublasLtMatmulDescSetAttribute(
                matmul, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                &weightScale, sizeof(weightScale)),
            "update weight scale pointer", err) ||
        !ok(cublasLtMatmulDescSetAttribute(
                matmul, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                &actScale, sizeof(actScale)),
            "update activation scale pointer", err)) return false;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return ok(cublasLtMatmul(
                  handle, matmul, &alpha,
                  weight, layoutWeight,
                  act, layoutAct,
                  &beta, output, layoutC, output, layoutD,
                  &heuristic.algo, workspace, kCublasWorkspaceBytes, stream),
              "cublasLtMatmul", err);
  }
};

thread_local std::vector<std::unique_ptr<LagunaNvfp4BatchedGemm>>
    nativeGemmCache;

static LagunaNvfp4BatchedGemm* nativeGemm(
    int device, int batch, int m, int n, int k,
    const void* actScale, const void* weightScale, std::string& err) {
  for (auto& gemm : nativeGemmCache) {
    if (gemm->device == device && gemm->batch == batch &&
        gemm->m == m && gemm->n == n && gemm->k == k) {
      return gemm.get();
    }
  }
  auto gemm = std::make_unique<LagunaNvfp4BatchedGemm>();
  if (!gemm->init(
          device, batch, m, n, k, actScale, weightScale, err)) {
    return nullptr;
  }
  auto* result = gemm.get();
  nativeGemmCache.push_back(std::move(gemm));
  return result;
}

static bool runNativeNvfp4Prefill(
    const at::Tensor& x, const at::Tensor& topIdx, const at::Tensor& topW,
    const at::Tensor& gatePacked, const at::Tensor& gateScale,
    const at::Tensor& gateGlobal, const at::Tensor& upPacked,
    const at::Tensor& upScale, const at::Tensor& upGlobal,
    const at::Tensor& downPacked, const at::Tensor& downScale,
    const at::Tensor& downGlobal, int64_t numExperts, int64_t moeInt,
    int64_t hidden, at::Tensor& result, std::string& err) {
  const int tokens = (int)x.size(0);
  const int topK = (int)topIdx.size(1);
  const int pairs = tokens * topK;
  const int device = x.get_device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  auto byteOpts = x.options().dtype(at::kByte);
  auto intOpts = x.options().dtype(at::kInt);
  auto floatOpts = x.options().dtype(at::kFloat);
  auto bf16Opts = x.options().dtype(at::kBFloat16);

  at::Tensor gateScaleNative;
  at::Tensor upScaleNative;
  at::Tensor downScaleNative;
  if (!prepackScale(
          gateScale, numExperts, moeInt, hidden, stream,
          gateScaleNative, err) ||
      !prepackScale(
          upScale, numExperts, moeInt, hidden, stream,
          upScaleNative, err) ||
      !prepackScale(
          downScale, numExperts, hidden, moeInt, stream,
          downScaleNative, err)) {
    return false;
  }

  at::Tensor counts = at::zeros({numExperts}, intOpts);
  at::Tensor pairSlot = at::empty({pairs}, intOpts);
  at::Tensor maxRows = at::zeros({1}, intOpts);
  const int routeBlocks = std::min(
      (pairs + kThreads - 1) / kThreads, 65535);
  laguna_nvfp4_count_rows<<<routeBlocks, kThreads, 0, stream>>>(
      topIdx.data_ptr<int64_t>(), counts.data_ptr<int>(),
      pairSlot.data_ptr<int>(), maxRows.data_ptr<int>(),
      pairs, (int)numExperts);
  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    err = std::string("native route-count launch failed: ") +
          cudaGetErrorString(cudaStatus);
    return false;
  }
  int maxRowsHost = 0;
  cudaStatus = cudaMemcpyAsync(
      &maxRowsHost, maxRows.data_ptr<int>(), sizeof(maxRowsHost),
      cudaMemcpyDeviceToHost, stream);
  if (cudaStatus != cudaSuccess) {
    err = std::string("native route-count readback failed: ") +
          cudaGetErrorString(cudaStatus);
    return false;
  }
  cudaStatus = cudaStreamSynchronize(stream);
  if (cudaStatus != cudaSuccess) {
    err = std::string("native route-count synchronization failed: ") +
          cudaGetErrorString(cudaStatus);
    return false;
  }
  if (maxRowsHost <= 0 || maxRowsHost > tokens) {
    err = "native route count is outside [1, tokens]";
    return false;
  }
  const int rowsPerExpert = (maxRowsHost + 3) / 4 * 4;
  const int paddedRows =
      (rowsPerExpert + kNvfp4ScaleRows - 1) /
      kNvfp4ScaleRows * kNvfp4ScaleRows;
  laguna_nvfp4_finalize_rows<<<routeBlocks, kThreads, 0, stream>>>(
      topIdx.data_ptr<int64_t>(), pairSlot.data_ptr<int>(),
      pairs, rowsPerExpert, (int)numExperts);
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    err = std::string("native route-finalize launch failed: ") +
          cudaGetErrorString(cudaStatus);
    return false;
  }

  const int64_t actPackedBytes =
      numExperts * (int64_t)rowsPerExpert * hidden / 2;
  const int64_t actScaleBytes =
      numExperts * (int64_t)paddedRows * (hidden / kNvfp4Block);
  at::Tensor actPacked = at::empty({actPackedBytes}, byteOpts);
  at::Tensor actScale = at::empty({actScaleBytes}, byteOpts);
  const int64_t quantXWork =
      (int64_t)pairs * (hidden / kNvfp4Block);
  const int quantXBlocks = (int)std::min<int64_t>(
      (quantXWork + kThreads - 1) / kThreads, 65535);
  laguna_nvfp4_quantize_routed<<<
      quantXBlocks, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      pairSlot.data_ptr<int>(),
      reinterpret_cast<uint8_t*>(actPacked.data_ptr()),
      reinterpret_cast<uint8_t*>(actScale.data_ptr()),
      pairs, topK, rowsPerExpert, (int)hidden);
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    err = std::string("native activation quantization launch failed: ") +
          cudaGetErrorString(cudaStatus);
    return false;
  }

  at::Tensor workspace =
      at::empty({(int64_t)kCublasWorkspaceBytes}, byteOpts);
  at::Tensor gateRaw = at::empty(
      {numExperts, rowsPerExpert, moeInt}, bf16Opts);
  at::Tensor upRaw = at::empty(
      {numExperts, rowsPerExpert, moeInt}, bf16Opts);
  auto* first = nativeGemm(
      device, (int)numExperts, rowsPerExpert, (int)moeInt, (int)hidden,
      actScale.data_ptr(), gateScaleNative.data_ptr(), err);
  if (first == nullptr ||
      !first->run(
          actPacked.data_ptr(), gatePacked.data_ptr(),
          actScale.data_ptr(), gateScaleNative.data_ptr(),
          gateRaw.data_ptr(), workspace.data_ptr(), stream, err) ||
      !first->run(
          actPacked.data_ptr(), upPacked.data_ptr(),
          actScale.data_ptr(), upScaleNative.data_ptr(),
          upRaw.data_ptr(), workspace.data_ptr(), stream, err)) {
    return false;
  }

  const int downPaddedRows = paddedRows;
  const int64_t downPackedBytes =
      numExperts * (int64_t)rowsPerExpert * moeInt / 2;
  const int64_t downScaleBytes =
      numExperts * (int64_t)downPaddedRows * (moeInt / kNvfp4Block);
  at::Tensor downActPacked = at::empty({downPackedBytes}, byteOpts);
  at::Tensor downActScale = at::empty({downScaleBytes}, byteOpts);
  const int64_t quantDownWork =
      (int64_t)pairs * (moeInt / kNvfp4Block);
  const int quantDownBlocks = (int)std::min<int64_t>(
      (quantDownWork + kThreads - 1) / kThreads, 65535);
  laguna_nvfp4_swiglu_quantize<<<
      quantDownBlocks, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(gateRaw.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(upRaw.data_ptr()),
      gateGlobal.data_ptr<float>(), upGlobal.data_ptr<float>(),
      pairSlot.data_ptr<int>(),
      reinterpret_cast<uint8_t*>(downActPacked.data_ptr()),
      reinterpret_cast<uint8_t*>(downActScale.data_ptr()),
      pairs, rowsPerExpert, (int)moeInt);
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    err = std::string("native SwiGLU quantization launch failed: ") +
          cudaGetErrorString(cudaStatus);
    return false;
  }

  /* The down GEMM follows the quantization kernel on the same stream, so the
     caching allocator may safely recycle the first-stage temporaries. */
  actPacked = at::Tensor();
  actScale = at::Tensor();
  gateRaw = at::Tensor();
  upRaw = at::Tensor();

  at::Tensor downRaw = at::empty(
      {numExperts, rowsPerExpert, hidden}, bf16Opts);
  auto* second = nativeGemm(
      device, (int)numExperts, rowsPerExpert, (int)hidden, (int)moeInt,
      downActScale.data_ptr(), downScaleNative.data_ptr(), err);
  if (second == nullptr ||
      !second->run(
          downActPacked.data_ptr(), downPacked.data_ptr(),
          downActScale.data_ptr(), downScaleNative.data_ptr(),
          downRaw.data_ptr(), workspace.data_ptr(), stream, err)) {
    return false;
  }

  at::Tensor routed = at::zeros({tokens, hidden}, floatOpts);
  const int64_t scatterWork = (int64_t)pairs * hidden;
  const int scatterBlocks = (int)std::min<int64_t>(
      (scatterWork + kThreads - 1) / kThreads, 65535);
  laguna_nvfp4_scatter_down<<<
      scatterBlocks, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(downRaw.data_ptr()),
      downGlobal.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(topW.data_ptr()),
      pairSlot.data_ptr<int>(), routed.data_ptr<float>(),
      pairs, topK, rowsPerExpert, (int)hidden);
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    err = std::string("native down scatter launch failed: ") +
          cudaGetErrorString(cudaStatus);
    return false;
  }
  result = routed.to(at::kBFloat16);
  return true;
}

#endif  // CUDART_VERSION >= 12080

static lean_object* lagunaMoeIoError(const std::string& msg) {
  return lean_io_result_mk_error(
      lean_mk_io_user_error(lean_mk_string(msg.c_str())));
}

/* Validate one packed bank triple against its expected runtime sizes. */
static bool checkBank(
    const at::Tensor& packed, const at::Tensor& scales, const at::Tensor& global,
    int64_t E, int64_t outF, int64_t inF, const char* name, std::string& err) {
  if (!packed.is_cuda() || packed.scalar_type() != at::kByte ||
      packed.dim() != 3 || packed.size(0) != E || packed.size(1) != outF ||
      packed.size(2) != inF / 2) {
    err = std::string(name) + "Packed must be CUDA UInt8 [E, out, in/2]";
    return false;
  }
  if (!scales.is_cuda() || scales.scalar_type() != at::kFloat8_e4m3fn ||
      scales.dim() != 3 || scales.size(0) != E || scales.size(1) != outF ||
      scales.size(2) != inF / 16) {
    err = std::string(name) + "Scale must be CUDA Float8_e4m3fn [E, out, in/16]";
    return false;
  }
  if (!global.is_cuda() || global.scalar_type() != at::kFloat ||
      global.dim() != 1 || global.size(0) != E) {
    err = std::string(name) + "Global must be CUDA Float32 [E]";
    return false;
  }
  return true;
}

}  // namespace

extern "C" lean_object* lean_torch_laguna_moe_fp4_forward(
    b_lean_obj_arg x, b_lean_obj_arg topIdx, b_lean_obj_arg topW,
    b_lean_obj_arg gatePacked, b_lean_obj_arg gateScale, b_lean_obj_arg gateGlobal,
    b_lean_obj_arg upPacked, b_lean_obj_arg upScale, b_lean_obj_arg upGlobal,
    b_lean_obj_arg downPacked, b_lean_obj_arg downScale, b_lean_obj_arg downGlobal,
    uint64_t numExperts, uint64_t moeInt, uint64_t hidden) {
  try {
    auto x_ = borrowTensor(x);
    auto topIdx_ = borrowTensor(topIdx);
    auto topW_ = borrowTensor(topW);
    auto gp_ = borrowTensor(gatePacked);
    auto gs_ = borrowTensor(gateScale);
    auto gg_ = borrowTensor(gateGlobal);
    auto up_ = borrowTensor(upPacked);
    auto us_ = borrowTensor(upScale);
    auto ug_ = borrowTensor(upGlobal);
    auto dp_ = borrowTensor(downPacked);
    auto ds_ = borrowTensor(downScale);
    auto dg_ = borrowTensor(downGlobal);

    const int64_t E = static_cast<int64_t>(numExperts);
    const int64_t mi = static_cast<int64_t>(moeInt);
    const int64_t h = static_cast<int64_t>(hidden);

    /* Validation returns IO errors (no TORCH_CHECK: c10::Error cannot
       reliably unwind across the Lean FFI boundary on this toolchain). */
    if (E <= 0 || mi <= 0 || h <= 0 || (h % 32) != 0 || (mi % 32) != 0) {
      return lagunaMoeIoError(
          "laguna_moe_fp4_forward: require hidden%32==0 and moeInt%32==0");
    }
    if (!x_.is_cuda() || x_.scalar_type() != at::kBFloat16 ||
        x_.dim() != 2 || x_.size(1) != h) {
      return lagunaMoeIoError(
          "laguna_moe_fp4_forward: x must be CUDA BFloat16 [tokens, hidden]");
    }
    const int64_t tokens = x_.size(0);
    if (!topIdx_.is_cuda() || topIdx_.scalar_type() != at::kLong ||
        topIdx_.dim() != 2 || topIdx_.size(0) != tokens) {
      return lagunaMoeIoError(
          "laguna_moe_fp4_forward: topIdx must be CUDA Int64 [tokens, k]");
    }
    const int64_t k = topIdx_.size(1);
    if (!topW_.is_cuda() || topW_.scalar_type() != at::kBFloat16 ||
        topW_.dim() != 2 || topW_.size(0) != tokens || topW_.size(1) != k) {
      return lagunaMoeIoError(
          "laguna_moe_fp4_forward: topW must be CUDA BFloat16 [tokens, k]");
    }
    if (tokens < 1 || k < 1 || tokens * k > 65535) {
      return lagunaMoeIoError(
          "laguna_moe_fp4_forward: require 1 <= tokens*k <= 65535");
    }
    std::string bankErr;
    if (!checkBank(gp_, gs_, gg_, E, mi, h, "gate", bankErr) ||
        !checkBank(up_, us_, ug_, E, mi, h, "up", bankErr) ||
        !checkBank(dp_, ds_, dg_, E, h, mi, "down", bankErr)) {
      return lagunaMoeIoError("laguna_moe_fp4_forward: " + bankErr);
    }

    const c10::cuda::OptionalCUDAGuard guard(x_.device());
    auto xc = x_.contiguous();
    auto ic = topIdx_.contiguous();
    auto wc = topW_.contiguous();
    auto gpc = gp_.contiguous(); auto gsc = gs_.contiguous(); auto ggc = gg_.contiguous();
    auto upc = up_.contiguous(); auto usc = us_.contiguous(); auto ugc = ug_.contiguous();
    auto dpc = dp_.contiguous(); auto dsc = ds_.contiguous(); auto dgc = dg_.contiguous();

    const int64_t pairs = tokens * k;

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12080
    /* The SM12x tensor-core path wins once routing covers most of the expert
       bank.  Decode and small prefills intentionally stay on the pairwise
       kernel, which reads only the selected experts. */
    int nativeDevice = 0;
    int ccMajor = 0;
    cudaGetDevice(&nativeDevice);
    cudaDeviceGetAttribute(
        &ccMajor, cudaDevAttrComputeCapabilityMajor, nativeDevice);
    const char* disableNative = std::getenv("TYR_LAGUNA_DISABLE_NATIVE_FP4");
    const bool nativeDisabled =
        disableNative != nullptr && std::string(disableNative) != "0";
    const bool nativeEligible =
        !nativeDisabled && ccMajor == 12 && tokens > 1 &&
        pairs >= 2 * E && (h % 128) == 0 && (mi % 128) == 0;
    if (nativeEligible) {
      at::Tensor nativeOut;
      std::string nativeErr;
      if (!runNativeNvfp4Prefill(
              xc, ic, wc,
              gpc, gsc, ggc, upc, usc, ugc, dpc, dsc, dgc,
              E, mi, h, nativeOut, nativeErr)) {
        return lagunaMoeIoError(
            "laguna_moe_fp4_forward: native NVFP4 prefill failed: " +
            nativeErr);
      }
      return lean_io_result_mk_ok(giveTensor(nativeOut));
    }
#endif

    auto fopts = xc.options().dtype(at::kFloat);
    at::Tensor hidBuf = at::empty({pairs, mi}, fopts);
    at::Tensor routed = at::zeros({tokens, h}, fopts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const dim3 block(kThreads);
    const dim3 gridA((unsigned)((mi + kRowsPerBlock - 1) / kRowsPerBlock),
                     (unsigned)pairs);
    const dim3 gridB((unsigned)((h + kRowsPerBlock - 1) / kRowsPerBlock),
                     (unsigned)pairs);
    const size_t smemA = ((size_t)h + 256) * sizeof(float) + 256 * sizeof(float2);
    const size_t smemB = ((size_t)mi + 256) * sizeof(float) + 256 * sizeof(float2);

    /* Multi-token prefill groups routed pairs by expert. This avoids reading
       one expert's complete packed matrices independently for every token.
       The single-token decode path keeps the original pairwise kernels. */
    const size_t groupedSmemA =
        (size_t)kPrefillGroup * h * sizeof(__nv_bfloat16) +
        256 * sizeof(float) + 256 * sizeof(float2);
    const size_t groupedSmemB =
        (size_t)kPrefillGroup * mi * sizeof(float) +
        256 * sizeof(float) + 256 * sizeof(float2);
    int device = 0;
    int smCount = 1;
    int maxOptinSmem = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    cudaDeviceGetAttribute(
        &maxOptinSmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    bool grouped = tokens > 1 &&
        groupedSmemA <= (size_t)maxOptinSmem &&
        groupedSmemB <= (size_t)maxOptinSmem;
    if (grouped) {
      cudaError_t attrA = cudaFuncSetAttribute(
          laguna_moe_fp4_stage_a_grouped,
          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)groupedSmemA);
      cudaError_t attrB = cudaFuncSetAttribute(
          laguna_moe_fp4_stage_b_grouped,
          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)groupedSmemB);
      grouped = attrA == cudaSuccess && attrB == cudaSuccess;
    }

    /* data_ptr() as void* + cast: the typed data_ptr<T>() check rejects
       Float8_e4m3fn storage, and the kernels only read raw bytes. */
    at::Tensor dispatch;
    if (grouped) {
      const int64_t dispatchInts = 4 * pairs + E + (E + 1) + E +
                                   pairs + 1;
      dispatch = at::empty({dispatchInts}, xc.options().dtype(at::kInt));
      int* dispatchBase = dispatch.data_ptr<int>();
      auto* tasks = reinterpret_cast<ExpertTask*>(dispatchBase);
      int* counts = dispatchBase + 4 * pairs;
      int* offsets = counts + E;
      int* cursors = offsets + E + 1;
      int* pairOrder = cursors + E;
      int* taskCount = pairOrder + pairs;

      cudaMemsetAsync(counts, 0, (size_t)E * sizeof(int), stream);
      cudaMemsetAsync(cursors, 0, (size_t)E * sizeof(int), stream);
      const int dispatchBlocks =
          min((int)((pairs + kThreads - 1) / kThreads), smCount * 4);
      laguna_moe_count_experts<<<dispatchBlocks, kThreads, 0, stream>>>(
          ic.data_ptr<int64_t>(), counts, (int)pairs, (int)E);
      laguna_moe_build_layout<<<1, 1, 0, stream>>>(
          counts, offsets, tasks, taskCount, (int)E);
      laguna_moe_scatter_pairs<<<dispatchBlocks, kThreads, 0, stream>>>(
          ic.data_ptr<int64_t>(), offsets, cursors, pairOrder,
          (int)pairs, (int)E);

      int blocksPerSmA = 1;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocksPerSmA, laguna_moe_fp4_stage_a_grouped,
          kThreads, groupedSmemA);
      const int maxWorkA = (int)pairs *
          ((int)mi + kGroupedRowsPerBlock - 1) / kGroupedRowsPerBlock;
      const int groupedBlocksA = min(maxWorkA, smCount * max(1, blocksPerSmA));
      laguna_moe_fp4_stage_a_grouped<<<
          groupedBlocksA, block, groupedSmemA, stream>>>(
          reinterpret_cast<const uint8_t*>(gpc.data_ptr()),
          reinterpret_cast<const uint8_t*>(gsc.data_ptr()),
          reinterpret_cast<const uint8_t*>(upc.data_ptr()),
          reinterpret_cast<const uint8_t*>(usc.data_ptr()),
          ggc.data_ptr<float>(), ugc.data_ptr<float>(),
          reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr()),
          tasks, taskCount, pairOrder, hidBuf.data_ptr<float>(),
          (int)k, (int)mi, (int)h);
    } else {
      laguna_moe_fp4_stage_a<<<gridA, block, smemA, stream>>>(
          reinterpret_cast<const uint8_t*>(gpc.data_ptr()),
          reinterpret_cast<const uint8_t*>(gsc.data_ptr()),
          reinterpret_cast<const uint8_t*>(upc.data_ptr()),
          reinterpret_cast<const uint8_t*>(usc.data_ptr()),
          ggc.data_ptr<float>(), ugc.data_ptr<float>(),
          reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr()),
          ic.data_ptr<int64_t>(),
          hidBuf.data_ptr<float>(),
          (int)k, (int)mi, (int)h, (int)E);
    }
    cudaError_t errA = cudaGetLastError();
    if (errA != cudaSuccess) {
      return lagunaMoeIoError(std::string(
          "laguna_moe_fp4_forward: stage A launch failed: ") +
          cudaGetErrorString(errA));
    }

    if (grouped) {
      int* dispatchBase = dispatch.data_ptr<int>();
      auto* tasks = reinterpret_cast<ExpertTask*>(dispatchBase);
      int* counts = dispatchBase + 4 * pairs;
      int* offsets = counts + E;
      int* cursors = offsets + E + 1;
      int* pairOrder = cursors + E;
      int* taskCount = pairOrder + pairs;
      int blocksPerSmB = 1;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocksPerSmB, laguna_moe_fp4_stage_b_grouped,
          kThreads, groupedSmemB);
      const int maxWorkB = (int)pairs *
          ((int)h + kGroupedRowsPerBlock - 1) / kGroupedRowsPerBlock;
      const int groupedBlocksB = min(maxWorkB, smCount * max(1, blocksPerSmB));
      laguna_moe_fp4_stage_b_grouped<<<
          groupedBlocksB, block, groupedSmemB, stream>>>(
          reinterpret_cast<const uint8_t*>(dpc.data_ptr()),
          reinterpret_cast<const uint8_t*>(dsc.data_ptr()),
          dgc.data_ptr<float>(), hidBuf.data_ptr<float>(),
          reinterpret_cast<const __nv_bfloat16*>(wc.data_ptr()),
          tasks, taskCount, pairOrder, routed.data_ptr<float>(),
          (int)k, (int)mi, (int)h);
    } else {
      laguna_moe_fp4_stage_b<<<gridB, block, smemB, stream>>>(
          reinterpret_cast<const uint8_t*>(dpc.data_ptr()),
          reinterpret_cast<const uint8_t*>(dsc.data_ptr()),
          dgc.data_ptr<float>(),
          hidBuf.data_ptr<float>(),
          reinterpret_cast<const __nv_bfloat16*>(wc.data_ptr()),
          ic.data_ptr<int64_t>(),
          routed.data_ptr<float>(),
          (int)k, (int)mi, (int)h, (int)E);
    }
    cudaError_t errB = cudaGetLastError();
    if (errB != cudaSuccess) {
      return lagunaMoeIoError(std::string(
          "laguna_moe_fp4_forward: stage B launch failed: ") +
          cudaGetErrorString(errB));
    }

    at::Tensor out = routed.to(at::kBFloat16);
    return lean_io_result_mk_ok(giveTensor(out));
  } catch (const std::exception& e) {
    return lagunaMoeIoError(std::string(
        "laguna_moe_fp4_forward: exception: ") + e.what());
  }
}

#endif  // __CUDACC__
