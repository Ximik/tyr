/-
  Tyr/GPU/Codegen/Tests.lean

  Expect-style tests using #guard_msgs for GPU kernel DSL.
  These tests verify exact code generation output.
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.Primitives
import Tyr.GPU.Codegen.Loop
import Tyr.GPU.Codegen.EmitNew
import Tyr.GPU.Codegen.ArchConfig
import Tyr.GPU.Codegen.Arch
import Tyr.GPU.Codegen.Attribute
import Tyr.GPU.Codegen.Pipeline
import Tyr.GPU.Codegen.PersistentKernel
import Tyr.GPU.Codegen.KernelTemplate
import Tyr.GPU.Codegen.Launch
import Tyr.GPU.Codegen.TileDispatch

namespace Tyr.GPU.Codegen.Tests

open Tyr.GPU
open Tyr.GPU.Codegen

/-! ## Basic Declaration Tests -/

/-- Simple kernel with one tile -/
def simpleTileKernel : Kernel :=
  buildKernelM "simple_tile" .SM90 #[] do
    let _t : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void simple_tile(/* empty parameter list */) {
  rt<bf16, 64, 64, row_l> v0;
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel simpleTileKernel)

/-- Kernel with multiple tile types -/
def multiTileKernel : Kernel :=
  buildKernelM "multi_tile" .SM90 #[] do
    let _rt : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let _st : ST GpuFloat.Float32 32 64 .Col ← allocST .Float32 32 64 .Col
    let _rv : RV GpuFloat.Float32 64 ← allocRV .Float32 64
    let _sv : SV GpuFloat.BFloat16 128 ← allocSV .BFloat16 128

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void multi_tile(/* empty parameter list */) {
  extern __shared__ int __shm[];
  tma_swizzle_allocator al(__shm);
  rt<bf16, 64, 64, row_l> v0;
  auto &v1 = al.allocate<st<float, 32, 64>>(); // layout: col_l (Tyr .Col; TK has no col-layout ST, traversed transposed)
  rv<float, 64> v2;
  __shared__ sv<bf16, 128> v3;
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel multiTileKernel)

/-! ## MMA Tests -/

/-- MMA with AB mode -/
def mmaABKernel : Kernel :=
  buildKernelM "mma_ab" .SM90 #[] do
    let a : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let b : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
    let c : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
    mma c a b c

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void mma_ab(/* empty parameter list */) {
  rt<bf16, 64, 64, row_l> v0;
  rt<bf16, 64, 64, col_l> v1;
  rt<float, 64, 64, row_l> v2;
  warp::zero(v2);
  warp::mma_AB(v2, v0, v1, v2);
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel mmaABKernel)

/-- MMA with ABt mode (B transposed) -/
def mmaABtKernel : Kernel :=
  buildKernelM "mma_abt" .SM90 #[] do
    let a : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let b : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let c : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
    mmaT c a b c

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void mma_abt(/* empty parameter list */) {
  rt<bf16, 64, 64, row_l> v0;
  rt<bf16, 64, 64, row_l> v1;
  rt<float, 64, 64, row_l> v2;
  warp::zero(v2);
  warp::mma_ABt(v2, v0, v1, v2);
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel mmaABtKernel)

/-! ## Loop Tests -/

/-- Simple for loop -/
def simpleLoopKernel : Kernel :=
  buildKernelM "simple_loop" .SM90 #[] do
    forLoop 0 4 do
      sync

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void simple_loop(/* empty parameter list */) {
  for (int v0 = 0; v0 < 4; v0++) {
    warp::sync(0);
  }
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel simpleLoopKernel)

/-- Nested loops -/
def nestedLoopKernel : Kernel :=
  buildKernelM "nested_loop" .SM90 #[] do
    forLoop 0 2 do
      forLoop 0 3 do
        sync 1

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void nested_loop(/* empty parameter list */) {
  for (int v0 = 0; v0 < 2; v0++) {
    for (int v1 = 0; v1 < 3; v1++) {
      warp::sync(1);
    }
  }
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel nestedLoopKernel)

/-! ## Architecture Guard Tests -/

/-- SM80 (Ampere) kernel -/
def sm80Kernel : Kernel :=
  buildKernelM "sm80_kernel" .SM80 #[] do
    let _t : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_AMPERE)
__global__ void sm80_kernel(/* empty parameter list */) {
  rt<bf16, 64, 64, row_l> v0;
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel sm80Kernel)

/-- SM100 (Blackwell) kernel -/
def sm100Kernel : Kernel :=
  buildKernelM "sm100_kernel" .SM100 #[] do
    let _t : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_BLACKWELL)
__global__ void sm100_kernel(/* empty parameter list */) {
  rt<bf16, 64, 64, row_l> v0;
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel sm100Kernel)

/-! ## Reduction and Broadcast Tests -/

/-- Row reduction operations -/
def rowReductionKernel : Kernel :=
  buildKernelM "row_reduction" .SM90 #[] do
    let t : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    let v : RV GpuFloat.Float32 64 ← allocRV .Float32 64
    rowSum v t
    rowMax v t
    rowSumAccum v t v

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void row_reduction(/* empty parameter list */) {
  rt<float, 64, 64, row_l> v0;
  rv<float, 64, ducks::rv_layout::ortho> v1;
  warp::row_sum(v1, v0);
  warp::row_max(v1, v0);
  warp::row_sum(v1, v0, v1);
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel rowReductionKernel)

/-- Explicit RV layout conversion keeps a row reduction in `ortho` layout while
    adapting its consumer to the `align` layout required by a column map. -/
def rvLayoutConversionKernel : Kernel :=
  buildKernelM "rv_layout_conversion" .SM90 #[] do
    let t : RT GpuFloat.Float32 16 16 ← allocRT .Float32 16 16
    let reduced : RV GpuFloat.Float32 16 ← allocRV .Float32 16
    let broadcast : RV GpuFloat.Float32 16 ← allocRV .Float32 16
    rowSum reduced t
    convertVecLayout broadcast reduced
    subRow t t broadcast

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void rv_layout_conversion(/* empty parameter list */) {
  rt<float, 16, 16, row_l> v0;
  rv<float, 16, ducks::rv_layout::ortho> v1;
  rv<float, 16, ducks::rv_layout::align> v2;
  warp::row_sum(v1, v0);
  warp::copy(v2, v1);
  warp::sub_col(v0, v0, v2);
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel rvLayoutConversionKernel)

/-- Column broadcast operations -/
def colBroadcastKernel : Kernel :=
  buildKernelM "col_broadcast" .SM90 #[] do
    let t : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    let v : RV GpuFloat.Float32 64 ← allocRV .Float32 64
    subCol t t v
    mulCol t t v
    divCol t t v

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void col_broadcast(/* empty parameter list */) {
  rt<float, 64, 64, row_l> v0;
  rv<float, 64, ducks::rv_layout::ortho> v1;
  warp::sub_row(v0, v0, v1);
  warp::mul_row(v0, v0, v1);
  warp::div_row(v0, v0, v1);
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel colBroadcastKernel)

/-! ## Masking Tests -/

/-- Causal and triangular masks -/
def maskKernel : Kernel :=
  buildKernelM "mask_ops" .SM90 #[] do
    let t : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    makeCausal t t (some (-1e10))
    tril t t 0 none
    triu t t 1 (some 0.0)

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void mask_ops(/* empty parameter list */) {
  rt<float, 64, 64, row_l> v0;
  warp::make_causal(v0, v0, -10000000000.000000);
  warp::tril(v0, v0, 0);
  warp::triu(v0, v0, 1, 0.000000);
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel maskKernel)

/-! ## Memory Operations Tests -/

/-- Load, store, and atomic operations -/
def memoryKernel : Kernel :=
  buildKernelM "memory_ops" .SM90 #[] do
    let r : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let s : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
    load r s
    store s r
    let rF : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    let sF : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64
    storeAdd sF rF

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

template<typename Dst, typename Src>
__device__ inline void store_add(Dst &dst, const Src &src) {
  kittens::warp::store(dst, src);
}
template<typename Dst, typename Src, typename Offset>
__device__ inline void store_add(Dst &dst, const Src &src, const Offset &offset) {
  kittens::warp::store(dst, src, offset);
}

#if defined(KITTENS_HOPPER)
__global__ void memory_ops(/* empty parameter list */) {
  extern __shared__ int __shm[];
  tma_swizzle_allocator al(__shm);
  rt<bf16, 64, 64, row_l> v0;
  auto &v1 = al.allocate<st<bf16, 64, 64>>(); // layout: row_l
  warp::load(v0, v1);
  warp::store(v1, v0);
  rt<float, 64, 64, row_l> v2;
  auto &v3 = al.allocate<st<float, 64, 64>>(); // layout: row_l
  store_add(v3, v2);
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel memoryKernel)

/-! ## Parameter Tests -/

/-- Kernel with typed parameters -/
def paramKernel : Kernel :=
  buildKernelM "with_params" .SM90 #[
    { name := "x_ptr", dtype := .BFloat16, isPointer := true },
    { name := "y_ptr", dtype := .Float32, isPointer := true },
    { name := "n", dtype := .Float32, isPointer := false }
  ] do
    let _t : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void with_params(gl<bf16, -1, -1, -1, -1> v0, gl<float, -1, -1, -1, -1> v1, uint64_t v2) {
  rt<bf16, 64, 64, row_l> v3;
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel paramKernel)

/-! ## Complete Mini-FlashAttention Test -/

/-- Minimal FlashAttention structure -/
def miniFlashAttn : Kernel :=
  buildKernelM "mini_flash_attn" .SM90 #[
    { name := "Q", dtype := .BFloat16, isPointer := true },
    { name := "K", dtype := .BFloat16, isPointer := true },
    { name := "V", dtype := .BFloat16, isPointer := true },
    { name := "O", dtype := .BFloat16, isPointer := true }
  ] do
    comment "Tiles"
    let q : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let k : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let v : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
    let s : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    let o : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
    let p : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    comment "Vectors"
    let rowMax : RV GpuFloat.Float32 64 ← negInftyRV .Float32 64
    let rowSum : RV GpuFloat.Float32 64 ← allocRV .Float32 64
    comment "Shared"
    let qS : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
    let kS : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
    let vS : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col

    load q qS
    forLoop 0 4 do
      load k kS
      load v vS
      mmaT s q k s
      makeCausal s s (some (-1e10))
      rowMaxAccum rowMax s rowMax
      subCol s s rowMax
      exp s s
      rowSumAccum rowSum s rowSum
      convert p s
      mma o p v o
      sync
    divCol o o rowSum

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void mini_flash_attn(gl<bf16, -1, -1, -1, -1> v0, gl<bf16, -1, -1, -1, -1> v1, gl<bf16, -1, -1, -1, -1> v2, gl<bf16, -1, -1, -1, -1> v3) {
  extern __shared__ int __shm[];
  tma_swizzle_allocator al(__shm);
  // Tiles
  rt<bf16, 64, 64, row_l> v4;
  rt<bf16, 64, 64, row_l> v5;
  rt<bf16, 64, 64, col_l> v6;
  rt<float, 64, 64, row_l> v7;
  rt<float, 64, 64, row_l> v8;
  warp::zero(v8);
  rt<bf16, 64, 64, row_l> v9;
  // Vectors
  rv<float, 64, ducks::rv_layout::ortho> v10;
  warp::neg_infty(v10);
  rv<float, 64, ducks::rv_layout::ortho> v11;
  // Shared
  auto &v12 = al.allocate<st<bf16, 64, 64>>(); // layout: row_l
  auto &v13 = al.allocate<st<bf16, 64, 64>>(); // layout: row_l
  auto &v14 = al.allocate<st<bf16, 64, 64>>(); // layout: col_l (Tyr .Col; TK has no col-layout ST, traversed transposed)
  warp::load(v4, v12);
  for (int v15 = 0; v15 < 4; v15++) {
    warp::load(v5, v13);
    warp::load(v6, v14);
    warp::mma_ABt(v7, v4, v5, v7);
    warp::make_causal(v7, v7, -10000000000.000000);
    warp::row_max(v10, v7, v10);
    warp::sub_row(v7, v7, v10);
    warp::exp(v7, v7);
    warp::row_sum(v11, v7, v11);
    warp::copy(v9, v7);
    warp::mma_AB(v8, v9, v6, v8);
    warp::sync(0);
  }
  warp::div_row(v8, v8, v11);
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel miniFlashAttn)

/-! ## Architecture Capability Tests -/

-- Test architecture capabilities
#guard GpuArch.SM80.capabilities.hasTMA = false
#guard GpuArch.SM90.capabilities.hasTMA = true
#guard GpuArch.SM100.capabilities.hasTMA = true

#guard GpuArch.SM80.capabilities.hasWGMMA = false
#guard GpuArch.SM90.capabilities.hasWGMMA = true
#guard GpuArch.SM100.capabilities.hasWGMMA = false

#guard GpuArch.SM80.capabilities.hasFP8 = false
#guard GpuArch.SM90.capabilities.hasFP8 = true
#guard GpuArch.SM100.capabilities.hasFP8 = true

-- Test dtype support
#guard GpuArch.SM80.supportsDtype .BFloat16 = true
#guard GpuArch.SM80.supportsDtype .FP8E4M3 = false
#guard GpuArch.SM90.supportsDtype .FP8E4M3 = true
#guard GpuArch.SM100.supportsDtype .FP8E5M2 = true
#guard GpuArch.SM100.supportsDtype .FP8E8M0 = true
#guard GpuArch.SM100.supportsDtype .FP4E2M1X2 = true

-- Test default config generation
#guard (ArchKernelConfig.default .SM80).pipelineStages = 2
#guard (ArchKernelConfig.default .SM90).pipelineStages = 4
#guard (ArchKernelConfig.default .SM100).pipelineStages = 4

#guard (ArchKernelConfig.default .SM80).useWGMMA = false
#guard (ArchKernelConfig.default .SM90).useWGMMA = true

/-! ## Polymorphic Kernel Tests -/

open Arch in
/-- Example polymorphic kernel that adapts to different GPU architectures.
    When marked with @[gpu_kernel] (no arch argument), it generates
    kernel variants for SM80, SM90, and SM100 automatically.

    The function takes an ArchLevel parameter and returns ArchKernelM arch Unit.
    Instance resolution happens at compile time when the kernel is instantiated
    for a specific architecture. -/
def examplePolyMatmul [HasMMA arch] [ArchConfig arch]
    (_A : GPtr .BFloat16) (_B : GPtr .BFloat16) (_C : GPtr .BFloat16)
    : ArchKernelM arch Unit := do
  -- Get architecture-specific configuration via typeclass
  let cfg := ArchConfig.toRecord (arch := arch)
  let (tileM, tileN, _) := cfg.mmaTileSize

  -- Emit architecture info as a comment
  archComment s!"Tile: {tileM}x{tileN}, TMA: {cfg.hasTMA}, WGMMA: {cfg.hasWGMMA}"

  -- Allocate tiles
  let a ← ArchKernelM.liftPortable (allocRT .BFloat16 64 64)
  let b ← ArchKernelM.liftPortable (allocRT .BFloat16 64 64 .Col)
  let c ← ArchKernelM.liftPortable (zeroRT .Float32 64 64)

  -- smartMMA dispatches at compile time via typeclass:
  -- - Ampere: plain mma_AB
  -- - Hopper/Blackwell: mma_fence + mma_AB + mma_commit_group
  smartMMA c a b c

  archSync

/-- Test that the polymorphic kernel generates code for Ampere (SM80) -/
def polyKernelSM80 : Kernel :=
  buildKernelM "examplePolyMatmul_SM80" .SM80 #[
    { name := "A", dtype := .BFloat16, isPointer := true },
    { name := "B", dtype := .BFloat16, isPointer := true },
    { name := "C", dtype := .BFloat16, isPointer := true }
  ] (examplePolyMatmul (arch := .Ampere)
      (GPtr.mk ⟨0⟩ "A")
      (GPtr.mk ⟨1⟩ "B")
      (GPtr.mk ⟨2⟩ "C")).run

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_AMPERE)
__global__ void examplePolyMatmul_SM80(gl<bf16, -1, -1, -1, -1> v0, gl<bf16, -1, -1, -1, -1> v1, gl<bf16, -1, -1, -1, -1> v2) {
  // Tile: 16x16, TMA: false, WGMMA: false
  rt<bf16, 64, 64, row_l> v3;
  rt<bf16, 64, 64, col_l> v4;
  rt<float, 64, 64, row_l> v5;
  warp::zero(v5);
  warp::mma_AB(v5, v3, v4, v5);
  warp::sync(0);
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel polyKernelSM80)

/-- Test that the polymorphic kernel generates code for Hopper (SM90) with WGMMA -/
def polyKernelSM90 : Kernel :=
  buildKernelM "examplePolyMatmul_SM90" .SM90 #[
    { name := "A", dtype := .BFloat16, isPointer := true },
    { name := "B", dtype := .BFloat16, isPointer := true },
    { name := "C", dtype := .BFloat16, isPointer := true }
  ] (examplePolyMatmul (arch := .Hopper)
      (GPtr.mk ⟨0⟩ "A")
      (GPtr.mk ⟨1⟩ "B")
      (GPtr.mk ⟨2⟩ "C")).run

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void examplePolyMatmul_SM90(gl<bf16, -1, -1, -1, -1> v0, gl<bf16, -1, -1, -1, -1> v1, gl<bf16, -1, -1, -1, -1> v2) {
  // Tile: 64x64, TMA: true, WGMMA: true
  rt<bf16, 64, 64, row_l> v3;
  rt<bf16, 64, 64, col_l> v4;
  rt<float, 64, 64, row_l> v5;
  warp::zero(v5);
  warpgroup::mma_fence(v5);
  warp::mma_AB(v5, v3, v4, v5);
  warpgroup::mma_commit_group();
  warp::sync(0);
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel polyKernelSM90)

/-- Test that the polymorphic kernel generates code for Blackwell (SM100) -/
def polyKernelSM100 : Kernel :=
  buildKernelM "examplePolyMatmul_SM100" .SM100 #[
    { name := "A", dtype := .BFloat16, isPointer := true },
    { name := "B", dtype := .BFloat16, isPointer := true },
    { name := "C", dtype := .BFloat16, isPointer := true }
  ] (examplePolyMatmul (arch := .Blackwell)
      (GPtr.mk ⟨0⟩ "A")
      (GPtr.mk ⟨1⟩ "B")
      (GPtr.mk ⟨2⟩ "C")).run

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_BLACKWELL)
__global__ void examplePolyMatmul_SM100(gl<bf16, -1, -1, -1, -1> v0, gl<bf16, -1, -1, -1, -1> v1, gl<bf16, -1, -1, -1, -1> v2) {
  // Tile: 64x64, TMA: true, WGMMA: false
  rt<bf16, 64, 64, row_l> v3;
  rt<bf16, 64, 64, col_l> v4;
  rt<float, 64, 64, row_l> v5;
  warp::zero(v5);
  warpgroup::mma_fence(v5);
  warp::mma_AB(v5, v3, v4, v5);
  warpgroup::mma_commit_group();
  warp::sync(0);
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel polyKernelSM100)

/-! ## Polymorphic Kernel Pattern

The recommended pattern for architecture-polymorphic kernels uses typeclass constraints
for compile-time dispatch. When instantiated with a concrete architecture, the correct
implementation is selected at compile time with zero runtime overhead.

```lean
-- Define polymorphic kernel with typeclass constraint
def myPolyKernel [HasMMA arch] (input : GPtr .BFloat16) (output : GPtr .BFloat16)
    : ArchKernelM arch Unit := do
  let cfg := ArchConfig.toRecord (arch := arch)
  -- Tile sizes, etc. are resolved at compile time
  let a ← ArchKernelM.liftPortable (allocRT .BFloat16 64 64)
  ...
  -- smartMMA dispatches via typeclass - compile-time selection
  smartMMA dst a b c

-- Instantiate for each architecture (generates different code)
def kernel_SM80 := buildKernelM "k" .SM80 #[...] (myPolyKernel (arch := .Ampere) ...).run
def kernel_SM90 := buildKernelM "k" .SM90 #[...] (myPolyKernel (arch := .Hopper) ...).run
```

The `@[gpu_kernel]` attribute can automate this instantiation when enhanced to detect
the typeclass-constrained pattern.
-/

/-! ## Blackwell TMEM / Cluster / tcgen05 Tests -/

/-- TMEM tile allocation emits tt<> declaration -/
def tmemAllocKernel : Kernel :=
  buildKernelM "tmem_alloc" .SM100 #[] do
    let _acc : TT GpuFloat.Float32 128 128 ← allocTT .Float32 128 128
    let _accZ : TT GpuFloat.Float32 64 64 ← zeroTT .Float32 64 64

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_BLACKWELL)
__global__ void tmem_alloc(/* empty parameter list */) {
  tt<float, 128, 128> v0;
  tt<float, 64, 64> v1;
  warp::zero(v1);
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel tmemAllocKernel)

/-- Cluster coordinate accessor emits clusterIdx() -/
def clusterIdxKernel : Kernel :=
  buildKernelM "cluster_idx" .SM100 #[] do
    let _cidx ← clusterIdx 0

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_BLACKWELL)
__global__ void cluster_idx(/* empty parameter list */) {
  int v0 = clusterIdx().x;
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel clusterIdxKernel)

/-- tcgen05 MMA operations emit mm2/mma2 calls -/
def tcgen05MmaKernel : Kernel :=
  buildKernelM "tcgen05_mma" .SM100 #[] do
    let acc ← allocTT .Float32 64 64
    let a ← allocST .BFloat16 64 64
    let b ← allocST .BFloat16 64 64 .Col
    tcgen05Mm .ABt acc a b
    tcgen05Mma .ABt acc a b acc
    let sem ← allocSemaphore
    tcgen05Commit sem 2

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_BLACKWELL)
__global__ void tcgen05_mma(/* empty parameter list */) {
  extern __shared__ int __shm[];
  tma_swizzle_allocator al(__shm);
  tt<float, 64, 64> v0;
  auto &v1 = al.allocate<st<bf16, 64, 64>>(); // layout: row_l
  auto &v2 = al.allocate<st<bf16, 64, 64>>(); // layout: col_l (Tyr .Col; TK has no col-layout ST, traversed transposed)
  warpgroup::mm2_ABt(v0, v1, v2);
  warpgroup::mma2_ABt(v0, v1, v2, v0);
  __shared__ semaphore v3;
  detail::tcgen05::commit<2>(v3);
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel tcgen05MmaKernel)

/-- Cluster TMA and barrier operations -/
def clusterTmaKernel : Kernel :=
  buildKernelM "cluster_tma" .SM100 #[
    { name := "A", dtype := .BFloat16, isPointer := true }
  ] do
    let sem ← allocSemaphore
    clusterArrive sem
    clusterWait sem

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_BLACKWELL)
__global__ void cluster_tma(gl<bf16, -1, -1, -1, -1> v0) {
  __shared__ semaphore v1;
  cluster::arrive(v1);
  cluster::wait(v1);
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel clusterTmaKernel)

/-- TMEM pool operations (allocate, provision, subtile) -/
def tmemPoolKernel : Kernel :=
  buildKernelM "tmem_pool" .SM100 #[] do
    -- Pool is raw because TMEMPool needs a `tensor_allocator<>` decl,
    -- which the emitter generates from a raw snippet for now.
    emitRaw "tensor_allocator<1, 2, false> tm_alloc;"
    let pool : TMEMPool := ⟨⟨0⟩⟩  -- placeholder VarId for the allocator
    tmemProvision pool 2
    tmemDeprovision pool

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_BLACKWELL)
__global__ void tmem_pool(/* empty parameter list */) {
  tensor_allocator<1, 2, false> tm_alloc;
  if(elect_one) v0.provision<2>(tmem_addr);
  if(elect_one) v0.deprovision();
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel tmemPoolKernel)

/-- Concrete one-CTA TMEM allocation and TMEM-to-register load path. -/
def tmemRegisterLoadKernel : Kernel :=
  buildKernelM "tmem_register_load" .SM100 #[] do
    let pool ← allocTMEMPool 1 1
    let tile ← tmemAllocate pool .Float32 64 64 0
    let fragment ← allocRT .Float32 16 64
    tensorLoadAsync fragment tile
    tensorLoadWait

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_BLACKWELL)
__global__ void tmem_register_load(/* empty parameter list */) {
  tensor_allocator<1, 1> v0{};
  auto v1 = v0.allocate<tt<float, 64, 64>>(0, 0);
  rt<float, 16, 64, row_l> v2;
  warpgroup::load_async(v2, v1);
  tensor_load_wait();
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel tmemRegisterLoadKernel)

/-- Static and runtime-indexed shared-tile fragment loads. -/
def sharedSubtileLoadKernel : Kernel :=
  buildKernelM "shared_subtile_load" .SM90 #[] do
    let src ← allocST .BFloat16 128 64
    let dst ← allocRT .BFloat16 64 64
    loadSubtile dst src 1 0
    let row ← constIntVal 0 "fragment_row"
    loadSubtileVal dst src row row

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void shared_subtile_load(/* empty parameter list */) {
  extern __shared__ int __shm[];
  tma_swizzle_allocator al(__shm);
  auto &v0 = al.allocate<st<bf16, 128, 64>>(); // layout: row_l
  rt<bf16, 64, 64, row_l> v1;
  warp::load(v1, v0.template subtile<64, 64>(make_int2(1, 0)));
  int v2 = 0;
  warp::load(v1, v0.template subtile<64, 64>(make_int2(v2, v2)));
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel sharedSubtileLoadKernel)

/-! ## Pipeline Tests -/

/-- 3-stage ring buffer emits 3 shared tiles + 3 semaphores -/
def ringBuffer3StageKernel : Kernel :=
  buildKernelM "ring_buffer_3" .SM90 #[] do
    let _rb ← allocRingBuffer .BFloat16 64 64 .Row 3

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void ring_buffer_3(/* empty parameter list */) {
  extern __shared__ int __shm[];
  tma_swizzle_allocator al(__shm);
  auto &v0 = al.allocate<st<bf16, 64, 64>>(); // layout: row_l
  __shared__ semaphore v1;
  auto &v2 = al.allocate<st<bf16, 64, 64>>(); // layout: row_l
  __shared__ semaphore v3;
  auto &v4 = al.allocate<st<bf16, 64, 64>>(); // layout: row_l
  __shared__ semaphore v5;
  if (threadIdx.x == 0) {
    init_semaphore(v1, 1, 0);
  }
  if (threadIdx.x == 0) {
    init_semaphore(v3, 0, 0);
  }
  if (threadIdx.x == 0) {
    init_semaphore(v5, 0, 0);
  }
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel ringBuffer3StageKernel)

/-- Warp-specialized pipeline emits ifWarpGroup blocks -/
def warpSpecPipelineKernel : Kernel :=
  buildKernelM "warp_spec_pipeline" .SM90 #[] do
    pipelinedRingLoop { numIters := 4, depth := 2, warpSpecialized := true }
      (fun _ => sync)
      (fun _ => sync 1)

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void warp_spec_pipeline(/* empty parameter list */) {
  // Pipeline prologue: fill 2 stages
  if (kittens::warpgroup::groupid() == 0) {
    warp::sync(0);
  }
  if (kittens::warpgroup::groupid() == 0) {
    warp::sync(0);
  }
  warp::sync(0);
  // Pipeline main loop: 2 iterations
  for (int v0 = 0; v0 < 2; v0++) {
    if (kittens::warpgroup::groupid() == 0) {
      warp::sync(0);
    }
    if (kittens::warpgroup::groupid() == 1) {
      warp::sync(1);
    }
    warp::sync(0);
  }
  // Pipeline epilogue: drain 2 stages
  if (kittens::warpgroup::groupid() == 1) {
    warp::sync(1);
  }
  warp::sync(0);
  if (kittens::warpgroup::groupid() == 1) {
    warp::sync(1);
  }
  warp::sync(0);
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel warpSpecPipelineKernel)

/-! ## Persistent Kernel Tests -/

/-- Fixed stride emits for loop with gridDim stride -/
def fixedStrideKernel : Kernel :=
  buildKernelM "fixed_stride" .SM90 #[
    { name := "totalWork", dtype := .Float32, isPointer := false, scalarTy := .UInt32 }
  ] do
    let totalWork : KVal UInt32 := ⟨⟨0⟩, "totalWork"⟩
    persistentLoop { mode := .fixedStride } totalWork.id fun _wi => do
      comment "work item body"

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void fixed_stride(uint32_t v0) {
  // Persistent loop (fixed stride)
  int v1 = blockIdx.x;
  int v2 = gridDim.x;
  int v3 = 1;
  for (auto v4 = v1; v4 < v0; v4 += v2) {
    // work item body
  }
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel fixedStrideKernel)

/-- workIdToSwizzledCoord emits div/mod scalar ops -/
def swizzleCoordKernel : Kernel :=
  buildKernelM "swizzle_coord" .SM90 #[] do
    let wid ← freshVar
    emit (.constInt wid 42)
    let ncols ← freshVar
    emit (.constInt ncols 16)
    let (_row, _col) ← workIdToSwizzledCoord wid ncols 8

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void swizzle_coord(/* empty parameter list */) {
  int v0 = 42;
  int v1 = 16;
  // L2 swizzle (group size = 8)
  int v2 = 8;
  int v4 = 7;
  auto v3 = v1 + v4;
  auto v5 = v3 / v2;
  auto v6 = v0 / v2;
  auto v7 = v0 % v2;
  auto v8 = v6 / v5;
  auto v9 = v6 % v5;
  auto v10 = v9 * v2;
  auto v11 = v10 + v7;
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel swizzleCoordKernel)

/-! ## KernelTemplate Tests -/

/-- 2-phase template emits both phases with barrier between -/
def twoPhaseKernel : Kernel :=
  let tmpl : KernelTemplate := {
    name := "two_phase"
    arch := .SM90
    phases := #[
      { name := "phase1", emit := do
          let _ ← allocRT .BFloat16 64 64
          pure () },
      { name := "phase2", emit := do
          let _ ← allocRT .Float32 64 64
          pure () }
    ]
  }
  tmpl.build

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void two_phase(/* empty parameter list */) {
  // === Phase: phase1 ===
  rt<bf16, 64, 64, row_l> v0;
  warp::sync(0);
  // === Phase: phase2 ===
  rt<float, 64, 64, row_l> v1;
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel twoPhaseKernel)

/-- FusedGemm emits GEMM loop + epilogue callback -/
def fusedGemmTestKernel : Kernel :=
  buildKernelM "fused_gemm" .SM90 #[] do
    let lhs ← allocST .BFloat16 64 64
    let rhs ← allocST .BFloat16 64 64 .Col
    fusedGemm { tileM := 64, tileN := 64, tileK := 64, kBlocks := 4 } lhs rhs fun accum => do
      let out ← allocRT .BFloat16 64 64
      convert out accum

/--
info: #include <type_traits>
#include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void fused_gemm(/* empty parameter list */) {
  extern __shared__ int __shm[];
  tma_swizzle_allocator al(__shm);
  auto &v0 = al.allocate<st<bf16, 64, 64>>(); // layout: row_l
  auto &v1 = al.allocate<st<bf16, 64, 64>>(); // layout: col_l (Tyr .Col; TK has no col-layout ST, traversed transposed)
  // Fused GEMM: 64x64x64, 4 K-blocks
  rt<float, 64, 64, row_l> v2;
  warp::zero(v2);
  rt<bf16, 64, 64, row_l> v3;
  rt<bf16, 64, 64, col_l> v4;
  for (int v5 = 0; v5 < 4; v5++) {
    warp::load(v3, v0);
    warp::load(v4, v1);
    warp::mma_AB(v2, v3, v4, v2);
    warp::sync(0);
  }
  // GEMM epilogue
  rt<bf16, 64, 64, row_l> v6;
  warp::copy(v6, v2);
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel fusedGemmTestKernel)

/-! ## TileDispatch Tests -/

-- selectVariant picks correct variant based on problem size
#guard (selectVariant { name := "test", variants := #[
  { name := "large", tileM := 128, tileN := 128, minM := 512, minN := 0 },
  { name := "small", tileM := 64, tileN := 64, minM := 0, minN := 0 }
] } 1024 1024).name = "large"

#guard (selectVariant { name := "test", variants := #[
  { name := "large", tileM := 128, tileN := 128, minM := 512, minN := 0 },
  { name := "small", tileM := 64, tileN := 64, minM := 0, minN := 0 }
] } 64 64).name = "small"

-- computeGridDims returns correct grid
#guard (computeGridDims 4096 4096 64 64).x = 64
#guard (computeGridDims 4096 4096 64 64).y = 64
#guard (computeGridDims 1000 2000 64 128).x = 16
#guard (computeGridDims 1000 2000 64 128).y = 16

/-! ## Launch Tests -/

-- computeLaunchConfig returns correct grid/block for problem sizes
private def dummyKernelForLaunch : Kernel :=
  { name := "dummy", arch := .SM90, family := .Hopper, params := #[], body := #[], sharedMemBytes := 49152 }

#guard (computeLaunchConfig dummyKernelForLaunch 4096 4096 64 64).grid.x = 64
#guard (computeLaunchConfig dummyKernelForLaunch 4096 4096 64 64).grid.y = 64
#guard (computeLaunchConfig dummyKernelForLaunch 4096 4096 64 64).sharedMemBytes = 49152

-- Persistent launch uses SM count
#guard (computePersistentLaunchConfigForArch dummyKernelForLaunch .SM90 2).grid.x = 264
#guard (computePersistentLaunchConfigForArch dummyKernelForLaunch .SM90 2).cooperative = true

-- Batched launch includes z dimension
#guard (computeBatchedLaunchConfig dummyKernelForLaunch 32 4096 4096 64 64).grid.z = 32

end Tyr.GPU.Codegen.Tests
