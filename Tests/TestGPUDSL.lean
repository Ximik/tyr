/-
  Tests/TestGPUDSL.lean

  Tests for the high-level GPU kernel DSL extensions:
  - Notation.lean: Expression-style operators and functions
  - Constraints.lean: Type-level dimension constraints
  - Macros.lean: Pattern macros for common kernel patterns
-/
import Tyr.GPU.Types
import Tyr.GPU.Kernels.RKCombine
import Tyr.GPU.Kernels.BrownianSample
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.Primitives
import Tyr.GPU.Codegen.Loop
import Tyr.GPU.Codegen.EmitNew
import Tyr.GPU.Codegen.Notation
import Tyr.GPU.Codegen.Constraints
import Tyr.GPU.Codegen.Macros
import LeanTest

namespace Tests.GPUDSL

open Tyr.GPU
open Tyr.GPU.Codegen
open LeanTest

/-! ## Notation Tests: Matrix Multiply Operators -/

/-- Test matmul function returns correctly typed tile -/
@[test]
def testMatmulFunction : IO Unit := do
  let kernel := buildKernelM "test_matmul" .SM90 #[] do
    let a ← allocRT .BFloat16 64 64
    let b ← allocRT .BFloat16 64 64 .Col
    let _c ← matmul a b .Float32
    comment "matmul complete"
    pure ()

  -- Should have: 2 decls for a,b + 1 decl for c + 1 mm + 1 comment
  assertTrue (kernel.body.size ≥ 4) "Should have allocations and mm"

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "mm_AB(") "Should have mm_AB call"

/-- Test matmulT function (B transposed) -/
@[test]
def testMatmulTFunction : IO Unit := do
  let kernel := buildKernelM "test_matmul_t" .SM90 #[] do
    let a ← allocRT .BFloat16 64 64
    let b ← allocRT .BFloat16 64 64  -- Row-major for transpose
    let _c ← matmulT a b .Float32
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "mma_ABt(") "Should have mma_ABt call"

/-- Test matrix multiply infix notation ⬝ -/
@[test]
def testMatmulInfix : IO Unit := do
  let kernel := buildKernelM "test_matmul_infix" .SM90 #[] do
    let a ← allocRT .BFloat16 64 64
    let b ← allocRT .BFloat16 64 64 .Col
    let _c ← a ⬝ b  -- Uses matmulF32
    comment s!"Result allocated"
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "mm_AB(") "Should have mm_AB from ⬝ operator"

/-- Test transposed multiply infix notation ⬝ᵀ -/
@[test]
def testMatmulTInfix : IO Unit := do
  let kernel := buildKernelM "test_matmul_t_infix" .SM90 #[] do
    let a ← allocRT .BFloat16 64 64
    let b ← allocRT .BFloat16 64 64
    let _c ← a ⬝ᵀ b  -- Uses matmulTF32
    comment s!"Result allocated"
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "mma_ABt(") "Should have mma_ABt from ⬝ᵀ operator"

/-! ## Notation Tests: Element-wise Operators -/

/-- Test tile addition with + operator -/
@[test]
def testTileAddOperator : IO Unit := do
  let kernel := buildKernelM "test_tile_add" .SM90 #[] do
    let a ← allocRT .Float32 64 64
    let b ← allocRT .Float32 64 64
    let _c ← a + b  -- HAdd instance
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "add(") "Should have add operation"

/-- Test tile subtraction with - operator -/
@[test]
def testTileSubOperator : IO Unit := do
  let kernel := buildKernelM "test_tile_sub" .SM90 #[] do
    let a ← allocRT .Float32 64 64
    let b ← allocRT .Float32 64 64
    let _c ← a - b  -- HSub instance
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "sub(") "Should have sub operation"

/-- Test tile multiplication with * operator -/
@[test]
def testTileMulOperator : IO Unit := do
  let kernel := buildKernelM "test_tile_mul" .SM90 #[] do
    let a ← allocRT .Float32 64 64
    let b ← allocRT .Float32 64 64
    let _c ← a * b  -- HMul instance
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "mul(") "Should have mul operation"

/-- Test tile division with / operator -/
@[test]
def testTileDivOperator : IO Unit := do
  let kernel := buildKernelM "test_tile_div" .SM90 #[] do
    let a ← allocRT .Float32 64 64
    let b ← allocRT .Float32 64 64
    let _c ← a / b  -- HDiv instance
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "div(") "Should have div operation"

/-- Test scalar multiplication -/
@[test]
def testScalarMul : IO Unit := do
  let kernel := buildKernelM "test_scalar_mul" .SM90 #[] do
    let a ← allocRT .Float32 64 64
    let _b ← a * 0.5  -- tile * scalar
    let _c ← 2.0 * a  -- scalar * tile
    pure ()

  let code := generateKernel kernel
  -- Should have two mul operations with scalar
  assertTrue (code.containsSubstr "mul(") "Should have scalar mul"
  assertTrue (code.containsSubstr "0x1.0000000000000p-1f") "Should have 0.5 scalar"
  assertTrue (code.containsSubstr "0x1.0000000000000p1f") "Should have 2.0 scalar"

/-- Test scalar addition -/
@[test]
def testScalarAdd : IO Unit := do
  let kernel := buildKernelM "test_scalar_add" .SM90 #[] do
    let a ← allocRT .Float32 64 64
    let _b ← a + 1.0
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "add(") "Should have scalar add"
  assertTrue (code.containsSubstr "0x1.0000000000000p0f") "Should have 1.0 scalar"

/-! ## Notation Tests: Vector Operators -/

/-- Test vector addition -/
@[test]
def testVectorAdd : IO Unit := do
  let kernel := buildKernelM "test_vec_add" .SM90 #[] do
    let a ← allocRV .Float32 64
    let b ← allocRV .Float32 64
    let _c ← a + b  -- HAdd for RV
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "add(") "Should have vector add"

/-- Test vector subtraction -/
@[test]
def testVectorSub : IO Unit := do
  let kernel := buildKernelM "test_vec_sub" .SM90 #[] do
    let a ← allocRV .Float32 64
    let b ← allocRV .Float32 64
    let _c ← a - b
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "sub(") "Should have vector sub"

/-! ## Notation Tests: Broadcasting Functions -/

/-- Test subBroadcastCol -/
@[test]
def testSubBroadcastCol : IO Unit := do
  let kernel := buildKernelM "test_sub_broadcast_col" .SM90 #[] do
    let tile ← allocRT .Float32 64 64
    let vec ← allocRV .Float32 64
    let _result ← subBroadcastCol tile vec
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "sub_row(") "Should have sub_row broadcast"

/-- Test divBroadcastCol -/
@[test]
def testDivBroadcastCol : IO Unit := do
  let kernel := buildKernelM "test_div_broadcast_col" .SM90 #[] do
    let tile ← allocRT .Float32 64 64
    let vec ← allocRV .Float32 64
    let _result ← divBroadcastCol tile vec
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "div_row(") "Should have div_row broadcast"

/-- Test mulBroadcastRow -/
@[test]
def testMulBroadcastRow : IO Unit := do
  let kernel := buildKernelM "test_mul_broadcast_row" .SM90 #[] do
    let tile ← allocRT .Float32 64 64
    let vec ← allocRV .Float32 64
    let _result ← mulBroadcastRow tile vec
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "mul_col(") "Should have mul_col broadcast"

/-! ## Notation Tests: Reduction Functions -/

/-- Test reduceRowMax -/
@[test]
def testReduceRowMax : IO Unit := do
  let kernel := buildKernelM "test_reduce_row_max" .SM90 #[] do
    let tile ← allocRT .Float32 64 64
    let _maxVec ← reduceRowMax tile
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "row_max(") "Should have row_max reduction"

/-- Test reduceRowSum -/
@[test]
def testReduceRowSum : IO Unit := do
  let kernel := buildKernelM "test_reduce_row_sum" .SM90 #[] do
    let tile ← allocRT .Float32 64 64
    let _sumVec ← reduceRowSum tile
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "row_sum(") "Should have row_sum reduction"

/-- Test reduceColSum -/
@[test]
def testReduceColSum : IO Unit := do
  let kernel := buildKernelM "test_reduce_col_sum" .SM90 #[] do
    let tile ← allocRT .Float32 64 64
    let _sumVec ← reduceColSum tile
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "col_sum(") "Should have col_sum reduction"

/-! ## Notation Tests: Unary Expression Functions -/

/-- Test expTile -/
@[test]
def testExpTile : IO Unit := do
  let kernel := buildKernelM "test_exp_tile" .SM90 #[] do
    let a ← allocRT .Float32 64 64
    let _b ← expTile a
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "exp(") "Should have exp operation"

/-- Test logTile -/
@[test]
def testLogTile : IO Unit := do
  let kernel := buildKernelM "test_log_tile" .SM90 #[] do
    let a ← allocRT .Float32 64 64
    let _b ← logTile a
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "log(") "Should have log operation"

/-- Test tanhTile -/
@[test]
def testTanhTile : IO Unit := do
  let kernel := buildKernelM "test_tanh_tile" .SM90 #[] do
    let a ← allocRT .Float32 64 64
    let _b ← tanhTile a
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "tanh(") "Should have tanh operation"

/-- Test causalMask -/
@[test]
def testCausalMaskFunction : IO Unit := do
  let kernel := buildKernelM "test_causal_mask_fn" .SM90 #[] do
    let scores ← allocRT .Float32 64 64
    let _masked ← causalMask scores (-1e10)
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "make_causal(") "Should have make_causal"

/-! ## Constraints Tests: DivisibleBy -/

-- DivisibleBy instances exist for common sizes.
example : DivisibleBy 64 16 := inferInstance
example : DivisibleBy 128 16 := inferInstance
example : DivisibleBy 256 16 := inferInstance
example : DivisibleBy 64 8 := inferInstance
example : DivisibleBy 128 32 := inferInstance
example : DivisibleBy 256 128 := inferInstance

example : Mult16 64 := inferInstance
example : Mult16 128 := inferInstance

/-- Test constrained MMA compiles with valid dimensions -/
@[test]
def testConstrainedMMA : IO Unit := do
  let kernel := buildKernelM "test_constrained_mma" .SM90 #[] do
    let a ← allocRT .BFloat16 64 64
    let b ← allocRT .BFloat16 64 64 .Col
    let c ← zeroRT .Float32 64 64
    -- This uses mmaConstrained which requires [Mult16 M] [Mult16 K] [Mult16 N]
    mmaConstrained c a b c
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "mma_AB(") "Should have mma_AB call"

/-! ## Constraints Tests: Architecture Capabilities -/

example : HasTMA Arch.SM90 := inferInstance
example : HasWGMMA Arch.SM90 := inferInstance
example : HasFP8 Arch.SM90 := inferInstance
example : HasCluster Arch.SM90 := inferInstance

/-! ## Macros Tests: Double Buffering -/

/-- Test allocDoubleBuffer creates two buffers and semaphores -/
@[test]
def testAllocDoubleBuffer : IO Unit := do
  let kernel := buildKernelM "test_double_buffer" .SM90 #[] do
    let _db ← allocDoubleBuffer .BFloat16 64 64
    comment "Double buffer allocated"
    pure ()

  -- Should have: 2 ST decls + 2 semaphore decls + 2 init semaphore + 1 comment
  assertTrue (kernel.body.size ≥ 6) "Should have buffer and semaphore allocations"

  let code := generateKernel kernel
  -- Check for two shared tile declarations
  assertTrue (code.containsSubstr "st<bf16, 64, 64") "Should have shared tile decls"
  assertTrue (code.containsSubstr "semaphore") "Should have semaphore decls"
  assertTrue (code.containsSubstr "init_semaphore") "Should have semaphore init"

/-! ## Macros Tests: Warp Specialization -/

/-- Test asProducer helper -/
@[test]
def testAsProducer : IO Unit := do
  let kernel := buildKernelM "test_as_producer" .SM90 #[] do
    asProducer do
      comment "Producer code"
      sync
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "warpgroup::groupid() == 0") "Should check warpgroup 0"

/-- Test asConsumer helper -/
@[test]
def testAsConsumer : IO Unit := do
  let kernel := buildKernelM "test_as_consumer" .SM90 #[] do
    asConsumer do
      comment "Consumer code"
      sync
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "warpgroup::groupid() == 1") "Should check warpgroup 1"

/-- Test inWarpGroup with arbitrary index -/
@[test]
def testInWarpGroup : IO Unit := do
  let kernel := buildKernelM "test_in_warpgroup" .SM90 #[] do
    inWarpGroup 2 do
      comment "Warpgroup 2 code"
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "warpgroup::groupid() == 2") "Should check warpgroup 2"

/-! ## Macros Tests: Online Softmax -/

/-- Test allocSoftmaxState creates all required vectors -/
@[test]
def testAllocSoftmaxState : IO Unit := do
  let kernel := buildKernelM "test_softmax_state" .SM90 #[] do
    let _state ← allocSoftmaxState .Float32 64
    comment "Softmax state allocated"
    pure ()

  -- Should have: rowMax (negInfty = 2 ops), rowSum (zero = 2 ops), prevMax, scale
  assertTrue (kernel.body.size ≥ 6) "Should have vector allocations"

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "rv<float, 64") "Should have Float32 vectors"

/-- Test onlineSoftmax generates correct sequence -/
@[test]
def testOnlineSoftmax : IO Unit := do
  let kernel := buildKernelM "test_online_softmax" .SM90 #[] do
    let scores ← allocRT .Float32 64 64
    let output ← allocRT .Float32 64 128
    let state ← allocSoftmaxState .Float32 64
    onlineSoftmax scores output state
    pure ()

  let code := generateKernel kernel
  -- Check key operations in online softmax
  assertTrue (code.containsSubstr "copy(") "Should have copy for prevMax"
  assertTrue (code.containsSubstr "row_max(") "Should have row_max update"
  assertTrue (code.containsSubstr "sub(") "Should have sub for scale"
  assertTrue (code.containsSubstr "exp(") "Should have exp operations"
  assertTrue (code.containsSubstr "mul_row(") "Should have mul_row for rescaling"
  assertTrue (code.containsSubstr "sub_row(") "Should have sub_row for scores"
  assertTrue (code.containsSubstr "row_sum(") "Should have row_sum update"

/-- Test finalizeSoftmax -/
@[test]
def testFinalizeSoftmax : IO Unit := do
  let kernel := buildKernelM "test_finalize_softmax" .SM90 #[] do
    let output ← allocRT .Float32 64 64
    let state ← allocSoftmaxState .Float32 64
    finalizeSoftmax output state
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "div_row(") "Should have div_row for normalization"

/-- Test computeLSE -/
@[test]
def testComputeLSE : IO Unit := do
  let kernel := buildKernelM "test_compute_lse" .SM90 #[] do
    let state ← allocSoftmaxState .Float32 64
    let _lse ← computeLSE state
    comment "LSE computed"
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "log(") "Should have log for LSE"
  assertTrue (code.containsSubstr "add(") "Should have add for log(sum) + max"

/-! ## Macros Tests: Named Barriers -/

/-- Test signalBarrier -/
@[test]
def testSignalBarrier : IO Unit := do
  let kernel := buildKernelM "test_signal_barrier" .SM90 #[] do
    signalNamedBarrier queryReadyBarrier
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "kittens::arrive(kittens::barrier<") "Should have barrier arrive"

/-- Test waitBarrier -/
@[test]
def testWaitBarrier : IO Unit := do
  let kernel := buildKernelM "test_wait_barrier" .SM90 #[] do
    waitNamedBarrier queryReadyBarrier
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "kittens::arrive_and_wait(kittens::barrier<") "Should have barrier sync"

/-! ## Macros Tests: Attention Block Pattern -/

/-- Test attentionBlockIter generates complete attention computation -/
@[test]
def testAttentionBlockIter : IO Unit := do
  let kernel := buildKernelM "test_attn_block" .SM90 #[] do
    let q ← allocRT .BFloat16 64 64
    let k ← allocRT .BFloat16 64 64
    let v ← allocRT .BFloat16 64 64 .Col
    let output ← zeroRT .Float32 64 64
    let state ← allocSoftmaxState .Float32 64
    attentionBlockIter q k v output state false
    pure ()

  let code := generateKernel kernel
  -- Check all attention components
  assertTrue (code.containsSubstr "mma_ABt(") "Should have Q @ K^T"
  assertTrue (code.containsSubstr "exp(") "Should have softmax exp"
  assertTrue (code.containsSubstr "row_max(") "Should have row max"
  assertTrue (code.containsSubstr "mma_AB(") "Should have P @ V"

/-- Test attentionBlockIter with causal mask -/
@[test]
def testAttentionBlockIterCausal : IO Unit := do
  let kernel := buildKernelM "test_attn_block_causal" .SM90 #[] do
    let q ← allocRT .BFloat16 64 64
    let k ← allocRT .BFloat16 64 64
    let v ← allocRT .BFloat16 64 64 .Col
    let output ← zeroRT .Float32 64 64
    let state ← allocSoftmaxState .Float32 64
    attentionBlockIter q k v output state true (-1e10)
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "make_causal(") "Should have causal mask"

/-! ## Integration Tests -/

/-- Test complete attention kernel using new DSL -/
@[test]
def testCompleteDSLAttention : IO Unit := do
  let kernel := buildKernelM "dsl_attention" .SM90 #[] do
    comment "Allocate tiles"
    let q ← allocRT .BFloat16 64 64
    let k ← allocRT .BFloat16 64 64
    let _v ← allocRT .BFloat16 64 64 .Col

    comment "Compute S = Q @ K^T using notation"
    let s ← q ⬝ᵀ k

    comment "Apply softmax"
    let state ← allocSoftmaxState .Float32 64
    let output ← zeroRT .Float32 64 64
    onlineSoftmax s output state
    finalizeSoftmax output state

    comment "Compute LSE"
    let _lse ← computeLSE state

    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "dsl_attention") "Should have kernel name"
  assertTrue (code.containsSubstr "mma_ABt(") "Should have Q @ K^T"
  assertTrue (code.containsSubstr "row_max(") "Should have softmax"
  assertTrue (code.containsSubstr "div_row(") "Should have normalization"
  assertTrue (code.containsSubstr "log(") "Should have LSE computation"

/-- Test expression chaining -/
@[test]
def testExpressionChaining : IO Unit := do
  let kernel := buildKernelM "test_expr_chain" .SM90 #[] do
    let a ← allocRT .Float32 64 64
    let b ← allocRT .Float32 64 64

    comment "Chain: (a + b) * 0.5"
    let sum ← a + b
    let scaled ← sum * 0.5

    comment "Reduction"
    let maxVec ← reduceRowMax scaled

    comment "Broadcast back"
    let _normalized ← subBroadcastCol scaled maxVec

    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "add(") "Should have add"
  assertTrue (code.containsSubstr "mul(") "Should have mul"
  assertTrue (code.containsSubstr "row_max(") "Should have reduction"
  assertTrue (code.containsSubstr "sub_row(") "Should have broadcast"

/-- Test warp specialized attention pattern -/
@[test]
def testWarpSpecializedPattern : IO Unit := do
  let kernel := buildKernelM "test_warp_spec" .SM90 #[] do
    let _db ← allocDoubleBuffer .BFloat16 64 64

    asProducer do
      comment "Producer: TMA loads"
      signalNamedBarrier queryReadyBarrier

    asConsumer do
      comment "Consumer: MMA computation"
      waitNamedBarrier queryReadyBarrier

      let q ← allocRT .BFloat16 64 64
      let k ← allocRT .BFloat16 64 64
      let _s ← q ⬝ᵀ k
      pure ()

    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "warpgroup::groupid() == 0") "Should have producer check"
  assertTrue (code.containsSubstr "warpgroup::groupid() == 1") "Should have consumer check"
  assertTrue (code.containsSubstr "kittens::arrive") "Should have barrier ops"


/-! ## Fused RK stage-combination kernel -/

/-- The tableau-driven combine body emits one fused kernel: per-stage
    scalar-mul + accumulate for the solution and embedded-error weights,
    skipping zero-weight stages at codegen time. -/
@[test]
def testRKCombineCodegen : IO Unit := do
  let params : Array KParam :=
    (Array.range 10).map fun i =>
      { name := s!"p{i}", dtype := .Float32, isPointer := true }
  let kPtrs : Array (GPtr GpuFloat.Float32) :=
    (Array.range 7).map fun i => { id := ⟨i + 1⟩, name := s!"k{i + 1}" }
  let kernel := buildKernelM "rk_combine_test" .SM90 params do
    Tyr.GPU.Kernels.rkCombineBody
      #[35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0,
        -2187.0 / 6784.0, 11.0 / 84.0, 0.0]
      #[1951.0 / 21600.0, 0.0, 22642.0 / 50085.0, 451.0 / 720.0,
        -12231.0 / 42400.0, 649.0 / 6300.0, 1.0 / 60.0]
      ({ id := ⟨0⟩, name := "y0" } : GPtr GpuFloat.Float32) kPtrs
      { id := ⟨8⟩, name := "y1" } { id := ⟨9⟩, name := "err" }
  let code := generateKernel kernel
  assertTrue (code.containsSubstr "mul(") "Should emit scalar multiplies"
  assertTrue (code.containsSubstr "add(") "Should emit accumulation adds"
  assertTrue (code.containsSubstr "zero(") "Should zero the error accumulator"
  assertTrue (code.containsSubstr "0x1.7555555555555p-4") "Should bake b₁ = 35/384 exactly into the kernel"
  -- stage 2 has b = bHat = 0 and must be skipped at codegen time
  assertTrue (!(code.containsSubstr "v14, v2,")) "Zero-weight stage k2 must not be loaded"
  -- y0 + 6 live stages in, y1 + err out: 9 global accesses in one launch
  let globalAccesses := (code.splitOn "kittens::coord").length - 1
  assertTrue (globalAccesses == 9)
    s!"Expected 9 global tile accesses (7 in, 2 out), got {globalAccesses}"


/-- The stage-sum generator skips zero coefficients (Dopri5 row 7 has a
    zero at k2) and emits one fused accumulation per live stage. -/
@[test]
def testRKStageSumCodegen : IO Unit := do
  let params : Array KParam :=
    (Array.range 8).map fun i =>
      { name := s!"p{i}", dtype := .Float32, isPointer := true }
  let kPtrs : Array (GPtr GpuFloat.Float32) :=
    (Array.range 6).map fun i => { id := ⟨i + 1⟩, name := s!"k{i + 1}" }
  let kernel := buildKernelM "rk_stage7_test" .SM90 params do
    Tyr.GPU.Kernels.rkStageSumBody
      #[35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0,
        -2187.0 / 6784.0, 11.0 / 84.0]
      ({ id := ⟨0⟩, name := "y0" } : GPtr GpuFloat.Float32) kPtrs
      { id := ⟨7⟩, name := "out" }
  let code := generateKernel kernel
  assertTrue (code.containsSubstr "mul(") "Should emit scalar multiplies"
  assertTrue (!(code.containsSubstr "v14, v2,")) "Zero-coefficient stage k2 must not be loaded"
  -- y0 + 5 live stages in, 1 out: 7 global tile accesses
  let globalAccesses := (code.splitOn "kittens::coord").length - 1
  assertTrue (globalAccesses == 7)
    s!"Expected 7 global tile accesses (6 in, 1 out), got {globalAccesses}"


/-- The keyed-normal sampler replicates the CPU PRNG chain: exact 64-bit
    LCG constants, Box-Muller via the new scalar Sqrt/Cos ops, one global
    store per element. -/
@[test]
def testKeyedNormalCodegen : IO Unit := do
  let params : Array KParam := #[
    { name := "out", dtype := .Float32, isPointer := true },
    { name := "root", dtype := .Float32, isPointer := false },
    { name := "n", dtype := .Float32, isPointer := false }
  ]
  let kernel := buildKernelM "keyed_normal_test" .SM90 params do
    Tyr.GPU.Kernels.keyedNormalBody
      ({ id := ⟨0⟩, name := "out" } : GPtr GpuFloat.Float32)
      ({ id := ⟨1⟩, name := "root" } : KVal UInt64)
      ({ id := ⟨2⟩, name := "n" } : KVal UInt64)
  let code := generateKernel kernel
  assertTrue (code.containsSubstr "6364136223846793005ULL") "Exact 64-bit LCG multiplier"
  assertTrue (code.containsSubstr "11400714819323198485ULL") "Exact 64-bit golden-ratio constant"
  assertTrue (code.containsSubstr "::cosf(") "Box-Muller cosine via the new scalar Cos"
  assertTrue (code.containsSubstr "::sqrtf(") "Box-Muller radius via the new scalar Sqrt"
  assertTrue (code.containsSubstr "::logf(") "Box-Muller log"
  assertTrue (code.containsSubstr "threadIdx.x") "Per-thread parallel iteration"


/-- The VBT descent kernel walks both query schedules per element:
    keyed chains for interval/midpoint/point draws, branchless updates. -/
@[test]
def testVbtDescentCodegen : IO Unit := do
  let params : Array KParam := #[
    { name := "out", dtype := .Float32, isPointer := true },
    { name := "schedA", dtype := .Float32, isPointer := true },
    { name := "schedB", dtype := .Float32, isPointer := true },
    { name := "root", dtype := .Float32, isPointer := false },
    { name := "n", dtype := .Float32, isPointer := false },
    { name := "lenA", dtype := .Float32, isPointer := false },
    { name := "lenB", dtype := .Float32, isPointer := false }
  ]
  let kernel := buildKernelM "vbt_descent_test" .SM90 params do
    Tyr.GPU.Kernels.vbtIncrementBody
      ({ id := ⟨0⟩, name := "out" } : GPtr GpuFloat.Float32)
      ({ id := ⟨1⟩, name := "schedA" } : GPtr GpuFloat.Float32)
      ({ id := ⟨2⟩, name := "schedB" } : GPtr GpuFloat.Float32)
      ({ id := ⟨3⟩, name := "root" } : KVal UInt64)
      ({ id := ⟨4⟩, name := "n" } : KVal UInt64)
      ({ id := ⟨5⟩, name := "lenA" } : KVal UInt64)
      ({ id := ⟨6⟩, name := "lenB" } : KVal UInt64)
  let code := generateKernel kernel
  assertTrue (code.containsSubstr "2654435769ULL") "interval op tag baked"
  assertTrue (code.containsSubstr "608135816ULL") "midpoint op tag baked"
  assertTrue (code.containsSubstr "320440878ULL") "point op tag baked"
  assertTrue (code.containsSubstr "::cosf(") "Box-Muller per node"
  assertTrue (code.containsSubstr "? ") "branchless selects emitted"


/-- The fused Euler–Maruyama step draws the VBT increment inline and
    applies `y + fh + g·dW` with one load per operand and one store. -/
@[test]
def testEmStepVbtCodegen : IO Unit := do
  let params : Array KParam := #[
    { name := "out", dtype := .Float32, isPointer := true },
    { name := "y0", dtype := .Float32, isPointer := true },
    { name := "fh", dtype := .Float32, isPointer := true },
    { name := "g", dtype := .Float32, isPointer := true },
    { name := "schedA", dtype := .Float32, isPointer := true },
    { name := "schedB", dtype := .Float32, isPointer := true },
    { name := "root", dtype := .Float32, isPointer := false },
    { name := "n", dtype := .Float32, isPointer := false },
    { name := "lenA", dtype := .Float32, isPointer := false },
    { name := "lenB", dtype := .Float32, isPointer := false }
  ]
  let kernel := buildKernelM "em_step_vbt_test" .SM90 params do
    Tyr.GPU.Kernels.emStepVbtBody
      ({ id := ⟨0⟩, name := "out" } : GPtr GpuFloat.Float32)
      ({ id := ⟨1⟩, name := "y0" } : GPtr GpuFloat.Float32)
      ({ id := ⟨2⟩, name := "fh" } : GPtr GpuFloat.Float32)
      ({ id := ⟨3⟩, name := "g" } : GPtr GpuFloat.Float32)
      ({ id := ⟨4⟩, name := "schedA" } : GPtr GpuFloat.Float32)
      ({ id := ⟨5⟩, name := "schedB" } : GPtr GpuFloat.Float32)
      ({ id := ⟨6⟩, name := "root" } : KVal UInt64)
      ({ id := ⟨7⟩, name := "n" } : KVal UInt64)
      ({ id := ⟨8⟩, name := "lenA" } : KVal UInt64)
      ({ id := ⟨9⟩, name := "lenB" } : KVal UInt64)
  let code := generateKernel kernel
  assertTrue (code.containsSubstr "2654435769ULL") "interval op tag baked"
  assertTrue (code.containsSubstr "608135816ULL") "midpoint op tag baked"
  assertTrue (code.containsSubstr "320440878ULL") "point op tag baked"
  assertTrue (code.containsSubstr "v1[") "state load emitted"
  assertTrue (code.containsSubstr "v2[") "scaled-drift load emitted"
  assertTrue (code.containsSubstr "v3[") "diffusion load emitted"
  assertTrue (code.containsSubstr "v0[") "fused-update store emitted"

end Tests.GPUDSL
