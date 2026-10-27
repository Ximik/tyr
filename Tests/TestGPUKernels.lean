/-
  Tests/TestGPUKernels.lean

  Tests for the native Lean4 GPU kernel DSL.
  Tests type safety, code generation, and architecture specialization.
-/
import Lean
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.Primitives
import Tyr.GPU.Codegen.Loop
import Tyr.GPU.Codegen.Macros
import Tyr.GPU.Codegen.EmitNew
import Tyr.GPU.Codegen.Attribute
import Tyr.GPU.Kernels.MhaH100Decode
import Tyr.GPU.Ops.AttentionProblem
import LeanTest
import Std.Sync.Mutex

namespace Tests.GPUKernels

open Tyr.GPU
open Tyr.GPU.Codegen
open LeanTest

private def mkTestKernel
    (name : String)
    (arch : GpuArch)
    (body : Array KStmt)
    (params : Array KParam := #[])
    (sharedMemBytes : Nat := 0) : Kernel := {
  name := name
  arch := arch
  family := arch.toFamily
  params := params
  body := body
  sharedMemBytes := sharedMemBytes
}

private def assertContainsAll (code : String) (checks : Array (String × String)) : IO Unit := do
  for (needle, msg) in checks do
    assertTrue (code.containsSubstr needle) msg

private def assertNotContains (code : String) (needle msg : String) : IO Unit := do
  assertTrue (!(code.containsSubstr needle)) msg

private partial def stmtHasRaw : KStmt → Bool
  | .raw _ => true
  | .forLoop _ _ _ body => body.any stmtHasRaw
  | .forLoopVal _ _ _ body => body.any stmtHasRaw
  | .forLoopRev _ _ _ body => body.any stmtHasRaw
  | .forLoopValRev _ _ _ body => body.any stmtHasRaw
  | .forLoopStrideVal _ _ _ _ body => body.any stmtHasRaw
  | .ifStmt _ thenBody elseBody => thenBody.any stmtHasRaw || elseBody.any stmtHasRaw
  | .ifWarpGroup _ body => body.any stmtHasRaw
  | _ => false

private def runCoreWithEnv (env : Lean.Environment) (x : Lean.CoreM α) : IO α := do
  let ctx : Lean.Core.Context := {
    fileName := "<gpu-kernel-test>"
    fileMap := default
  }
  let st : Lean.Core.State := { env := env }
  x.toIO' ctx st

private unsafe def evalKernelConstExprForTest (constName : Lean.Name) : Lean.CoreM Kernel := do
  withTheReader Lean.Core.Context (fun ctx => { ctx with maxHeartbeats := 0 }) do
    Lean.Meta.MetaM.run' do
      let info ← Lean.getConstInfo constName
      let value ← match info with
        | .defnInfo info => pure info.value
        | .thmInfo info => pure info.value
        | _ => throwError "Kernel companion '{constName}' is not reducible."
      Lean.Meta.evalExpr
        Kernel
        (Lean.mkConst ``Tyr.GPU.Codegen.Kernel)
        value

-- Keep imported compacted regions alive for returned kernel expressions, and
-- reuse the immutable base across decode variants instead of importing again.
private initialize kernelTestEnvironment : Std.Mutex (Option Lean.Environment) ←
  Std.Mutex.new none

private unsafe def getKernelTestEnvironment : IO Lean.Environment :=
  kernelTestEnvironment.atomically do
    if let some env := ← get then
      return env
    Lean.initSearchPath (← Lean.findSysroot)
    Lean.enableInitializersExecution
    let env ← Lean.importModules
      #[{ module := `LeanTest }, { module := `Tyr.GPU.Kernels.MhaH100Decode }]
      {} (loadExts := true)
    set (some env)
    return env

private unsafe def evalKernelForTest (kernelConst : Lean.Name) : IO Kernel := do
  let env ← getKernelTestEnvironment
  runCoreWithEnv env <|
    evalKernelConstExprForTest kernelConst

/-! ## Basic Tile Allocation Tests -/

/-- Test basic register tile allocation -/
@[test]
def testAllocRT : IO Unit := do
  let kernel := buildKernelM "test_alloc_rt" .SM90 #[] do
    let _a : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let _b : Tyr.GPU.Codegen.RT GpuFloat.Float32 32 32 .Col ← allocRT .Float32 32 32 .Col
    pure ()

  assertEqual kernel.body.size 2 "Should have 2 declarations"
  match kernel.body[0]! with
  | .declRT _ dtype rows cols layout =>
    assertEqual dtype GpuFloat.BFloat16 "First tile should be BFloat16"
    assertEqual rows 64 "First tile should have 64 rows"
    assertEqual cols 64 "First tile should have 64 cols"
    assertEqual layout TileLayout.Row "First tile should be row-major"
  | _ => fail "First statement should be declRT"

/-- Test shared tile allocation tracks memory -/
@[test]
def testAllocSTTracksMemory : IO Unit := do
  let kernel := buildKernelM "test_shared_mem" .SM90 #[] do
    let _s1 : Tyr.GPU.Codegen.ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64  -- 64*64*2 = 8192
    let _s2 : Tyr.GPU.Codegen.ST GpuFloat.Float32 32 32 ← allocST .Float32 32 32    -- 32*32*4 = 4096
    pure ()

  assertEqual kernel.sharedMemBytes (8192 + 4096) "Should track shared memory"

/-- Test vector allocation -/
@[test]
def testAllocRV : IO Unit := do
  let kernel := buildKernelM "test_alloc_rv" .SM90 #[] do
    let _v : Tyr.GPU.Codegen.RV GpuFloat.Float32 64 ← allocRV .Float32 64
    pure ()

  match kernel.body[0]! with
  | .declRV _ dtype len =>
    assertEqual dtype GpuFloat.Float32 "Vector should be Float32"
    assertEqual len 64 "Vector should have length 64"
  | _ => fail "Should be declRV"

/-- Test zero-initialized allocation -/
@[test]
def testZeroRT : IO Unit := do
  let kernel := buildKernelM "test_zero_rt" .SM90 #[] do
    let _z : Tyr.GPU.Codegen.RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
    pure ()

  assertEqual kernel.body.size 2 "Should have decl + zero"
  match kernel.body[1]! with
  | .unary op _ _ => assertEqual op UnaryOp.Zero "Should be zero operation"
  | _ => fail "Second statement should be unary zero"

/-! ## MMA Type Safety Tests -/

/-- Test MMA generates correct IR -/
@[test]
def testMMAGeneration : IO Unit := do
  let kernel := buildKernelM "test_mma" .SM90 #[] do
    let a : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let b : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
    let c : Tyr.GPU.Codegen.RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
    mma c a b c
    pure ()

  -- Find the MMA statement
  let mmaStmt := kernel.body.find? fun s =>
    match s with
    | .mma _ _ _ _ _ => true
    | _ => false

  assertTrue mmaStmt.isSome "Should contain MMA statement"
  match mmaStmt with
  | some (.mma trans _ _ _ _) =>
    assertEqual trans MMATranspose.AB "Should be AB transpose mode"
  | _ => fail "Should be MMA with AB mode"

/-- Test MMA with B transposed -/
@[test]
def testMMAT : IO Unit := do
  let kernel := buildKernelM "test_mma_t" .SM90 #[] do
    let a : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let b : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64  -- Row-major for transpose
    let c : Tyr.GPU.Codegen.RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
    mmaT c a b c
    pure ()

  let mmaStmt := kernel.body.find? fun s =>
    match s with
    | .mma _ _ _ _ _ => true
    | _ => false

  match mmaStmt with
  | some (.mma trans _ _ _ _) =>
    assertEqual trans MMATranspose.ABt "Should be ABt transpose mode"
  | _ => fail "Should be MMA with ABt mode"

/-! ## Loop Generation Tests -/

/-- Test for loop generates correct structure -/
@[test]
def testForLoop : IO Unit := do
  let kernel := buildKernelM "test_loop" .SM90 #[] do
    forLoop 0 16 do
      sync
    pure ()

  assertEqual kernel.body.size 1 "Should have one loop"
  match kernel.body[0]! with
  | .forLoop _ lo hi body =>
    assertEqual lo 0 "Loop should start at 0"
    assertEqual hi 16 "Loop should end at 16"
    assertEqual body.size 1 "Loop body should have 1 statement"
  | _ => fail "Should be forLoop"

/-- Test nested loops -/
@[test]
def testNestedLoops : IO Unit := do
  let kernel := buildKernelM "test_nested" .SM90 #[] do
    forLoop 0 4 do
      forLoop 0 8 do
        sync
    pure ()

  match kernel.body[0]! with
  | .forLoop _ _ _ outerBody =>
    match outerBody[0]! with
    | .forLoop _ lo hi innerBody =>
      assertEqual lo 0 "Inner loop start"
      assertEqual hi 8 "Inner loop end"
      assertEqual innerBody.size 1 "Inner body size"
    | _ => fail "Should have inner forLoop"
  | _ => fail "Should have outer forLoop"

@[test]
def testReverseLoopCodegen : IO Unit := do
  let kernel := mkTestKernel
    "test_reverse_loop"
    .SM90
    #[.forLoopRev { idx := 0 } 0 4 #[.comment "reverse body"]]

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "for (int v0 = 4; v0-- > 0; )")
    "Reverse loops should lower to descending iteration order"

@[test]
def testReverseLoopValCodegen : IO Unit := do
  let kernel := mkTestKernel
    "test_reverse_loop_val"
    .SM90
    #[.forLoopValRev { idx := 0 } 2 { idx := 1 } #[.comment "reverse body"]]

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "for (int v0 = v1; v0-- > 2; )")
    "Reverse value-bounded loops should lower to descending iteration order"

/-! ## Architecture Metadata Tests -/

@[test]
def testKernelDefaultsFamilyFromArchFloor : IO Unit := do
  let hopperKernel := buildKernelM "hopper_floor" .SM90 #[] do
    pure ()
  let blackwellKernel := buildKernelM "blackwell_floor" .SM100 #[] do
    pure ()

  assertEqual hopperKernel.arch .SM90 "Kernel floor should stay SM90"
  assertEqual hopperKernel.family .Hopper "SM90 kernels should default to Hopper family"
  assertEqual blackwellKernel.arch .SM100 "Kernel floor should stay SM100"
  assertEqual blackwellKernel.family .Blackwell "SM100 kernels should default to Blackwell family"

@[test]
def testFamilyOverrideChangesGuardWithoutChangingFloor : IO Unit := do
  let kernel := buildKernelM "gb10_guard" .SM90 #[] do
    setFamily .Blackwell
    let _tile : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    pure ()

  assertEqual kernel.arch .SM90 "GB10-compatible kernels should keep an SM90 capability floor"
  assertEqual kernel.family .Blackwell "GB10-compatible kernels should emit under the Blackwell family guard"

  let code := generateKernel kernel
  assertContainsAll code #[
    ("#if defined(KITTENS_BLACKWELL)",
      "Family override should drive the emitted availability guard")
  ]
  assertNotContains code "#if defined(KITTENS_HOPPER)"
    "Family override should replace the default Hopper guard"

@[test]
def testSetArchResetsFamilyToArchDefault : IO Unit := do
  let kernel := buildKernelM "family_reset" .SM90 #[] do
    setFamily .Blackwell
    setArch .SM80
    pure ()

  assertEqual kernel.arch .SM80 "setArch should update the capability floor"
  assertEqual kernel.family .Ampere "setArch should restore the default family for the new floor"

@[test]
def testCppLauncherUsesFamilyGuardAndFloorMessage : IO Unit := do
  let kernel : Kernel := {
    name := "gb10_floor_launcher"
    arch := .SM90
    family := .Blackwell
    params := #[{ name := "x", dtype := .BFloat16, isPointer := true }]
    body := #[]
  }
  let cpp := generateCppLauncherCode kernel

  assertContainsAll cpp #[
    ("#if defined(KITTENS_BLACKWELL)",
      "Launchers should be guarded by the build family, not the capability floor"),
    ("requires Blackwell family, SM90 floor",
      "Unavailable-launcher diagnostics should report both family and floor")
  ]

/-! ## Code Generation Tests -/

/-- Test C++ code generation for declarations -/
@[test]
def testCodeGenDeclarations : IO Unit := do
  let kernel := buildKernelM "test_codegen_decl" .SM90 #[] do
    let _rt : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let _st : Tyr.GPU.Codegen.ST GpuFloat.Float32 32 64 .Col ← allocST .Float32 32 64 .Col
    let _rv : Tyr.GPU.Codegen.RV GpuFloat.Float32 64 ← allocRV .Float32 64
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "rt<bf16, 64, 64, row_l>") "Should have RT declaration"
  assertTrue (code.containsSubstr "al.allocate<st<float, 32, 64>>") "Should have ST declaration"
  assertTrue (code.containsSubstr "rv<float, 64") "Should have RV declaration"

@[test]
def testUnsupportedSliceRowsCodegenFailsLoudly : IO Unit := do
  let kernel := mkTestKernel
    "test_bad_slice_rows"
    .SM90
    #[
      .declRT { idx := 0 } .Float32 64 64 .Row,
      .declTT { idx := 1 } .Float32 64 64,
      .sliceRows { idx := 0 } { idx := 1 } 0 64
    ]

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "static_assert(false, \"unsupported sliceRows between non-matching tile kinds\")")
    "Unsupported sliceRows lowering should fail loudly instead of becoming a comment"

@[test]
def testMixedSliceRowsFromSharedToRegisterCodegen : IO Unit := do
  let kernel := mkTestKernel
    "test_slice_rows_mixed"
    .SM90
    #[
      .declRT { idx := 0 } .Float32 64 64 .Row,
      .declST { idx := 1 } .Float32 64 64 .Row,
      .sliceRows { idx := 0 } { idx := 1 } 0 64
    ]

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "auto _tk_src_sub = v1.template subtile<64, 64>(make_int2(0, 0));")
    "Shared-to-register sliceRows should create a shared subtile view"
  assertTrue (code.containsSubstr "kittens::warp::copy(v0, _tk_src_sub);")
    "Shared-to-register sliceRows should copy from the shared subtile into the register tile"

@[test]
def testMixedSliceRowsFromRegisterToSharedCodegen : IO Unit := do
  let kernel := mkTestKernel
    "test_slice_rows_into_shared_supported"
    .SM90
    #[
      .declST { idx := 0 } .Float32 64 64 .Row,
      .declRT { idx := 1 } .Float32 64 64 .Row,
      .sliceRows { idx := 0 } { idx := 1 } 0 64
    ]

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "auto _tk_dst_sub = v0.template subtile<64, 64>(make_int2(0, 0));")
    "Register-to-shared sliceRows should create a shared destination subtile"
  assertTrue (code.containsSubstr "kittens::warp::copy(_tk_dst_sub, v1);")
    "Register-to-shared sliceRows should copy the register tile into the shared subtile"

@[test]
def testMixedSliceColsFromSharedToRegisterCodegen : IO Unit := do
  let kernel := mkTestKernel
    "test_slice_cols_mixed"
    .SM90
    #[
      .declRT { idx := 0 } .Float32 64 64 .Row,
      .declST { idx := 1 } .Float32 64 64 .Row,
      .sliceCols { idx := 0 } { idx := 1 } 0 64
    ]

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "auto _tk_src_sub = v1.template subtile<64, 64>(make_int2(0, 0));")
    "Shared-to-register sliceCols should create a shared subtile view"
  assertTrue (code.containsSubstr "kittens::warp::copy(v0, _tk_src_sub);")
    "Shared-to-register sliceCols should copy from the shared subtile into the register tile"

@[test]
def testMixedSliceColsFromRegisterToSharedCodegen : IO Unit := do
  let kernel := mkTestKernel
    "test_slice_cols_into_shared_supported"
    .SM90
    #[
      .declST { idx := 0 } .Float32 64 64 .Row,
      .declRT { idx := 1 } .Float32 64 64 .Row,
      .sliceCols { idx := 0 } { idx := 1 } 0 64
    ]

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "auto _tk_dst_sub = v0.template subtile<64, 64>(make_int2(0, 0));")
    "Register-to-shared sliceCols should create a shared destination subtile"
  assertTrue (code.containsSubstr "kittens::warp::copy(_tk_dst_sub, v1);")
    "Register-to-shared sliceCols should copy the register tile into the shared subtile"

@[test]
def testMixedConcatColsIntoSharedCodegen : IO Unit := do
  let kernel := mkTestKernel
    "test_concat_cols_into_shared"
    .SM90
    #[
      .declST { idx := 0 } .Float32 64 128 .Row,
      .declRT { idx := 1 } .Float32 64 64 .Row,
      .declST { idx := 2 } .Float32 64 64 .Row,
      .concatCols { idx := 0 } { idx := 1 } { idx := 2 }
    ]

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "auto _tk_dst_left = v0.template subtile<64, 64>(make_int2(0, 0));")
    "Shared concat should create a left subtile"
  assertTrue (code.containsSubstr "auto _tk_dst_right = v0.template subtile<64, 64>(make_int2(0, 64));")
    "Shared concat should create a right subtile"
  assertTrue (code.containsSubstr "kittens::warp::copy(_tk_dst_left, v1);")
    "Shared concat should accept register-tile left inputs"
  assertTrue (code.containsSubstr "kittens::warp::copy(_tk_dst_right, v2);")
    "Shared concat should accept shared-tile right inputs"

@[test]
def testMixedConcatColsIntoRegisterCodegen : IO Unit := do
  let kernel := mkTestKernel
    "test_concat_cols_into_register_supported"
    .SM90
    #[
      .declRT { idx := 0 } .Float32 64 128 .Row,
      .declRT { idx := 1 } .Float32 64 64 .Row,
      .declST { idx := 2 } .Float32 64 64 .Row,
      .concatCols { idx := 0 } { idx := 1 } { idx := 2 }
    ]

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "rt<float, 64, 64, row_l> _tk_right_rt;")
    "Mixed concat into a register tile should materialize a temporary register tile for the shared input"
  assertTrue (code.containsSubstr "auto _tk_right_sub = v2.template subtile<64, 64>(make_int2(0, 0));")
    "Mixed concat into a register tile should create a shared subtile view for the shared input"
  assertTrue (code.containsSubstr "kittens::warp::copy(_tk_right_rt, _tk_right_sub);")
    "Mixed concat into a register tile should copy the shared side into the temporary register tile"

@[test]
def testEqMaskCodegenEmitsHelpers : IO Unit := do
  let kernel := mkTestKernel
    "test_eq_mask"
    .SM90
    #[
      .declRT { idx := 0 } .Float32 64 64 .Row,
      .declRT { idx := 1 } .Float32 64 64 .Row,
      .declRT { idx := 2 } .Float32 64 64 .Row,
      .eqMask { idx := 0 } { idx := 1 } { idx := 2 }
    ]

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "tk_eq_mask(v0, v1, v2);")
    "eqMask should lower to the dedicated helper call"
  assertTrue (code.containsSubstr "template<ducks::rt::all T>")
    "eqMask helper templates should be emitted when the kernel uses eqMask"

@[test]
def testUnsupportedSliceColsCodegenFailsLoudly : IO Unit := do
  let kernel := mkTestKernel
    "test_bad_slice_cols"
    .SM90
    #[
      .declRT { idx := 0 } .Float32 64 64 .Row,
      .declTT { idx := 1 } .Float32 64 64,
      .sliceCols { idx := 0 } { idx := 1 } 0 64
    ]

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "static_assert(false, \"unsupported sliceCols between non-matching tile kinds\")")
    "Unsupported sliceCols lowering should fail loudly instead of becoming a comment"

@[test]
def testUnsupportedConcatColsCodegenFailsLoudly : IO Unit := do
  let kernel := mkTestKernel
    "test_bad_concat_cols"
    .SM90
    #[
      .declTT { idx := 0 } .Float32 64 128,
      .declRT { idx := 1 } .Float32 64 64 .Row,
      .declRT { idx := 2 } .Float32 64 64 .Row,
      .concatCols { idx := 0 } { idx := 1 } { idx := 2 }
    ]

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "static_assert(false, \"unsupported concatCols between non-matching tile kinds\")")
    "Unsupported concatCols lowering should fail loudly instead of becoming a comment"

/-- Test C++ code generation for operations -/
@[test]
def testCodeGenOperations : IO Unit := do
  let kernel := buildKernelM "test_codegen_ops" .SM90 #[] do
    let a : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let b : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    add a a b
    exp a a
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "add(") "Should have add operation"
  assertTrue (code.containsSubstr "exp(") "Should have exp operation"

/-- Test C++ code generation for MMA -/
@[test]
def testCodeGenMMA : IO Unit := do
  let kernel := buildKernelM "test_codegen_mma" .SM90 #[] do
    let a : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let b : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
    let c : Tyr.GPU.Codegen.RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
    mma c a b c
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "mma_AB(") "Should have mma_AB call"

/-- Test architecture guard in generated code -/
@[test]
def testCodeGenArchGuard : IO Unit := do
  let kernelSM90 := buildKernelM "test_sm90" .SM90 #[] (pure ())
  let kernelSM80 := buildKernelM "test_sm80" .SM80 #[] (pure ())
  let kernelSM100 := buildKernelM "test_sm100" .SM100 #[] (pure ())

  let code90 := generateKernel kernelSM90
  let code80 := generateKernel kernelSM80
  let code100 := generateKernel kernelSM100

  assertTrue (code90.containsSubstr "KITTENS_HOPPER") "SM90 should use HOPPER guard"
  assertTrue (code80.containsSubstr "KITTENS_AMPERE") "SM80 should use AMPERE guard"
  assertTrue (code100.containsSubstr "KITTENS_BLACKWELL") "SM100 should use BLACKWELL guard"

/-! ## Reduction and Broadcast Tests -/

/-- Test row-wise reductions -/
@[test]
def testRowReduction : IO Unit := do
  let kernel := buildKernelM "test_row_reduce" .SM90 #[] do
    let t : Tyr.GPU.Codegen.RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    let v : Tyr.GPU.Codegen.RV GpuFloat.Float32 64 ← allocRV .Float32 64
    rowSum v t
    rowMax v t
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "row_sum(") "Should have row_sum"
  assertTrue (code.containsSubstr "row_max(") "Should have row_max"

/-- Test column broadcast operations -/
@[test]
def testColBroadcast : IO Unit := do
  let kernel := buildKernelM "test_col_broadcast" .SM90 #[] do
    let t : Tyr.GPU.Codegen.RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    let v : Tyr.GPU.Codegen.RV GpuFloat.Float32 64 ← allocRV .Float32 64
    subCol t t v
    divCol t t v
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "sub_row(") "Should have sub_row"
  assertTrue (code.containsSubstr "div_row(") "Should have div_row"

/-! ## Masking Tests -/

/-- Test causal mask generation -/
@[test]
def testCausalMask : IO Unit := do
  let kernel := buildKernelM "test_causal" .SM90 #[] do
    let s : Tyr.GPU.Codegen.RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    makeCausal s s (some (-1e10))
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "make_causal(") "Should have make_causal"
  assertTrue (code.containsSubstr "-10000000000") "Should have fill value"

/-- Test triangular masks -/
@[test]
def testTriangularMasks : IO Unit := do
  let kernel := buildKernelM "test_tril_triu" .SM90 #[] do
    let t : Tyr.GPU.Codegen.RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    tril t t 0 (some 0.0)
    triu t t 1 none
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "tril(") "Should have tril"
  assertTrue (code.containsSubstr "triu(") "Should have triu"

/-! ## Synchronization Tests -/

/-- Test sync and arrive operations -/
@[test]
def testSynchronization : IO Unit := do
  let kernel := buildKernelM "test_sync" .SM90 #[] do
    sync 0
    arrive 1
    mmaCommitGroup
    mmaAsyncWait 2
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "sync(0)") "Should have sync"
  assertTrue (code.containsSubstr "arrive(1)") "Should have arrive"
  assertTrue (code.containsSubstr "mma_commit_group()") "Should have mma_commit_group"
  assertTrue (code.containsSubstr "mma_async_wait<2>()") "Should have mma_async_wait"

/-! ## Memory Operations Tests -/

/-- Test load/store operations -/
@[test]
def testLoadStore : IO Unit := do
  let kernel := buildKernelM "test_load_store" .SM90 #[] do
    let r : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let s : Tyr.GPU.Codegen.ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
    load r s
    store s r
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "load(") "Should have load"
  assertTrue (code.containsSubstr "store(") "Should have store"

/-- Test atomic store-add for gradients -/
@[test]
def testStoreAdd : IO Unit := do
  let kernel := buildKernelM "test_store_add" .SM90 #[] do
    let r : Tyr.GPU.Codegen.RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    let s : Tyr.GPU.Codegen.ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64
    storeAdd s r
    storeAddAsync s r
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "store_add(") "Should have store_add"
  assertTrue (code.containsSubstr "tma::store_add_async(") "Should have tma store_add_async"

/-! ## Type Conversion Tests -/

/-- Test type conversion -/
@[test]
def testConvert : IO Unit := do
  let kernel := buildKernelM "test_convert" .SM90 #[] do
    let bf : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let fp : Tyr.GPU.Codegen.RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    convert fp bf
    convert bf fp
    pure ()

  let code := generateKernel kernel
  -- convert uses copy in ThunderKittens
  assertTrue (code.containsSubstr "copy(") "Should have copy for conversion"

/-- Test layout swap -/
@[test]
def testSwapLayout : IO Unit := do
  let kernel := buildKernelM "test_swap_layout" .SM90 #[] do
    let row : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let col : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
    swapLayout row col
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "swap_layout(") "Should have swap_layout"

/-! ## Parameter Handling Tests -/

/-- Test kernel parameter generation -/
@[test]
def testKernelParams : IO Unit := do
  let kernel := buildKernelM "test_params" .SM90 #[
    { name := "x_ptr", dtype := .BFloat16, isPointer := true },
    { name := "y_ptr", dtype := .Float32, isPointer := true },
    { name := "size", dtype := .Float32, isPointer := false }
  ] (pure ())

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "gl<bf16, -1, -1, -1, -1> v0") "Should have bf16 global descriptor param"
  assertTrue (code.containsSubstr "gl<float, -1, -1, -1, -1> v1") "Should have float global descriptor param"
  assertTrue (code.containsSubstr "uint64_t v2") "Should have scalar param"

/-- Test the shared SIMT helpers used by decode-style kernels. -/
@[test]
def testDecodeSimtHelperCodegen : IO Unit := do
  let kernel := buildKernelM "decode_helper_test" .SM90 #[] do
    let dim ← constUInt64Val 64 "head_dim"
    let _scale ← runtimeDefaultScoreScaleLog2e dim
    for _ in (← parallelThreadRange dim) do
      blockSync

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "threadIdx.x")
    "parallelThreadRange should use the current thread index"
  assertTrue (code.containsSubstr "blockDim.x")
    "parallelThreadRange should stride by the CTA size"
  assertTrue (code.containsSubstr "::rsqrtf")
    "runtimeDefaultScoreScaleLog2e should lower through scalar rsqrt"
  assertTrue (code.containsSubstr "0x1.71547652b953ap0f")
    "runtimeDefaultScoreScaleLog2e should include log2(e)"

/-- Decode attention should be authored through the DSL, not raw CUDA snippets.

    The TK-style tile-based decode kernel (V1) lowers QK^T and PV through
    WGMMA, K/V loads through TMA, and uses online softmax in `log2` form. -/
@[test]
unsafe def testDecodeKernelDslCodegen : IO Unit := do
  let kernel ← evalKernelForTest ``Tyr.GPU.Kernels.tkMhaH100DecodeFwd.kernel
  assertTrue (!kernel.body.any stmtHasRaw)
    "Decode kernel IR should not contain raw backend escape statements"

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "extern __shared__ int __shm[]")
    "Dynamic shared-memory views should emit a shared-memory arena"
  assertTrue (code.containsSubstr "mm_ABt(")
    "QK^T should lower through warpgroup MM overwrite (mm_ABt)"
  assertTrue (code.containsSubstr "mma_AB(")
    "PV should lower through warpgroup MMA (mma_AB)"
  assertTrue (code.containsSubstr "tma::load_async")
    "K/V loads should lower through TMA async loads"
  assertTrue (code.containsSubstr "wait(")
    "Pipeline semaphores should lower through kittens::wait"
  assertTrue (code.containsSubstr "exp2")
    "Online softmax should use base-2 exponent (TK convention)"
  assertTrue (code.containsSubstr "right_fill(")
    "Tail-block scores should be masked via TK right_fill so non-multiple-of-64 kv_seq works"

/-- The head_dim=64 decode variant emits the same TK-style structure. -/
@[test]
unsafe def testDecodeKernelDslCodegen64 : IO Unit := do
  let kernel ← evalKernelForTest ``Tyr.GPU.Kernels.tkMhaH100DecodeFwd64.kernel
  assertTrue (!kernel.body.any stmtHasRaw)
    "Decode-64 kernel IR should not contain raw backend escape statements"
  let code := generateKernel kernel
  assertTrue (code.containsSubstr "mm_ABt(")
    "QK^T should still lower through warpgroup MM overwrite in the 64-dim variant"
  assertTrue (code.containsSubstr "mma_AB(")
    "PV should still lower through warpgroup MMA in the 64-dim variant"
  assertTrue (code.containsSubstr "tma::load_async")
    "K/V loads should still lower through TMA async loads in the 64-dim variant"
  assertTrue (code.containsSubstr "right_fill(")
    "Tail-block masking should also be present in the 64-dim variant"

/-- Eligibility predicate routes Llama-3-8B decode shapes to the TK kernel. -/
@[test]
def testDecodeEligibilityLlama3 : IO Unit := do
  -- Llama-3-8B: 32 q heads, 8 kv heads (GQA ratio 4), head_dim=128, BF16, decode.
  let llama3 : Tyr.GPU.Ops.AttentionProblem := {
    batch := 1, numQHeads := 32, numKVHeads := 8, qSeq := 1, kvSeq := 2048,
    headDim := 128, dtype := .BFloat16, device := .CUDA 0, arch := .SM90,
    mode := .decode, enableGqa := true,
  }
  assertTrue (Tyr.GPU.Ops.AttentionProblem.currentSpecialization llama3
              == Tyr.GPU.Ops.AttentionSpecialization.tkMhaH100Decode)
    "Llama-3-8B decode shape should select tkMhaH100Decode"

  -- head_dim=64 should also select decode (Qwen3-4B style shape via the 64-dim variant).
  let qwen4b := { llama3 with numQHeads := 32, numKVHeads := 8, headDim := 64 }
  assertTrue (Tyr.GPU.Ops.AttentionProblem.currentSpecialization qwen4b
              == Tyr.GPU.Ops.AttentionSpecialization.tkMhaH100Decode)
    "head_dim=64 with GQA should also select tkMhaH100Decode (64-dim variant)"

  -- head_dim=256 selects decode now (Qwen 3.5/3.6 family, Gemma-2 27B).
  let qwen36 := { llama3 with numQHeads := 16, numKVHeads := 2, headDim := 256 }
  assertTrue (Tyr.GPU.Ops.AttentionProblem.currentSpecialization qwen36
              == Tyr.GPU.Ops.AttentionSpecialization.tkMhaH100Decode)
    "head_dim=256 with GQA should select tkMhaH100Decode (256-dim variant)"

  -- Truly unsupported head dims still fall back to portable.
  let llama3D192 := { llama3 with headDim := 192 }
  assertTrue (Tyr.GPU.Ops.AttentionProblem.currentSpecialization llama3D192
              == Tyr.GPU.Ops.AttentionSpecialization.portable)
    "head_dim=192 should fall back to portable (V1 supports 64/128/256)"

  -- Non-multiple-of-64 kv_seq should still select decode; the kernel handles
  -- the tail with a runtime right_fill mask.
  let llama3Tail := { llama3 with kvSeq := 2049 }
  assertTrue (Tyr.GPU.Ops.AttentionProblem.currentSpecialization llama3Tail
              == Tyr.GPU.Ops.AttentionSpecialization.tkMhaH100Decode)
    "kv_seq=2049 should still select tkMhaH100Decode (tail mask handles the partial block)"

  -- qSeq>1 is not single-token decode and must fall through.
  let llama3Prefill := { llama3 with qSeq := 16, mode := .densePrefill }
  assertTrue (Tyr.GPU.Ops.AttentionProblem.currentSpecialization llama3Prefill
              == Tyr.GPU.Ops.AttentionSpecialization.portable)
    "qSeq>1 should fall back to portable (decode requires qSeq=1)"

/-! ## Integration Tests -/

/-- Test complete FlashAttention-like kernel structure -/
@[test]
def testFlashAttnStructure : IO Unit := do
  let kernel := buildKernelM "flash_attn_test" .SM90 #[
    { name := "Q", dtype := .BFloat16, isPointer := true },
    { name := "K", dtype := .BFloat16, isPointer := true },
    { name := "V", dtype := .BFloat16, isPointer := true },
    { name := "O", dtype := .BFloat16, isPointer := true }
  ] do
    let q : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let k : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let _v : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
    let s : Tyr.GPU.Codegen.RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    let o : Tyr.GPU.Codegen.RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

    let rowMax : Tyr.GPU.Codegen.RV GpuFloat.Float32 64 ← negInftyRV .Float32 64
    let rowSum : Tyr.GPU.Codegen.RV GpuFloat.Float32 64 ← allocRV .Float32 64

    let qShared : Tyr.GPU.Codegen.ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
    let kShared : Tyr.GPU.Codegen.ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

    load q qShared

    forLoop 0 4 do
      load k kShared
      mmaT s q k s
      makeCausal s s (some (-1e10))
      rowMaxAccum rowMax s rowMax
      subCol s s rowMax
      exp s s
      rowSumAccum rowSum s rowSum
      sync

    divCol o o rowSum

  let code := generateKernel kernel

  -- Check all expected components
  assertTrue (code.containsSubstr "flash_attn_test") "Should have kernel name"
  assertTrue (code.containsSubstr "gl<bf16, -1, -1, -1, -1> v0") "Should have first global descriptor param"
  assertTrue (code.containsSubstr "mma_ABt(") "Should have mmaT"
  assertTrue (code.containsSubstr "make_causal(") "Should have causal mask"
  assertTrue (code.containsSubstr "row_max(") "Should have row_max"
  assertTrue (code.containsSubstr "for (int") "Should have for loop"
  assertTrue (code.containsSubstr "div_row(") "Should have final normalization"

/-- Test complete LayerNorm-like kernel structure -/
@[test]
def testLayerNormStructure : IO Unit := do
  let kernel := buildKernelM "layernorm_test" .SM90 #[] do
    let x : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let xf : Tyr.GPU.Codegen.RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    let mean : Tyr.GPU.Codegen.RV GpuFloat.Float32 64 ← allocRV .Float32 64
    let var : Tyr.GPU.Codegen.RV GpuFloat.Float32 64 ← allocRV .Float32 64

    convert xf x
    rowSum mean xf
    subCol xf xf mean
    mul xf xf xf
    rowSum var xf
    -- rsqrt would go here
    convert x xf

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "row_sum(") "Should have row_sum"
  assertTrue (code.containsSubstr "sub_row(") "Should have sub_row"
  assertTrue (code.containsSubstr "mul(") "Should have mul"

end Tests.GPUKernels
