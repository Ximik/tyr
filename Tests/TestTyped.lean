import LeanTest
import Tyr.Typed
import Tyr.Typed.Neural
import Tyr.Torch

open torch

namespace Tests.TestTyped

@[test]
def testTypedDefaultsAndPlacement : IO Unit := do
  for dtype in #[DType.UInt8, .Int8, .Int16, .Int32, .Int64, .Float16, .BFloat16,
      .Float32, .Float64, .Bool, .Float8E4M3FN, .Float8E5M2, .Unknown "dynamic"] do
    let spec : StaticSpec := { shape := #[2, 3], dtype, device := .exact .CPU }
    let tensor : Tensor spec := default
    match tensor.validate with
    | .ok () => pure ()
    | .error message => LeanTest.fail s!"Invalid default for {dtype}: {message}"
    let moved := tensor.to .CPU
    match moved.validate with
    | .ok () => pure ()
    | .error message => LeanTest.fail s!"CPU move lost {dtype} contract: {message}"
    LeanTest.assertTrue (!moved.requiresGrad) "Defaults should not track gradients."
  let scalar := (Tensor.full #[] 2.5).setRequiresGrad true
  LeanTest.assertTrue scalar.requiresGrad "Typed autograd metadata should reflect requires_grad."
  LeanTest.assertEqual scalar.item 2.5 "Typed scalar extraction should preserve the value."

example (_matrix : DTensor #[2, 3] .Float32) : True := by
  fail_if_success
    have _ := Tensor.item _matrix
  trivial

/- Compile-time rejection tests. `fail_if_success` checks elaboration failure
   without inserting sorry-backed declarations into the runtime test module. -/
example (_x : DTensor #[2, 3] .Float32) : True := by
  fail_if_success
    have _wrongShape : DTensor #[3, 2] .Float32 := _x
  trivial

example (_x : DTensor #[2, 3] .Float32) (_w : DTensor #[4, 5] .Float32) : True := by
  fail_if_success
    have _wrongInnerDims := Tensor.mm _x _w
  trivial

example (_x : DTensor #[2, 3] .Float32) : True := by
  fail_if_success
    have _wrongNumel := _x.reshape #[5]
  trivial

example (_x : DTensor #[2, 3] .Float32) (_w : DTensor #[3, 4] .Int64) : True := by
  fail_if_success
    have _wrongDType := Tensor.mm _x _w
  trivial

@[test]
def testDTypePromotionPolicy : IO Unit := do
  LeanTest.assertEqual (DType.promote .Float16 .BFloat16) .Float32
    "mixed f16/bf16 should promote to f32"
  LeanTest.assertEqual (DType.promote .Int16 .UInt8) .Int16
    "small mixed integer promotion should keep enough signed range"
  LeanTest.assertEqual (DType.promote .UInt8 .Int8) .Int16
  LeanTest.assertEqual (DType.promote .Int8 .UInt8) .Int16
  LeanTest.assertEqual DType.Float8E4M3FN.bitWidth? (some 8)
  LeanTest.assertEqual DType.Float8E5M2.bitWidth? (some 8)
  LeanTest.assertEqual (DType.meanResult .Int64) .Float32
    "integer means should promote to a floating dtype"
  LeanTest.assertEqual (DType.sumResult .Float8E4M3FN) .Float32
    "float8 reductions should accumulate as f32 in the typed policy"

@[test]
def testTensorSpecAlgebra : IO Unit := do
  LeanTest.assertEqual (TensorSpec.numelOfShape #[2, 3, 4]) 24
  LeanTest.assertEqual (TensorSpec.broadcastShapes? #[3] #[2, 3]) (some #[2, 3])
  LeanTest.assertEqual (TensorSpec.broadcastShapes? #[2, 3] #[4, 3]) none
  LeanTest.assertEqual (TensorSpec.matmulShape? #[2, 3] #[3, 4]) (some #[2, 4])
  LeanTest.assertEqual (TensorSpec.matmulShape? #[2, 3] #[5, 4]) none
  let lhs : TensorSpec := { shape := #[2, 3], dtype := .Float16 }
  let rhs : TensorSpec := { shape := #[3], dtype := .BFloat16 }
  LeanTest.assertTrue (TensorSpec.pointwise? lhs rhs == some { shape := #[2, 3], dtype := .Float32 })
    "pointwise specs should combine broadcast shape and promoted dtype"

@[test]
def testStrictSpecChecks : IO Unit := do
  let expected : TensorSpec := { shape := #[2, 3], dtype := .Float32 }
  let same : TensorSpec := { shape := #[2, 3], dtype := .Float32 }
  let wrongShape : TensorSpec := { shape := #[3, 2], dtype := .Float32 }
  let wrongDType : TensorSpec := { shape := #[2, 3], dtype := .BFloat16 }
  match TensorSpec.checkCompatible expected same with
  | .ok () => pure ()
  | .error err => LeanTest.fail err
  match TensorSpec.checkCompatible expected wrongShape with
  | .ok () => LeanTest.fail "shape mismatch should fail"
  | .error err => LeanTest.assertTrue (err.containsSubstr "Expected shape")
  match TensorSpec.checkCompatible expected wrongDType with
  | .ok () => LeanTest.fail "dtype mismatch should fail"
  | .error err => LeanTest.assertTrue (err.containsSubstr "Expected dtype")

@[test]
def testDevicePolicyChecks : IO Unit := do
  match DevicePolicy.check (.exact .CPU) .CPU with
  | .ok () => pure ()
  | .error err => LeanTest.fail err
  match DevicePolicy.check (.exact .MPS) .CPU with
  | .ok () => LeanTest.fail "device mismatch should fail"
  | .error err => LeanTest.assertTrue (err.containsSubstr "Expected device")

/-- Assert that a typed tensor's runtime metadata matches its phantom spec
    (shape, dtype, and device policy). -/
private def assertValid {σ : StaticSpec}
    (t : Tensor σ) (label : String) : IO Unit := do
  match t.validate with
  | .ok () => pure ()
  | .error err => LeanTest.fail s!"{label}: {err}"

@[test]
def testTypedCheckedShapeBoundary : IO Unit := do
  let raw := torch.RawTensor.ones #[2]
  LeanTest.assertTrue (Tensor.ofTensor? (shape := #[99]) .Float32 raw).isNone
    "a native handle cannot establish the typed shape invariant"
  match Tensor.ofTensor (shape := #[99]) .Float32 raw with
  | .ok _ => LeanTest.fail "a mismatched runtime shape must be rejected"
  | .error error => LeanTest.assertTrue (error.containsSubstr "Expected shape")
  match Tensor.ofTensor (shape := #[2]) .Float32 raw with
  | .ok tensor => assertValid tensor "checked constructor"
  | .error error => LeanTest.fail error
  assertValid ((Tensor.ones #[1]).reshape #[]) "scalar reshape"
  assertValid ((Tensor.ones #[]).reshape #[1, 1]) "reshape from scalar"
  assertValid ((Tensor.ones #[0, 3]).reshape #[0]) "empty tensor reshape"

/-- Check the dtype algebra against actual tensor arithmetic for every pair
    of regular CPU dtypes, including the mixed unsigned/signed edge case.
    Float8 storage metadata is covered separately: CPU float8 arithmetic
    kernels are not generally available in libtorch. -/
@[test]
def testTypedDtypePromotionMatrix : IO Unit := do
  let path := "Tests/fixtures/typed/dtype_scalars.safetensors"
  let dtypes : Array DType := #[.Bool, .UInt8, .Int8, .Int16, .Int32, .Int64,
    .Float16, .BFloat16, .Float32, .Float64]
  for lhs in dtypes do
    let a ← safetensors.loadTensor path lhs.canonicalName #[1]
    let .ok a := Tensor.ofTensor (shape := #[1]) lhs a | LeanTest.fail s!"fixture dtype {lhs}"
    for rhs in dtypes do
      let b ← safetensors.loadTensor path rhs.canonicalName #[1]
      let .ok b := Tensor.ofTensor (shape := #[1]) rhs b | LeanTest.fail s!"fixture dtype {rhs}"
      assertValid (Tensor.add a b) s!"{lhs} + {rhs}"
  let u8 ← safetensors.loadTensor path "UInt8" #[1]
  let i8 ← safetensors.loadTensor path "Int8" #[1]
  LeanTest.assertEqual ((torch.add u8 i8).getValues.toList.toArray) #[254.0]
  for dtype in #[DType.Float8E4M3FN, DType.Float8E5M2] do
    let raw ← safetensors.loadTensor path dtype.canonicalName #[1]
    LeanTest.assertEqual raw.dtype dtype
    LeanTest.assertEqual dtype.bitWidth? (some 8)

/-- The typed wrappers re-index the phantom dtype according to the promotion
    algebra in `Tyr.Typed.DType`. These tests run the real ops and assert the
    runtime dtype agrees with the type-level claim, so the algebra cannot
    silently diverge from libtorch's behavior. -/
@[test]
def testTypedOpsDtypeParity : IO Unit := do
  -- same-dtype elementwise: preserve
  let f32 := Tensor.ones #[2, 2]
  let bf16 := f32.toBFloat16
  assertValid (f32 + f32) "f32 + f32"
  assertValid (bf16 * bf16) "bf16 * bf16"
  -- mixed-dtype elementwise: promote (f32 absorbs bf16)
  assertValid (f32 + bf16) "f32 + bf16 promotes"
  LeanTest.assertEqual (f32 + bf16).raw.dtype DType.Float32
    "runtime add(f32, bf16) should produce f32"
  -- true division floats integers
  let i64 := Tensor.fullInt #[4] 3
  assertValid (i64 / Tensor.fullInt #[4] 2) "i64 / i64 floats"
  LeanTest.assertEqual (i64 / Tensor.fullInt #[4] 2).raw.dtype DType.Float32
    "runtime div(i64, i64) should produce f32"
  -- float scalars promote integer tensors
  assertValid (i64 * (2.0 : Float)) "i64 * float scalar floats"
  assertValid (bf16 * (2.0 : Float)) "bf16 * float scalar preserves"
  -- integral sums accumulate as i64
  assertValid i64.sumAll "sum(i64) stays i64"
  LeanTest.assertEqual i64.sumAll.raw.dtype DType.Int64
    "runtime sum(i64) should produce i64"
  -- float mean preserves
  assertValid bf16.meanAll "mean(bf16) preserves"

@[test]
def testTypedOpsShapeAndUnaryParity : IO Unit := do
  let x := Tensor.ones #[2, 3]
  -- float-only unary preserves dtype
  assertValid x.exp "exp(f32)"
  assertValid x.toBFloat16.silu "silu(bf16)"
  assertValid (x.softmax) "softmax(f32)"
  -- relu/abs work on any dtype and preserve it
  assertValid (Tensor.fullInt #[3] (-1)).relu "relu(i64)"
  assertValid (Tensor.fullInt #[3] (-1)).abs "abs(i64)"
  -- matmul family requires and preserves one dtype
  let w := Tensor.ones #[4, 3]
  assertValid (x.matmul w.transpose2d) "matmul(f32)"
  assertValid (x.mm w.transpose2d) "mm(f32)"
  assertValid (x.linear w) "linear(f32)"
  -- shape ops preserve dtype
  assertValid (x.reshape #[3, 2]) "reshape"
  assertValid (x.transpose 0 1) "transpose"
  assertValid (x.unsqueeze 0) "unsqueeze"
  assertValid (Tensor.cat x x 0) "cat"
  -- casts land where they claim
  assertValid x.toBFloat16 "toBFloat16"
  assertValid x.toBFloat16.toFloat32 "toFloat32"
  assertValid x.toInt64 "toInt64"

/-- Pure broadcast-shape algebra (NumPy/PyTorch rules: right-align, pad with
    leading 1s, dims must match or be 1). -/
@[test]
def testBroadcastShapeAlgebra : IO Unit := do
  LeanTest.assertEqual (TensorSpec.broadcastShapes? #[2, 3] #[3]) (some #[2, 3])
    "trailing-dim bias pattern"
  LeanTest.assertEqual (TensorSpec.broadcastShapes? #[3, 2, 1] #[2, 7]) (some #[3, 2, 7])
    "rank extension plus dim-1 expansion"
  LeanTest.assertEqual (TensorSpec.broadcastShapes? #[1, 3] #[2, 1]) (some #[2, 3])
    "mutual expansion produces a shape neither input has"
  LeanTest.assertEqual (TensorSpec.broadcastShapes? #[] #[2, 3]) (some #[2, 3])
    "scalar broadcasts to anything"
  LeanTest.assertEqual (TensorSpec.broadcastShapes? #[2, 3] #[4, 3]) none
    "mismatched non-1 dims must fail"
  LeanTest.assertEqual (TensorSpec.broadcastList? [#[1, 2, 3], #[2, 3], #[]]) (some #[1, 2, 3])
    "n-ary broadcast folds the binary rule"
  LeanTest.assertEqual (TensorSpec.broadcastList? []) none
    "empty list has no broadcast shape"
  LeanTest.assertTrue (TensorSpec.broadcastShapes? #[3] #[2, 3] == some #[2, 3])
    "fixed-side check: #[3] broadcastsTo #[2, 3]"
  LeanTest.assertTrue (TensorSpec.broadcastShapes? #[2, 3] #[3] != some #[3])
    "fixed-side check: #[2, 3] does not broadcastTo #[3]"

/-- Broadcasting typed ops: output shape is inferred at compile time via the
    `rfl` auto-param; these assert the runtime (native libtorch broadcasting
    under type erasure) lands on the same shape and dtype. -/
@[test]
def testBroadcastOpsParity : IO Unit := do
  -- bias-add pattern: [2, 3] + [3]
  let x := Tensor.ones #[2, 3]
  let bias := Tensor.full #[3] 2.0
  let y := x.addB bias
  assertValid y "addB bias pattern"
  let vals ← data.tensorToFloatArray' y.raw
  LeanTest.assertEqual vals.size 6 "bias add yields 2x3 elements"
  for v in vals do
    LeanTest.assertTrue (v == 3.0) s!"1 + 2 broadcast value, got {v}"
  -- mask pattern: mutual expansion [1, 3] * [2, 1] -> [2, 3]
  let m := Tensor.ones #[1, 3]
  let n := Tensor.full #[2, 1] 5.0
  assertValid (m.mulB n) "mulB mutual expansion"
  -- mixed dtype with broadcast: f32 [2, 3] + bf16 [3] -> f32 [2, 3]
  assertValid (x.addB bias.toBFloat16) "addB promotes across broadcast"
  -- integer division broadcasts and floats
  assertValid ((Tensor.fullInt #[2, 2] 7).divB (Tensor.fullInt #[2] 2)) "divB floats ints"
  -- subB scalar-shape operand
  assertValid (x.subB (Tensor.full #[] 1.0)) "subB against scalar tensor"
  -- asymmetric broadcastTo is a view onto the target shape
  assertValid (bias.broadcastTo #[2, 3]) "broadcastTo bias"
  assertValid ((Tensor.ones #[2, 1]).broadcastTo #[2, 3]) "broadcastTo dim-1 expansion"
  -- the checked broadcast operators expand to addB/mulB, so the
  -- broadcastability proof is discharged at the call site
  let opSum := x +ᵇ bias
  assertValid opSum "operator +ᵇ broadcasts"
  let opVals ← data.tensorToFloatArray' opSum.raw
  for v in opVals do
    LeanTest.assertTrue (v == 3.0) s!"operator broadcast value, got {v}"
  assertValid (x *ᵇ bias) "operator *ᵇ broadcasts"
  -- a shape-only (dynamic dtype) claim validates against any runtime dtype
  let dyn : Tensor { shape := #[2, 3] } := Tensor.assumeSpec (torch.RawTensor.ones #[2, 3])
  assertValid dyn "shape-only claim is dtype-agnostic"

/-- Fully typed model layers: shape and dtype agree end to end, runtime
    metadata matches the phantom claims, and values are sane. -/
@[test]
def testTypedLayers : IO Unit := do
  -- Linear with bias: [2, 3] -> [2, 4] and [2, 5, 3] -> [2, 5, 4]
  let lin ← Typed.Linear.init 3 4
  let x2 := Tensor.ones #[2, 3]
  assertValid (lin.forward2d x2) "typed Linear forward2d"
  let x3 := Tensor.ones #[2, 5, 3]
  assertValid (lin.forward3d x3) "typed Linear forward3d (affine)"
  -- without bias the 3d path goes through linear3d
  let linNb ← Typed.Linear.init 3 4 (withBias := false)
  assertValid (linNb.forward3d x3) "typed Linear forward3d (no bias)"
  -- RMSNorm: normalizing all-ones with unit weights gives ~1.0 everywhere
  let rn := Typed.RMSNorm.init 3
  let y := rn.forward3d x3
  assertValid y "typed RMSNorm forward3d"
  let vals ← data.tensorToFloatArray' y.raw
  LeanTest.assertEqual vals.size 30 "rmsnorm preserves numel"
  for v in vals do
    LeanTest.assertTrue ((v - 1.0).abs < 1e-3) s!"rmsnorm(ones) ≈ 1, got {v}"
  -- bf16 layer: the f32 stability upcast is visible in the result type
  -- (promote Float32 BFloat16 = Float32), and the runtime agrees
  let rnBf : Typed.RMSNorm 3 .BFloat16 := { weight := (Tensor.ones #[3]).toBFloat16 }
  let yBf := rnBf.forward2d x2.toBFloat16
  assertValid yBf "typed RMSNorm bf16 upcasts to f32"
  LeanTest.assertEqual yBf.raw.dtype DType.Float32
    "rmsnorm computes in f32 regardless of layer dtype"
  assertValid yBf.toBFloat16 "explicit cast back restores bf16"

/-- Device policy in the spec: `.exact` pins a tensor's placement, `validate`
    checks it against the runtime device, and spec-preserving ops keep the
    pin. (Cross-policy application errors are compile-time and covered by
    the prototype probes in `dev/static_spec_tensor_proto.lean`.) -/
@[test]
def testDevicePinnedSpec : IO Unit := do
  let pinned : Tensor { shape := #[2, 2], dtype := .Float32, device := .exact .CPU } :=
    Tensor.assumeSpec (torch.RawTensor.ones #[2, 2])
  assertValid pinned "CPU-pinned tensor on CPU"
  assertValid pinned.relu "spec-preserving op keeps the pin"
  assertValid (Add.add pinned pinned) "homogeneous add keeps the pin"
  -- a wrong pin is a *runtime* validate failure (the claim is checked, not trusted)
  let misPinned : Tensor { shape := #[2, 2], dtype := .Float32, device := .exact (.CUDA 0) } :=
    Tensor.assumeSpec (torch.RawTensor.ones #[2, 2])
  match misPinned.validate with
  | .ok () => LeanTest.fail "CUDA pin on a CPU tensor should fail validation"
  | .error err => LeanTest.assertTrue (err.containsSubstr "device") s!"got: {err}"

/-- The stack bridge consumes the temporary raw array; Lean must not release it twice. -/
@[test]
def testTypedStackArrayOwnership : IO Unit := do
  for i in [:128] do
    let a := Tensor.fromFloatArray #[i.toFloat, 2, 3]
    let b := Tensor.fromFloatArray #[4, 5, 6]
    let vectors := #[a, b]
    let stacked := Tensor.stack1d vectors
    LeanTest.assertEqual (← stacked.toFloatArray) #[i.toFloat, 2, 3, 4, 5, 6]
    -- Allocate additional shape arrays and reuse the original tensor handles.
    let reshaped := stacked.reshapeChecked #[3, 2]
    LeanTest.assertEqual reshaped.runtimeShape #[3, 2]
    LeanTest.assertEqual (← vectors[0]!.toFloatArray) #[i.toFloat, 2, 3]
    LeanTest.assertEqual (← (Tensor.stack1d vectors).toFloatArray)
      #[i.toFloat, 2, 3, 4, 5, 6]

@[test]
def testTypedExplicitCotangentBackward : IO Unit := do
  let input := (Tensor.fromFloatArray #[2.0, 3.0]).setRequiresGrad true
  let output := Tensor.mul input input
  LeanTest.assertTrue (Tensor.hasGradFn output)
    "Squared differentiable input should retain its graph"
  LeanTest.assertTrue (!Tensor.hasGradFn output.detach)
    "Detached output should not retain its graph"
  let cotangent := Tensor.fromFloatArray #[4.0, 5.0]
  Tensor.backward output cotangent
  LeanTest.assertEqual (← input.gradOf.toFloatArray) #[16.0, 30.0]
    "Backward must multiply the derivative by the explicit cotangent"

end Tests.TestTyped
