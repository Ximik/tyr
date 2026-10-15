import LeanTest
import Tyr.Module.Affine
import Tyr.Module.Conv2d
import Tyr.Module.Derive
import Tyr.Torch

namespace Tests.ModuleTyped

open torch

example (_a : Affine (n := 2) (m := 3)) (_x : Tensor { shape := #[4, 2], dtype := .Float32, device := .CPU }) : True := by
  fail_if_success
    have _ := _a.step _x
  trivial

example (_w : Tensor { shape := #[2, 3], dtype := .Float32, device := .CPU }) (_b : Tensor { shape := #[2], dtype := .Float32, device := .CPU }) : True := by
  fail_if_success
    have _ : Affine (n := 2) (m := 3) := ⟨_w, _b⟩
  trivial

private structure MyLinear (inDim outDim : UInt64) where
  weight : Tensor { shape := #[outDim, inDim], dtype := .Float32, device := .CPU }
  bias : Option (Tensor { shape := #[outDim], dtype := .Float32, device := .CPU }) := none
  deriving TensorStruct

private instance {inDim outDim batch : UInt64} :
    Module (MyLinear inDim outDim) (Tensor { shape := #[batch, inDim], dtype := .Float32, device := .CPU })
      (Tensor { shape := #[batch, outDim], dtype := .Float32, device := .CPU }) where
  forward m x :=
    let y := Tensor.linear x m.weight
    match m.bias with
    | some b => Add.add y (Tensor.expand b #[batch, outDim])
    | none => y

private structure MyDropout where
  p : Static Float := ⟨0.1⟩
  deriving TensorStruct

private instance {σ : StaticSpec} : ModuleIO MyDropout (Tensor σ) (Tensor σ) where
  forward m x := Tensor.dropout x m.p.val true

private structure MyLinearCtx (inDim outDim : UInt64) where
  lin : MyLinear inDim outDim
  deriving TensorStruct

private instance {inDim outDim batch : UInt64} :
    ModuleCtx (MyLinearCtx inDim outDim) TrainingCtx
      (Tensor { shape := #[batch, inDim], dtype := .Float32, device := .CPU }) (Tensor { shape := #[batch, outDim], dtype := .Float32, device := .CPU }) where
  forward m ctx x := do
    let y := Module.forward m.lin x
    Tensor.dropout y ctx.dropout_p ctx.training

@[test]
def testAffineNonSquareForwardAndGradient : IO Unit := do
  let weight : Tensor ⟨#[3, 2], .Float64, .CPU⟩ := Tensor.onesLike (Tensor.zerosFor _)
  let bias : Tensor ⟨#[2], .Float64, .CPU⟩ := Tensor.onesLike (Tensor.zerosFor _)
  let a : @Affine 2 3 .Float64 (.CPU) := {
    w := ← Tensor.parameter (Tensor.Preserving.mulScalar weight 3.0)
    b := ← Tensor.parameter (Tensor.Preserving.mulScalar bias 5.0) }
  let input : Tensor ⟨#[4, 3], .Float64, .CPU⟩ :=
    Tensor.Preserving.mulScalar (Tensor.onesLike (Tensor.zerosFor _)) 2.0
  let output : Tensor ⟨#[4, 2], .Float64, .CPU⟩ := Module.forward a input
  match output.validate with
  | .ok () => pure ()
  | .error message => LeanTest.fail message
  LeanTest.assertTrue
    (Tensor.frobeniusNorm (Sub.sub output (Tensor.Preserving.mulScalar (Tensor.onesLike output) 23.0)) < 1e-10)
    "Affine must compute input @ weight + bias with distinct input/output dimensions"
  let loss := Tensor.sumAll output
  Tensor.backwardLoss loss
  let gradient : @Affine 2 3 .Float64 (.CPU) ← differentiable.grad a
  LeanTest.assertTrue
    (Tensor.frobeniusNorm (Sub.sub gradient.w (Tensor.Preserving.mulScalar weight 8.0)) < 1e-10)
    "Affine weight gradient must sum all four input rows"
  LeanTest.assertTrue
    (Tensor.l2Norm (Sub.sub gradient.b (Tensor.Preserving.mulScalar bias 4.0)) < 1e-10)
    "Affine bias gradient must count batch rows"
  let tensorGradient : Tensor ⟨#[3, 2], .Float64, .CPU⟩ ← differentiable.grad a.w
  LeanTest.assertEqual (Tensor.frobeniusNorm (Sub.sub tensorGradient gradient.w)) 0.0
    "Tensor and module gradient interfaces must agree"
  let traversed : @Affine 2 3 .Float64 (.CPU) ← TensorStruct.grads a
  LeanTest.assertEqual (Tensor.frobeniusNorm (Sub.sub traversed.w gradient.w)) 0.0
    "Structural gradient traversal must retain indexed affine fields"

@[test]
def testDocumentedModuleInterfaces : IO Unit := do
  let lin : MyLinear 3 2 := ⟨Tensor.ones #[2, 3], some (Tensor.ones #[2])⟩
  let input := Tensor.ones #[4, 3]
  let pureOutput : Tensor { shape := #[4, 2], dtype := .Float32, device := .CPU } := Module.forward lin input
  let ioOutput : Tensor { shape := #[4, 2], dtype := .Float32, device := .CPU } ← ModuleIO.forward lin input
  LeanTest.assertEqual (Tensor.frobeniusNorm (pureOutput - ioOutput)) 0.0
    "Pure modules should lift to IO without changing their tensor contract"
  let contextual : Tensor { shape := #[4, 2], dtype := .Float32, device := .CPU } ←
    ModuleCtx.forward (MyLinearCtx.mk lin) TrainingCtx.eval input
  LeanTest.assertEqual (Tensor.frobeniusNorm (pureOutput - contextual)) 0.0
    "Evaluation context should retain the pure forward result"
  let dropped : Tensor { shape := #[4, 3], dtype := .Float32, device := .CPU } ← ModuleIO.forward (MyDropout.mk ⟨1.0⟩) input
  LeanTest.assertEqual (Tensor.frobeniusNorm dropped) 0.0
    "Training dropout with p=1 should zero every activation"
  match dropped.validate with
  | .ok () => pure ()
  | .error message => LeanTest.fail message

example : True := by
  fail_if_success
    have _ : Conv2d := {
      in_channels := 4, out_channels := 6, kernel_size := (2, 3), groups := 2
      weight := Tensor.zeros #[6, 4, 2, 3] }
  trivial

@[test]
def testConv2dIndexedInitialization : IO Unit := do
  let config : Conv2d .Float64 (.CPU) := {
    in_channels := 4, out_channels := 6, kernel_size := (2, 3), groups := 2
    outputBias := Tensor.onesLike (Tensor.zerosFor _) }
  let initialized ← config.init
  match initialized.weight.validate, initialized.outputBias.validate with
  | .ok (), .ok () => pure ()
  | .error message, _ | _, .error message => LeanTest.fail message
  LeanTest.assertEqual initialized.weight.runtimeShape #[6, 2, 2, 3]
    "Grouped convolution must allocate one input-channel block per group"
  LeanTest.assertEqual (Tensor.item (Tensor.sumAll initialized.outputBias)) 6.0
    "Weight initialization should retain the supplied bias"
  LeanTest.assertEqual (Tensor.item (Tensor.sumAll config.weight)) 0.0
    "Initialization must not mutate the template weight"
  let bound := 1.0 / Float.sqrt 12.0
  let values := initialized.weight.raw.getValues
  LeanTest.assertEqual values.size 72 "Every kernel element must be initialized"
  for i in [:values.size] do
    LeanTest.assertTrue (values[i]!.isFinite && values[i]! >= -bound && values[i]! <= bound)
      "Grouped-convolution initialization must obey its fan-in bound"
  LeanTest.assertTrue
    (Tensor.item (Tensor.sumAll (Mul.mul initialized.weight initialized.weight)) > 0.0)
    "Uniform initialization must replace the zero template"
  LeanTest.assertEqual (initialized.outShape #[2, 4, 10, 12]) #[2, 6, 9, 10]
    "Non-square kernels must retain their configured dimensions"

@[test]
def testUniformLikeDTypeAndErrors : IO Unit := do
  for dtype in #[DType.Float16, .BFloat16, .Float32, .Float64] do
    let reference : Tensor ⟨#[2, 3], dtype, .CPU⟩ := Tensor.zerosFor _
    let sample ← Tensor.uniformLike reference (-0.25) 0.5
    match sample.validate with
    | .ok () => pure ()
    | .error message => LeanTest.fail message
    let values := sample.raw.getValues
    for i in [:values.size] do
      LeanTest.assertTrue (values[i]!.isFinite && values[i]! >= -0.25 && values[i]! <= 0.5)
        s!"Uniform sample must retain valid {dtype} values and range"
  let reference := Tensor.zeros #[2, 3]
  let badBounds ← try
    let _ ← Tensor.uniformLike reference 1.0 (-1.0)
    pure false
  catch _ => pure true
  LeanTest.assertTrue badBounds "Reversed uniform bounds must return an IO error"
  let badDtype ← try
    let _ ← Tensor.uniformLike (Tensor.fullInt #[2, 3] 0)
    pure false
  catch _ => pure true
  LeanTest.assertTrue badDtype "Integral uniform initialization must return an IO error"

@[test]
def testConv2dRejectsInvalidInitialization : IO Unit := do
  let configs : Array Conv2d := #[
    { in_channels := 4, out_channels := 6, kernel_size := (2, 3), groups := 0 },
    { in_channels := 3, out_channels := 4, kernel_size := (2, 3), groups := 2 },
    { in_channels := 4, out_channels := 3, kernel_size := (2, 3), groups := 2 },
    { in_channels := 4, out_channels := 6, kernel_size := (0, 3), groups := 2 }]
  for config in configs do
    let rejected ← try
      let _ ← config.init
      pure false
    catch _ => pure true
    LeanTest.assertTrue rejected "Invalid convolution groups or kernels must be rejected"

end Tests.ModuleTyped
