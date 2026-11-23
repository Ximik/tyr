import LeanTest
import Tyr.Module.Linear
import Tyr.Module.Derive
import Tyr.Torch

namespace Tests.ModuleTyped

open torch

-- Linear maps require floating or complex dtypes; integral data is cast explicitly.
example (_layer : Linear 2 3 .Int32) (_x : Tensor ⟨#[4, 2], .Int32, .CPU⟩) : True := by
  fail_if_success have _ := _layer.forward2d _x
  trivial
example (layer : Linear 2 3 .Complex64) (x : Tensor ⟨#[4, 2], .Complex64, .CPU⟩) :
    Tensor ⟨#[4, 3], .Complex64, .CPU⟩ := layer.forward2d x
example : FloatingTensorStruct (Linear 2 3 .Float64) := inferInstance

private structure MyLinear (inDim outDim : Nat) where
  weight : Tensor { shape := #[outDim, inDim], dtype := .Float32, device := .CPU }
  bias : Option (Tensor { shape := #[outDim], dtype := .Float32, device := .CPU }) := none
  deriving TensorStruct

private instance {inDim outDim batch : Nat} :
    Module (MyLinear inDim outDim) (Tensor { shape := #[batch, inDim], dtype := .Float32, device := .CPU })
      (Tensor { shape := #[batch, outDim], dtype := .Float32, device := .CPU }) where
  forward m x :=
    let y := Tensor.linear x m.weight
    match m.bias with
    | some b => Add.add y (Tensor.broadcastTo b #[batch, outDim])
    | none => y

@[test]
def testPureModuleForward : IO Unit := do
  let lin : MyLinear 3 2 := ⟨Tensor.ones #[2, 3], some (Tensor.ones #[2])⟩
  let input := Tensor.ones #[4, 3]
  let pureOutput : Tensor { shape := #[4, 2], dtype := .Float32, device := .CPU } := Module.forward lin input
  LeanTest.assertEqual (Tensor.item (Tensor.sumAll pureOutput)) 32.0
    "Each output is the sum of three unit products plus a unit bias"

@[test]
def testUniformLikeDTypeAndErrors : IO Unit := do
  for dtype in #[DType.Float16, .BFloat16, .Float32, .Float64] do
    let support ← dtype.acquireConstantSupport .CPU
    let _ := support.down
    let reference : Tensor ⟨#[2, 3], dtype, .CPU⟩ := Tensor.zerosFor _
    let sample ← Tensor.uniformLike reference (-0.25) 0.5
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

end Tests.ModuleTyped
