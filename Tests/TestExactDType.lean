import LeanTest
import Tests.Devices
import Tyr.Typed.Neural

namespace Tests.ExactDType

open torch

example {σ : StaticSpec} (tensor : Tensor σ) : SomeDTypeTensor σ.shape σ.device :=
  tensor.packDType

example (_tensor : Tensor ⟨#[2], .Float32, .CPU⟩) : True := by
  fail_if_success
    have _ : SomeDTypeTensor #[2] .CPU := _tensor
  trivial

example {_otherDevice : Device} (_packed : SomeDTypeTensor #[2] .CPU) : True := by
  fail_if_success
    have _ : Tensor ⟨#[2], .Float32, .CPU⟩ := _packed
  fail_if_success
    have _ : Tensor ⟨#[2], .Float32, .CPU⟩ := _packed.2
  fail_if_success
    have _ : SomeDTypeTensor #[2] _otherDevice := _packed
  trivial

@[test] def testExactDTypeReificationRetainsValuesAndMetadata : IO Unit := do
  for dtype in #[DType.Float32, .BFloat16, .Int64, .Complex64] do
    let source := ((Tensor.fromFloatArray #[2, -3]).castDType dtype)
    let packed := source.packDType
    LeanTest.assertEqual packed.1 dtype
    LeanTest.assertEqual packed.2.dtype dtype
    LeanTest.assertEqual packed.2.raw.dtype dtype
    LeanTest.assertEqual packed.2.runtimeShape #[2]
    LeanTest.assertTrue (packed.2.runtimeDevice == .CPU)
    LeanTest.assertEqual (← packed.2.toFloatArray) #[2, -3]
    LeanTest.assertEqual source.raw.dtype dtype

private def alignDType {shape : Shape} {device : Device}
    (left right : SomeDTypeTensor shape device) :
    Except String (Tensor ⟨shape, left.1, device⟩) := do
  let ⟨same⟩ ← left.sameDType? right
  pure (same.symm ▸ right.2)

@[test] def testExactDTypeEqualityBeforeSharedDTypeUse : IO Unit := do
  let source := (Tensor.fromFloatArray #[1, 2, 3, 4]).reshape #[2, 2]
  let left := source.packDType
  let right := source.packDType
  let aligned ← IO.ofExcept (alignDType left right)
  LeanTest.assertEqual (← (Tensor.mm! left.2 aligned).toFloatArray) #[7, 10, 15, 22]
  let different := (source.castDType .BFloat16).packDType
  LeanTest.assertTrue (left.sameDType? different).toOption.isNone
  LeanTest.assertTrue (left.atDType? .Int64).toOption.isNone
  let recovered ← IO.ofExcept (left.atDType? .Float32)
  LeanTest.assertEqual (← recovered.toFloatArray) #[1, 2, 3, 4]

@[test] def testExactDTypePackagingPreservesAutogradIdentity : IO Unit := do
  let parameter ← (Tensor.fromFloatArray #[2, 3]).parameter
  let staticallyPacked := parameter.packDType
  let packed := staticallyPacked.2.packDType
  let recovered ← IO.ofExcept (packed.atDType? .Float32)
  LeanTest.assertTrue recovered.requiresGrad
  LeanTest.assertFalse recovered.hasGradFn
  let intermediate := recovered * recovered
  Tensor.backwardLoss (Tensor.sumAll intermediate) (retainGraph := some true)
  LeanTest.assertEqual (← (← parameter.gradOf).toFloatArray) #[4, 6]
  LeanTest.assertEqual (← (← recovered.gradOf).toFloatArray) #[4, 6]
  parameter.zeroGrad
  let graph := intermediate.packDType
  let recoveredGraph ← IO.ofExcept (graph.atDType? .Float32)
  LeanTest.assertTrue recoveredGraph.requiresGrad
  LeanTest.assertTrue recoveredGraph.hasGradFn
  Tensor.backwardLoss (Tensor.sumAll recoveredGraph)
  LeanTest.assertEqual (← (← parameter.gradOf).toFloatArray) #[4, 6]

@[test] def testExactDTypeReificationKeepsImaginaryValues : IO Unit := do
  let negative : Tensor ⟨#[1], .Complex64, .CPU⟩ :=
    (Tensor.full #[1] (-4)).castDType .Complex64
  let imaginary := Tensor.sqrt negative
  let packed := imaginary.packDType
  let recovered ← IO.ofExcept (packed.atDType? .Complex64)
  LeanTest.assertEqual (← (Tensor.abs recovered).toFloatArray) #[2]
  LeanTest.assertEqual (← (recovered * recovered).toFloatArray) #[-4]

end Tests.ExactDType
