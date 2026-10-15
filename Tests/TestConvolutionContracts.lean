import LeanTest
import Tyr.Typed.Neural

open torch

namespace Tests.TestConvolutionContracts

private def checked {α : Type} (result : Except String α) : IO α :=
  match result with
  | .ok value => pure value
  | .error message => LeanTest.fail message

private def expectError {α : Type} (result : Except String α) (fragment : String := "") : IO Unit :=
  match result with
  | .ok _ => LeanTest.fail "Expected a recoverable convolution error"
  | .error message => do
    LeanTest.assertTrue (!message.isEmpty)
    LeanTest.assertTrue (message.containsSubstr fragment) message

example (_input : Tensor ⟨#[1, 1, 3], .Float32, .CPU⟩)
    (_weight : Tensor ⟨#[1, 1, 2], .Float32, .CUDA 0⟩) : True := by
  fail_if_success have _ := Tensor.conv1d? _input _weight
  trivial

example (_input : Tensor ⟨#[1, 1, 3, 3], .Float32, .CPU⟩)
    (_weight : Tensor ⟨#[1, 1, 2, 2], .Float32, .CPU⟩)
    (_bias : Tensor ⟨#[1], .Float32, .CUDA 0⟩) : True := by
  fail_if_success have _ := Tensor.conv2dBias? _input _weight _bias
  trivial

example (input : Tensor ⟨#[1, 1, 5, 5], .Float32, .CPU⟩)
    (weight : Tensor ⟨#[1, 1, 2, 2], .Float32, .CPU⟩)
    (bias : Tensor ⟨#[1], .Float32, .CPU⟩) :
    Except String (Tensor ⟨#[1, 1, 2, 2], .Float32, .CPU⟩) :=
  Tensor.conv2dBias? input weight bias (stride := #[2])

example (input : Tensor ⟨#[1, 1, 5, 5], .Float32, .CPU⟩)
    (weight : Tensor ⟨#[1, 1, 2, 2], .Float32, .CPU⟩)
    (bias : Tensor ⟨#[1], .Float32, .CPU⟩) :
    Tensor ⟨#[1, 1, 3, 3], .Float32, .CPU⟩ :=
  Tensor.conv2dBias input weight bias (dilation := #[2])

@[test]
def testConvolutionMetadataFailureIsRecoverable : IO Unit := do
  let input := (Tensor.ones #[1, 1, 3, 3]).forgetDType
  let weight := (Tensor.ones #[1, 1, 2, 2]).forgetDType
  let bias := (Tensor.zeros #[1]).forgetDType
  let otherWeight := (weight.castDType .Float64).forgetDType
  let otherBias := (bias.castDType .Float64).forgetDType
  expectError (Tensor.conv2dBias? input otherWeight bias) "Expected dtype Float32, got Float64"
  expectError (Tensor.conv2dBias? input weight otherBias) "Expected dtype Float32, got Float64"
  let line := (Tensor.ones #[1, 1, 3]).forgetDType
  let kernel := (Tensor.ones #[1, 1, 2]).forgetDType
  let otherKernel := (kernel.castDType .Float64).forgetDType
  expectError (Tensor.conv1d? line otherKernel) "Expected dtype Float32, got Float64"
  expectError (Tensor.conv1dGroups? line otherKernel bias) "Expected dtype Float32, got Float64"
  expectError (Tensor.conv1dGroups? line kernel otherBias) "Expected dtype Float32, got Float64"
  expectError (Tensor.convTranspose1d? line otherKernel bias) "Expected dtype Float32, got Float64"
  expectError (Tensor.convTranspose1d? line kernel otherBias) "Expected dtype Float32, got Float64"
  let output ← checked (Tensor.conv2dBias? (input.castDType .Float64).forgetDType otherWeight otherBias)
  LeanTest.assertEqual output.raw.dtype .Float64
  LeanTest.assertEqual (← output.toFloatArray) #[4, 4, 4, 4]

@[test]
def testConvolutionArgumentsAreRecoverable : IO Unit := do
  let line := (Tensor.fromFloatArray #[1, 2, 3]).reshapeChecked #[1, 1, 3]
  let kernel := Tensor.ones #[1, 1, 2]
  let bias := Tensor.zeros #[1]
  expectError (Tensor.conv1d? line kernel (stride := 0))
  expectError (Tensor.conv1d? line kernel (dilation := 0))
  expectError (Tensor.conv1d? line kernel (padding := 18446744073709551615))
  expectError (Tensor.conv1dGroups? line kernel bias (groups := 0))
  expectError (Tensor.conv1dGroups? line kernel bias (groups := 2))
  expectError (Tensor.convTranspose1d? line kernel bias (stride := 0))
  expectError (Tensor.convTranspose1d? line kernel bias (stride := 2) (outputPadding := 2))
  let image := (Tensor.fromFloatArray #[1, 2, 3, 4, 5, 6, 7, 8, 9]).reshapeChecked #[1, 1, 3, 3]
  let square := Tensor.ones #[1, 1, 2, 2]
  expectError (Tensor.conv2dBias? image square bias (stride := #[0, 1]))
  expectError (Tensor.conv2dBias? image square bias (stride := #[1, 1, 1]))
  expectError (Tensor.conv2dBias? image square bias (dilation := #[1, 0]))
  -- Every family remains usable after native failures, with exact values and shapes.
  let lineResult ← checked (Tensor.conv1d? line kernel)
  LeanTest.assertEqual lineResult.runtimeShape #[1, 1, 2]
  LeanTest.assertEqual (← lineResult.toFloatArray) #[3, 5]
  let imageResult ← checked (Tensor.conv2dBias? image square (Tensor.full #[1] 0.5))
  LeanTest.assertEqual imageResult.runtimeShape #[1, 1, 2, 2]
  LeanTest.assertEqual (← imageResult.toFloatArray) #[12.5, 16.5, 24.5, 28.5]
  let groupedInput := (Tensor.fromFloatArray #[1, 2, 3, 4, 5, 6]).reshapeChecked #[1, 2, 3]
  let groupedWeight := (Tensor.fromFloatArray #[2, 3]).reshapeChecked #[2, 1, 1]
  let grouped ← checked (Tensor.conv1dGroups? groupedInput groupedWeight (Tensor.zeros #[2]) (groups := 2))
  LeanTest.assertEqual grouped.runtimeShape #[1, 2, 3]
  LeanTest.assertEqual (← grouped.toFloatArray) #[2, 4, 6, 12, 15, 18]
  let short := (Tensor.fromFloatArray #[1, 2]).reshapeChecked #[1, 1, 2]
  let expanded ← checked (Tensor.convTranspose1d? short short bias (stride := 2))
  LeanTest.assertEqual expanded.runtimeShape #[1, 1, 4]
  LeanTest.assertEqual (← expanded.toFloatArray) #[1, 2, 2, 4]

@[test]
def testConvolutionSingletonSpatialParameters : IO Unit := do
  let input := Tensor.ones #[1, 1, 5, 5]
  let weight := Tensor.ones #[1, 1, 2, 2]
  let bias := Tensor.zeros #[1]
  let strided ← checked (Tensor.conv2dBias? input weight bias (stride := #[2]))
  LeanTest.assertEqual strided.runtimeShape #[1, 1, 2, 2]
  LeanTest.assertEqual (← strided.toFloatArray) #[4, 4, 4, 4]
  let padded ← checked (Tensor.conv2dBias? input weight bias (padding := #[1]))
  LeanTest.assertEqual padded.runtimeShape #[1, 1, 6, 6]
  let explicitPadding ← checked (Tensor.conv2dBias? input weight bias (padding := #[1, 1]))
  LeanTest.assertEqual (← padded.toFloatArray) (← explicitPadding.toFloatArray)
  let dilated ← checked (Tensor.conv2dBias? input weight bias (dilation := #[2]))
  LeanTest.assertEqual dilated.runtimeShape #[1, 1, 3, 3]
  LeanTest.assertEqual (← dilated.toFloatArray) #[4, 4, 4, 4, 4, 4, 4, 4, 4]
  for invalid in #[#[], #[1, 1, 1]] do
    expectError (Tensor.conv2dBias? input weight bias (stride := invalid))
    expectError (Tensor.conv2dBias? input weight bias (padding := invalid))
    expectError (Tensor.conv2dBias? input weight bias (dilation := invalid))

@[test]
def testCheckedConvolutionRetainsComplexDType : IO Unit := do
  let input := ((Tensor.fromFloatArray #[1, 2, 3]).reshapeChecked #[1, 1, 3]).castDType .Complex64
  let weight := (Tensor.ones #[1, 1, 2]).castDType .Complex64
  let output ← checked (Tensor.conv1d? input weight)
  LeanTest.assertEqual output.raw.dtype .Complex64
  LeanTest.assertEqual (← output.toFloat32.toFloatArray) #[3, 5]

end Tests.TestConvolutionContracts
