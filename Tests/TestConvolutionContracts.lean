import LeanTest
import Tyr.Typed.Neural

open torch

namespace Tests.TestConvolutionContracts

/-! Operand dtype and device are fixed by the signatures; integer inputs have no instance. -/
example {otherDevice : Device}
    (_image : Tensor ⟨#[1, 1, 3, 3], .Float32, .CPU⟩)
    (_square : Tensor ⟨#[1, 1, 2, 2], .Float32, .CPU⟩)
    (_otherSquare : Tensor ⟨#[1, 1, 2, 2], .Float64, .CPU⟩)
    (_remoteSquare : Tensor ⟨#[1, 1, 2, 2], .Float32, otherDevice⟩)
    (_intImage : Tensor ⟨#[1, 1, 3, 3], .Int64, .CPU⟩)
    (_intSquare : Tensor ⟨#[1, 1, 2, 2], .Int64, .CPU⟩)
    (_intBias : Tensor ⟨#[1], .Int64, .CPU⟩)
    (_line : Tensor ⟨#[1, 1, 3], .Float32, .CPU⟩)
    (_kernel : Tensor ⟨#[1, 1, 2], .Float32, .CPU⟩)
    (_otherKernel : Tensor ⟨#[1, 1, 2], .Float64, .CPU⟩)
    (_remoteKernel : Tensor ⟨#[1, 1, 2], .Float32, otherDevice⟩)
    (_bias : Tensor ⟨#[1], .Float32, .CPU⟩)
    (_otherBias : Tensor ⟨#[1], .Float64, .CPU⟩) : True := by
  have _ := Tensor.conv2dBias _image _square _bias
  fail_if_success have _ := Tensor.conv2dBias _image _otherSquare _bias
  fail_if_success have _ := Tensor.conv2dBias _image _square _otherBias
  fail_if_success have _ := Tensor.conv2dBias _image _remoteSquare _bias
  fail_if_success have _ := Tensor.conv2dBias _intImage _intSquare _intBias
  have _ := Tensor.conv1d _line _kernel
  fail_if_success have _ := Tensor.conv1d _line _otherKernel
  fail_if_success have _ := Tensor.conv1d _line _remoteKernel
  have _ := Tensor.conv1dGroups _line _kernel _bias
  fail_if_success have _ := Tensor.conv1dGroups _line _otherKernel _bias
  fail_if_success have _ := Tensor.conv1dGroups _line _kernel _otherBias
  have _ := Tensor.convTranspose1d _line _kernel _bias
  fail_if_success have _ := Tensor.convTranspose1d _line _otherKernel _bias
  fail_if_success have _ := Tensor.convTranspose1d _line _kernel _otherBias
  trivial

/-! Stride, dilation, groups and output padding are checked against the shapes. -/
example (_line : Tensor ⟨#[1, 1, 3], .Float32, .CPU⟩) (_kernel : Tensor ⟨#[1, 1, 2], .Float32, .CPU⟩)
    (_bias : Tensor ⟨#[1], .Float32, .CPU⟩)
    (_image : Tensor ⟨#[1, 1, 3, 3], .Float32, .CPU⟩)
    (_square : Tensor ⟨#[1, 1, 2, 2], .Float32, .CPU⟩) : True := by
  have _ := Tensor.conv1d _line _kernel (stride := 2) (dilation := 2)
  fail_if_success have _ := Tensor.conv1d _line _kernel (stride := 0)
  fail_if_success have _ := Tensor.conv1d _line _kernel (dilation := 0)
  -- A dilated window of 4 needs padding to fit 3 samples.
  have _ := Tensor.conv1d _line _kernel (dilation := 3) (padding := 1)
  fail_if_success have _ := Tensor.conv1d _line _kernel (dilation := 3)
  have _ := Tensor.conv1dGroups _line _kernel _bias (groups := 1)
  fail_if_success have _ := Tensor.conv1dGroups _line _kernel _bias (groups := 0)
  fail_if_success have _ := Tensor.conv1dGroups _line _kernel _bias (groups := 2)
  have _ := Tensor.convTranspose1d _line _kernel _bias (stride := 2) (outputPadding := 1)
  fail_if_success have _ := Tensor.convTranspose1d _line _kernel _bias (stride := 0)
  fail_if_success have _ := Tensor.convTranspose1d _line _kernel _bias (stride := 2) (outputPadding := 2)
  have _ := Tensor.conv2dBias _image _square _bias (stride := (2, 1)) (dilation := (1, 2))
  fail_if_success have _ := Tensor.conv2dBias _image _square _bias (stride := (0, 1))
  fail_if_success have _ := Tensor.conv2dBias _image _square _bias (dilation := (1, 0))
  fail_if_success have _ := Tensor.conv2dBias _image _square _bias (dilation := (3, 1))
  trivial

@[test]
def testConvolutionSpatialPairsAreHeightThenWidth : IO Unit := do
  let input := Tensor.ones #[1, 1, 5, 5]
  let weight := Tensor.ones #[1, 1, 2, 2]
  let bias := Tensor.zeros #[1]
  let strided := Tensor.conv2dBias input weight bias (stride := (2, 1))
  LeanTest.assertEqual strided.runtimeShape #[1, 1, 2, 4]
  let padded := Tensor.conv2dBias input weight bias (padding := (1, 0))
  LeanTest.assertEqual padded.runtimeShape #[1, 1, 6, 4]
  let dilated := Tensor.conv2dBias input weight bias (dilation := (2, 1))
  LeanTest.assertEqual dilated.runtimeShape #[1, 1, 3, 4]
  LeanTest.assertEqual (← dilated.toFloatArray) (Array.replicate 12 4)

@[test]
def testConvolutionRetainsWideDTypes : IO Unit := do
  let image := (Tensor.ones #[1, 1, 3, 3]).castDType .Float64
  let square := (Tensor.ones #[1, 1, 2, 2]).castDType .Float64
  let wide := Tensor.conv2dBias image square ((Tensor.zeros #[1]).castDType .Float64)
  LeanTest.assertEqual wide.raw.dtype .Float64
  LeanTest.assertEqual (← wide.toFloatArray) #[4, 4, 4, 4]
  let line := ((Tensor.fromFloatArray #[1, 2, 3]).reshape #[1, 1, 3]).castDType .Complex64
  let kernel := (Tensor.ones #[1, 1, 2]).castDType .Complex64
  let complex := Tensor.conv1d line kernel
  LeanTest.assertEqual complex.raw.dtype .Complex64
  LeanTest.assertEqual (← complex.toFloat32.toFloatArray) #[3, 5]

end Tests.TestConvolutionContracts
