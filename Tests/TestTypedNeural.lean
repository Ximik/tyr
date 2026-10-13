import Tyr.Typed.Neural
import Tyr.Module.RMSNorm
import LeanTest

open torch

@[test] def testTypedNeuralGatherAndMasks : IO Unit := do
  let x := (Tensor.fromFloatArray #[1, 2, 3, 4, 5, 6]).reshapeChecked #[2, 3]
  let indices := (Tensor.fromInt64Array #[2, 0, 1, 1]).reshapeChecked #[2, 2]
  let selected := Tensor.gather x (-1) indices
  LeanTest.assertEqual (← selected.toFloatArray) #[3, 1, 5, 5]
    "Signed gather dimension preserves native negative-axis semantics"
  let mask := (Tensor.fromInt64Array #[1, 0, 1, 0, 0, 1]).reshapeChecked #[2, 3] |>.eqScalar 1
  let packed := Tensor.maskedSelect x mask
  LeanTest.assertEqual packed.1 3
  LeanTest.assertEqual (← packed.2.toFloatArray) #[1, 3, 6]
  let replaced := Tensor.maskedScatter x mask (Tensor.fromFloatArray #[10, 30, 60])
  LeanTest.assertEqual (← replaced.toFloatArray) #[10, 2, 30, 4, 5, 60]
  let empty := Tensor.maskedSelect x (Tensor.fullInt #[2, 3] 0 |>.eqScalar 1)
  LeanTest.assertEqual empty.1 0
  LeanTest.assertEqual empty.2.runtimeShape #[0]

@[test] def testTypedNeuralConvAndExistentialConcat : IO Unit := do
  let x := (Tensor.fromFloatArray #[1, 2, 3, 4, 5, 6, 7, 8, 9]).reshapeChecked #[1, 1, 3, 3]
  let kernel := Tensor.ones #[1, 1, 2, 2]
  let result := Tensor.conv2dBias x kernel (Tensor.full #[1] 0.5)
  LeanTest.assertEqual result.runtimeShape #[1, 1, 2, 2]
  LeanTest.assertEqual (← result.toFloatArray) #[12.5, 16.5, 24.5, 28.5]
  let joined := Tensor.catManyChecked
    #[(Tensor.fromFloatArray #[1, 2]).packShape, (Tensor.fromFloatArray #[3]).packShape] 0 #[3]
  LeanTest.assertEqual (← joined.toFloatArray) #[1, 2, 3]

@[test] def testTypedRandintSignedBounds : IO Unit := do
  let sampled ← Tensor.randint (-7) (-6) #[16]
  LeanTest.assertEqual sampled.actualSpec.dtype DType.Int64
  LeanTest.assertEqual (← sampled.toFloatArray) (Array.replicate 16 (-7.0))
    "Lean Int64 bounds cross the C ABI with their signed bit pattern intact"

@[test] def testTypedRMSNormBFloat16Rounding : IO Unit := do
  let values := (Tensor.fromFloatArray
    #[0.17, -0.41, 0.83, 1.27, 0.9, 0.5, -0.7, 1.1]).reshapeChecked #[1, 2, 4]
  let weights := Tensor.fromFloatArray #[0.73, 1.19, -0.67, 1.53]
  for bf16 in #[false, true] do
    let x := if bf16 then values.toBFloat16.forgetDType else values.forgetDType
    let w := if bf16 then weights.toBFloat16.forgetDType else weights.forgetDType
    let norm : RMSNorm 4 (.Unknown "dynamic") := { weight := w, eps := ⟨0.01⟩ }
    let normalized := Tensor.rmsNormWeighted x.toFloat32 (Tensor.ones #[4]) 0.01
    let expected := (Tensor.castLike x normalized) * (w.expand #[1, 2, 4])
    let outputs := #[norm.forward3d x,
      (norm.forward2d (x.reshapeChecked #[2, 4])).reshapeChecked #[1, 2, 4],
      (norm.forward4d (x.reshapeChecked #[1, 1, 2, 4])).reshapeChecked #[1, 2, 4],
      (norm.forward5d (x.reshapeChecked #[1, 1, 2, 4])).reshapeChecked #[1, 2, 4]]
    for actual in outputs do
      LeanTest.assertEqual actual.actualSpec.dtype x.actualSpec.dtype
      LeanTest.assertEqual (← actual.toFloatArray) (← expected.toFloatArray)
        "RMSNorm restores activation dtype before multiplying non-unit weights"
