import Tests.Assertions
import LeanTest
import Tyr.Typed.Layers
import Tyr.Module.RMSNorm
import Tyr.Model.Utils
import Tyr.Model.Flux.TypedOps
import Tyr.Model.KittenTTS.TypedOps
import Tyr.Model.Qwen35.Multimodal
import Tyr.Model.Whisper.TypedOps
import Tyr.Torch

open torch

example : True := by
  fail_if_success have _ := Tensor.onesFor ⟨#[2], .QInt8, .CPU⟩
  fail_if_success have _ := Tensor.eyeFor (n := 2) .Float8E8M0FNU .CPU
  fail_if_success have _ : Tensor ⟨#[2], .QInt8, .CPU⟩ := default
  trivial

namespace Tests.TestFactoryRecovery

/-- Invalid runtime dimensions are rejected before pure factory evaluation. -/
@[test]
def testInitializerShapeValidation : IO Unit := do
  let invalid : Nat := 18446744073709551615
  Tests.assertThrows (Typed.RMSNorm.init invalid)
  Tests.assertThrows (RMSNorm.init invalid)
  Tests.assertThrows (Model.initBias #[invalid])
  Tests.assertThrows (IO.ofExcept (Shape.validateAllocation #[invalid, invalid]))
  Tests.assertThrows (IO.ofExcept (Shape.validateAllocation #[0, 4294967296, 4294967296]))
  IO.ofExcept (Shape.validateAllocation #[4294967296, 0, 4294967296])

/-- Model initializers must reject dimensions before evaluating pure constants. -/
@[test]
def testModelConstantInitializerShapeValidation : IO Unit := do
  let invalid : Nat := 9223372036854775808
  Tests.assertThrows (flux.initLayerNorm invalid)
  Tests.assertThrows (whisper.initLayerNorm invalid)
  Tests.assertThrows (qwen35.Qwen35RMSNorm.initZeroCentered invalid)
  Tests.assertThrows (qwen35.Qwen35RMSNorm.initOneCentered invalid)
  Tests.assertThrows (qwen35.Qwen35RMSNormGated.init invalid)
  Tests.assertThrows (qwen35.Qwen35VisionLayerNorm.init invalid)
  let norm ← flux.initLayerNorm 2
  LeanTest.assertEqual (← norm.weight.toFloatArray) #[1.0, 1.0]
  LeanTest.assertEqual (← norm.bias.toFloatArray) #[0.0, 0.0]
  LeanTest.assertTrue norm.weight.requiresGrad

/-- Random factories retain their effects while validating the pure template. -/
@[test]
def testRandomTemplateShapeValidation : IO Unit := do
  let invalid : Shape := #[9223372036854775808]
  Tests.assertThrows (kittentts.uniformTensor (device := .CPU) invalid 0 1)
  Tests.assertThrows (kittentts.randTensor invalid)
  Tests.assertThrows (qwen35.randDynamic invalid)
  let values ← kittentts.uniformTensor (device := .CPU) #[2] 0 1
  LeanTest.assertTrue ((← values.toFloatArray).all fun x => x >= 0 && x <= 1)
  LeanTest.assertEqual values.runtimeShape #[2]

/-- Device validation is a separate acquisition, before constructing tensors. -/
@[test]
def testDeviceAcquisitionPrecedesInitialization : IO Unit := do
  Tests.assertThrows (Device.cuda 128)

@[test]
def testPureConstantFactoriesPreserveSpecification : IO Unit := do
  let zeros := Tensor.zerosFor { shape := #[2], dtype := .Float64, device := .CPU }
  let ones := Tensor.onesFor { shape := #[2], dtype := .BFloat16, device := .CPU }
  LeanTest.assertEqual zeros.raw.dtype .Float64
  LeanTest.assertEqual ones.raw.dtype .BFloat16
  LeanTest.assertEqual (← zeros.toFloatArray) #[0.0, 0.0]
  LeanTest.assertEqual (← ones.toFloatArray) #[1.0, 1.0]
  LeanTest.assertTrue (!zeros.requiresGrad && !ones.requiresGrad)
  let single := Tensor.ones #[2]
  LeanTest.assertEqual single.raw.dtype .Float32
  Tests.assertThrows (DType.QInt8.acquireConstantSupport .CPU)
  let norm ← RMSNorm.init 2 (dtype := .Float64)
  LeanTest.assertEqual norm.weight.raw.dtype .Float64
  LeanTest.assertTrue norm.weight.requiresGrad
  let identity := Tensor.eyeFor (n := 2) .Float64
  LeanTest.assertEqual identity.raw.dtype .Float64
  LeanTest.assertEqual (← identity.toFloatArray) #[1.0, 0.0, 0.0, 1.0]

end Tests.TestFactoryRecovery
