import Tests.Assertions
import LeanTest
import Tyr.Typed.Layers
import Tyr.Module.RMSNorm
import Tyr.Model.Utils

open torch

namespace Tests.TestFactoryRecovery

/-- UInt64 overflow becomes a negative native extent, so failure precedes allocation. -/
@[test]
def testInitializerShapeFailuresAreRecoverable : IO Unit := do
  let invalid : UInt64 := 18446744073709551615
  Tests.assertThrows (Typed.RMSNorm.init invalid)
  Tests.assertThrows (RMSNorm.init invalid)
  Tests.assertThrows (Model.initBias #[invalid])
  Tests.assertThrows (Tensor.eyeIO invalid)

/-- CUDA indices above DeviceIndex's range are invalid independently of hardware. -/
@[test]
def testInitializerDeviceFailuresAreRecoverable : IO Unit := do
  let invalid : Device := .CUDA 128
  Tests.assertThrows (Typed.RMSNorm.init 2 (device := invalid))
  Tests.assertThrows (RMSNorm.init 2 (device := invalid))
  Tests.assertThrows (Model.initBias #[2] (device := invalid))
  Tests.assertThrows (Tensor.eyeForIO (n := 2) .Float32 invalid)

@[test]
def testCheckedConstantFactoriesPreserveSpecification : IO Unit := do
  let zeros ← Tensor.zerosForIO { shape := #[2], dtype := .Float64, device := .CPU }
  let ones ← Tensor.onesForIO { shape := #[2], dtype := .BFloat16, device := .CPU }
  LeanTest.assertEqual zeros.raw.dtype .Float64
  LeanTest.assertEqual ones.raw.dtype .BFloat16
  LeanTest.assertEqual (← zeros.toFloatArray) #[0.0, 0.0]
  LeanTest.assertEqual (← ones.toFloatArray) #[1.0, 1.0]
  LeanTest.assertTrue (!zeros.requiresGrad && !ones.requiresGrad)
  let dynamic ← Tensor.onesForIO { shape := #[2], device := .CPU }
  LeanTest.assertEqual dynamic.raw.dtype .Float32
  Tests.assertThrows
    (Tensor.zerosForIO { shape := #[2], dtype := .Unknown "invalid", device := .CPU })
  Tests.assertThrows
    (Tensor.onesForIO { shape := #[2], dtype := .QInt8, device := .CPU })
  let norm ← RMSNorm.init 2 (dtype := .Float64)
  LeanTest.assertEqual norm.weight.raw.dtype .Float64
  LeanTest.assertTrue norm.weight.requiresGrad
  let identity ← Tensor.eyeForIO (n := 2) .Float64
  LeanTest.assertEqual identity.raw.dtype .Float64
  LeanTest.assertEqual (← identity.toFloatArray) #[1.0, 0.0, 0.0, 1.0]

end Tests.TestFactoryRecovery
