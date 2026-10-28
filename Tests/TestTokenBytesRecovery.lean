import Tests.Assertions
import LeanTest
import Tyr.Tokenizer.TokenBytes
import Tyr.TensorStruct.Cast

namespace Tests.TokenBytesRecovery

open torch tokenizer

private def mapping : TokenBytes :=
  { vocabSize := 4, bytes := Tensor.fromInt64Array #[0, 1, 3, 5] }

@[test] def testInvalidArrayTokenIdsAreRecoverable : IO Unit := do
  Tests.assertThrows (mapping.totalBytes #[4]) (some "indexSelect")
  Tests.assertThrows (mapping.totalBytes #[4294967295]) (some "indexSelect")

@[test] def testInvalidTensorTokenIdsAreRecoverable : IO Unit := do
  Tests.assertThrows (mapping.totalBytesFromTensor (Tensor.fromInt64Array #[-1]))
    (some "indexSelect")
  Tests.assertThrows (mapping.totalBytesFromTensor (Tensor.fromInt64Array #[4]))
    (some "indexSelect")

@[test] def testInvalidMaskedTokenIdsAreRecoverable : IO Unit := do
  let ids := (Tensor.fromInt64Array #[1, 4]).reshape #[1, 2]
  -- Masking out an invalid token must not make an out-of-bounds lookup legal.
  let mask := (Tensor.fromFloatArray #[1, 0]).reshape #[1, 2]
  Tests.assertThrows (mapping.totalBytesWithMask ids mask) (some "indexSelect")
  Tests.assertThrows (mapping.totalBytesWithMask' ids mask) (some "indexSelect")
  let negative := (Tensor.fromInt64Array #[-1, 1]).reshape #[1, 2]
  Tests.assertThrows (mapping.totalBytesWithMask negative mask) (some "indexSelect")

@[test] def testTokenByteTotalsAndEmptyInputs : IO Unit := do
  LeanTest.assertEqual (← mapping.totalBytes #[0, 1, 3, 2, 1]) 10
  LeanTest.assertEqual (← mapping.totalBytesFromTensor (Tensor.fromInt64Array #[3, 2, 1])) 9
  let ids := (Tensor.fromInt64Array #[1, 2, 3, 0]).reshape #[2, 2]
  let mask := (Tensor.fromFloatArray #[1, 0, 1, 1]).reshape #[2, 2]
  LeanTest.assertEqual (← mapping.totalBytesWithMask ids mask) 6
  LeanTest.assertEqual (← mapping.totalBytesWithMask' ids mask) 6
  LeanTest.assertEqual (← mapping.totalBytes #[]) 0
  LeanTest.assertEqual (← mapping.totalBytesFromTensor (Tensor.fromInt64Array #[])) 0
  let emptyIds := (Tensor.fromInt64Array #[]).reshape #[2, 0]
  LeanTest.assertEqual (← mapping.totalBytesWithMask emptyIds (Tensor.zeros #[2, 0])) 0
  let emptyMapping : TokenBytes := { vocabSize := 0, bytes := Tensor.fromInt64Array #[] }
  LeanTest.assertEqual (← emptyMapping.totalBytesFromTensor (Tensor.fromInt64Array #[])) 0

private def checkDevice (device : Device) : IO Unit := do
  let ids ← TensorStruct.castTo (Tensor ⟨#[3], .Int64, device⟩)
    (Tensor.fromInt64Array #[3, 2, 1])
  LeanTest.assertEqual (← mapping.totalBytesFromTensor ids) 9
  let invalid ← TensorStruct.castTo (Tensor ⟨#[1], .Int64, device⟩)
    (Tensor.fromInt64Array #[4])
  Tests.assertThrows (mapping.totalBytesFromTensor invalid) (some "indexSelect")
  let negative ← TensorStruct.castTo (Tensor ⟨#[1], .Int64, device⟩)
    (Tensor.fromInt64Array #[-1])
  Tests.assertThrows (mapping.totalBytesFromTensor negative) (some "indexSelect")
  -- Invalid GPU indexing must leave the device usable for the next operation.
  LeanTest.assertEqual (← mapping.totalBytesFromTensor ids) 9
  let matrixIds ← TensorStruct.castTo (Tensor ⟨#[1, 3], .Int64, device⟩)
    ((Tensor.fromInt64Array #[3, 2, 1]).reshape #[1, 3])
  let mask ← TensorStruct.castTo (Tensor ⟨#[1, 3], .Float32, device⟩)
    (Tensor.ones #[1, 3])
  LeanTest.assertEqual (← mapping.totalBytesWithMask matrixIds mask) 9

@[test] def testTokenByteDevicePreservation : IO Unit := do
  checkDevice .CPU
  if ← mps_is_available then checkDevice (← Device.mps)
  else IO.println "SKIP TokenBytes MPS: unavailable"
  if ← cuda_is_available then checkDevice (← Device.cuda 0)
  else IO.println "SKIP TokenBytes CUDA: unavailable"

end Tests.TokenBytesRecovery
