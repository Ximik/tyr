import Tests.Assertions
import LeanTest
import Tyr.Torch

namespace Tests.TorchRecovery

open torch

@[test] def testResamplerOutputSizeErrorsAreRecoverable : IO Unit := do
  -- The first input used to throw std::length_error across the IO FFI boundary.
  Tests.assertThrows (data.resampleSoxrHQ #[0.25] 1 9223372036854775808)
    (some "resample")
  Tests.assertThrows (data.resampleSoxrHQ #[0.25] 1 18446744073709551615)
    (some "resample")
  Tests.assertThrows (data.resampleSoxrHQ #[0.25, 0.5] 1 9223372036854775808)
    (some "resample")

@[test] def testResamplerRateErrorsAreRecoverable : IO Unit := do
  Tests.assertThrows (data.resampleSoxrHQ #[0.25] 0 24000)
  Tests.assertThrows (data.resampleSoxrHQ #[0.25] 24000 0)
  Tests.assertThrows (data.resampleSoxrHQ #[] 0 0)

@[test] def testResamplerValidAndBorrowedInputs : IO Unit := do
  let samples : Array Float := #[0.25, -0.5, 0.75, 0.0]
  let same ← data.resampleSoxrHQ samples 24000 24000
  LeanTest.assertEqual same samples
  LeanTest.assertEqual samples #[0.25, -0.5, 0.75, 0.0]
  LeanTest.assertEqual (← data.resampleSoxrHQ #[] 48000 24000) #[]
  let input := Array.replicate 16 (0.25 : Float)
  let output ← data.resampleSoxrHQ input 48000 24000
  LeanTest.assertEqual output.size 8
  LeanTest.assertTrue (output.all fun x => !x.isNaN && !x.isInf)
  LeanTest.assertEqual input (Array.replicate 16 (0.25 : Float))
  let fractional ← data.resampleSoxrHQ #[0.25, 0.25, 0.25] 2 3
  LeanTest.assertEqual fractional.size 5
  LeanTest.assertTrue (fractional.all fun x => !x.isNaN && !x.isInf)

end Tests.TorchRecovery
