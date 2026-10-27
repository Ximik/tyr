import LeanTest
import Tyr.Torch

namespace Tests.TestResampleOwnership

/-- Build fresh boxed floats rather than relying on immortal literal arrays.
The input array has no owner left when this function returns. -/
@[noinline] private def freshPassthrough : IO (Float × Array Float) := do
  let tick ← IO.monoNanosNow
  let value := (tick % 100).toFloat
  let output ← torch.data.resampleSoxrHQ #[value, value + 0.5, value + 1] 16000 16000
  pure (value, output)

@[test]
def testResamplePassthroughOwnsReturnedSamples : IO Unit := do
  for _ in [:10] do
    let (value, output) ← freshPassthrough
    -- Reuse recently freed Float allocations before observing the result.
    let allocations := (Array.range 1000).map fun n => n.toFloat * 1.25
    LeanTest.assertEqual allocations[999]! 1248.75
    LeanTest.assertEqual output #[value, value + 0.5, value + 1]

@[test]
def testResamplePassthroughPreservesArrayAliases : IO Unit := do
  let tick ← IO.monoNanosNow
  let value := (tick % 100).toFloat
  let source := #[value, value + 0.5]
  let output ← torch.data.resampleSoxrHQ source 24000 24000
  let changed := output.set! 0 (value + 9)
  LeanTest.assertEqual changed #[value + 9, value + 0.5]
  LeanTest.assertEqual source #[value, value + 0.5]
  LeanTest.assertEqual (← torch.data.resampleSoxrHQ #[] 16000 16000) #[]
  LeanTest.assertEqual (← torch.data.resampleSoxrHQ #[] 16000 24000) #[]

end Tests.TestResampleOwnership
