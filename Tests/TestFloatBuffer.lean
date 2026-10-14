import LeanTest
import Tyr.Audio.FloatBuffer

namespace Tests.FloatBuffer

@[test]
def testNativeReadsAndExactTailCounts : IO Unit := do
  let values : Array Float := #[10.0, -20.5, 30.25, 40.0, 50.5]
  let buffer ← _root_.FloatBuffer.mkEmpty 0
  let buffer ← buffer.appendArray values
  LeanTest.assertEqual buffer.size values.size
  for i in [:values.size] do
    LeanTest.assertEqual (buffer.uget i) values[i]!
      s!"Native Float return and boxed Nat index must agree at index {i}"
  LeanTest.assertEqual (← buffer.toArray) values
  let buffer ← buffer.keepLast 2
  LeanTest.assertEqual (← buffer.toArray) #[40.0, 50.5]
  let buffer ← buffer.keepLast 1
  LeanTest.assertEqual (← buffer.toArray) #[50.5]
  let buffer ← buffer.keepLast 0
  LeanTest.assertEqual buffer.size 0
  LeanTest.assertEqual (← buffer.toArray) (#[] : Array Float)

@[test]
def testBoxedLargeNatBoundaries : IO Unit := do
  let huge : Nat := 2 ^ 128
  for capacity in #[2 ^ 61, huge] do
    let rejected ← try
      let _ ← _root_.FloatBuffer.mkEmpty capacity
      pure false
    catch error =>
      LeanTest.assertTrue (error.toString.containsSubstr "capacity")
      pure true
    LeanTest.assertTrue rejected "Unaddressable Nat capacities must return an IO error"
  let buffer ← _root_.FloatBuffer.mkEmpty 3
  let buffer ← buffer.appendArray #[1.0, 2.0, 3.0]
  let buffer ← buffer.keepLast huge
  LeanTest.assertEqual (← buffer.toArray) #[1.0, 2.0, 3.0]
  let buffer ← buffer.keepLast 3
  LeanTest.assertEqual (← buffer.toArray) #[1.0, 2.0, 3.0]
  let buffer ← buffer.keepLast 1
  LeanTest.assertEqual (← buffer.toArray) #[3.0]

@[test]
def testNativeMutationAndRMS : IO Unit := do
  let buffer ← _root_.FloatBuffer.mkEmpty 1
  LeanTest.assertEqual (← buffer.rms) 0.0
  let buffer ← buffer.push 3.0
  let suffix ← _root_.FloatBuffer.mkEmpty 1
  let suffix ← suffix.push 4.0
  let buffer ← buffer.append suffix
  LeanTest.assertEqual (← buffer.toArray) #[3.0, 4.0]
  LeanTest.assertTrue (((← buffer.rms) - Float.sqrt 12.5).abs < 1e-12)
  let buffer ← buffer.clear
  LeanTest.assertEqual buffer.size 0
  let buffer ← buffer.push (-2.5)
  LeanTest.assertEqual buffer.size 1
  LeanTest.assertEqual (buffer.uget 0) (-2.5)
  LeanTest.assertEqual (← buffer.toArray) #[-2.5]

end Tests.FloatBuffer
