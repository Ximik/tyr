import LeanTest
import Examples.GPT.NanoGPTCopy

open Examples.GPT.NanoGPTCopy

@[test]
def testNanoGPTCopyPrototypeSmoke : IO Unit := do
  let cfg : PrototypeConfig := {
    PrototypeConfig.tiny with
    trainSteps := 2
    streamTokens := 384
  }
  let stats ← runPrototype cfg
  LeanTest.assertEqual stats.steps cfg.trainSteps
  LeanTest.assertTrue (stats.avgLoss > 0.0) s!"Expected avgLoss > 0, got {stats.avgLoss}"
  LeanTest.assertTrue (stats.finalLoss.isFinite) s!"Expected finite finalLoss, got {stats.finalLoss}"

@[test]
def testNanoGPTCopyBenchmarkStats : IO Unit := do
  let cfg : PrototypeConfig := {
    PrototypeConfig.tiny with
    benchmarkSteps := 3
    streamTokens := 384
  }
  let stats ← runBenchmark cfg
  LeanTest.assertEqual stats.steps cfg.benchmarkSteps
  LeanTest.assertTrue (stats.elapsedNs > 0) s!"Expected positive elapsedNs, got {stats.elapsedNs}"
  LeanTest.assertTrue (stats.elapsedMs > 0.0) s!"Expected positive elapsedMs, got {stats.elapsedMs}"
  LeanTest.assertTrue (stats.avgLoss.isFinite) s!"Expected finite avgLoss, got {stats.avgLoss}"
  LeanTest.assertTrue (stats.finalLoss.isFinite) s!"Expected finite finalLoss, got {stats.finalLoss}"
