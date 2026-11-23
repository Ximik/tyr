import LeanTest
import Examples.GPT.RiemannianNanoGPT

open Examples.GPT.RiemannianNanoGPT

@[test]
def testRiemannianNanoGPTPrototypeSmoke : IO Unit := do
  let cfg : PrototypeConfig := {
    PrototypeConfig.tiny with
    trainSteps := 1
    streamTokens := 128
  }
  let stats ← runPrototype cfg
  LeanTest.assertEqual stats.steps cfg.trainSteps
  LeanTest.assertTrue (stats.avgLoss > 0.0) s!"Expected avgLoss > 0, got {stats.avgLoss}"
  LeanTest.assertTrue (stats.finalLoss.isFinite) s!"Expected finite finalLoss, got {stats.finalLoss}"
  LeanTest.assertTrue (stats.finalDiagnostics.factorRank > 0)
    s!"Expected positive factorRank, got {stats.finalDiagnostics.factorRank}"

@[test]
def testRiemannianNanoGPTBenchmarkStats : IO Unit := do
  let cfg : PrototypeConfig := {
    PrototypeConfig.tiny with
    benchmarkSteps := 1
    streamTokens := 128
  }
  let stats ← runBenchmark cfg
  LeanTest.assertEqual stats.steps cfg.benchmarkSteps
  LeanTest.assertTrue (stats.elapsedNs > 0) s!"Expected positive elapsedNs, got {stats.elapsedNs}"
  LeanTest.assertTrue (stats.elapsedMs > 0.0) s!"Expected positive elapsedMs, got {stats.elapsedMs}"
  LeanTest.assertTrue (stats.avgLoss.isFinite) s!"Expected finite avgLoss, got {stats.avgLoss}"
  LeanTest.assertTrue (stats.finalLoss.isFinite) s!"Expected finite finalLoss, got {stats.finalLoss}"
  LeanTest.assertTrue (stats.finalDiagnostics.gradientNorm >= 0.0)
    s!"Expected non-negative gradientNorm, got {stats.finalDiagnostics.gradientNorm}"

@[test]
def testRiemannianNanoGPTSampledFisherPrototype : IO Unit := do
  let cfg : PrototypeConfig := {
    PrototypeConfig.tiny with
    trainSteps := 1
    streamTokens := 128
    pullbackMetric := {
      mode := .sampledFisher
      fisherProbeCount := 4
    }
  }
  let stats ← runPrototype cfg
  LeanTest.assertEqual stats.steps cfg.trainSteps
  LeanTest.assertTrue (stats.finalLoss.isFinite) s!"Expected finite finalLoss, got {stats.finalLoss}"
  LeanTest.assertEqual stats.finalDiagnostics.factorRank 4
