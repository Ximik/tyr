import Tyr.Model.BranchingFlows.MoleculeTransformer
import Tyr.Checkpoint
import LeanTest

open torch torch.branching

private def fixture : BranchingBridgeResult MoleculeAtom := {
  t := #[0.25]
  segments := #[#[]]
  Xt := #[BranchingState.mkDefault #[{ coord := { x := 0.2, y := -0.1, z := 0.3 }, label := 0 }] #[0]]
  X1anchor := #[#[{ coord := { x := 0.8, y := 0.2, z := -0.4 }, label := 1 }]]
  descendants := #[#[1]]
  del := #[#[false]]
  splitsTarget := #[#[0]]
  prevCoalescence := #[#[0.0]]
}

@[test] def testBranchingIndexedBatchAndStaticBuffers : IO Unit := do
  let cfg : BranchingTrainConfig := { maxLen := 2 }
  let ⟨_, packed⟩ ← packBranchingMolecule cfg fixture
  LeanTest.assertEqual packed.label.actualSpec.dtype .Int64
  LeanTest.assertEqual packed.padmask.actualSpec.dtype .Int64
  LeanTest.assertEqual packed.coord.actualSpec.dtype .Float32
  let buffer ← BranchingMoleculeBatch.zeros 1 2 .CPU true
  let copied ← buffer.copyInto packed
  LeanTest.assertEqual copied.coord.actualSpec.dtype .BFloat16
  LeanTest.assertEqual copied.t.actualSpec.dtype .BFloat16
  LeanTest.assertEqual copied.label.actualSpec.dtype .Int64
  LeanTest.assertEqual copied.padmask.actualSpec.dtype .Int64
  LeanTest.assertEqual (← copied.label.toUInt64Array) #[0, 0]
  LeanTest.assertEqual (← copied.labelAnchor.toUInt64Array) #[1, 0]
  LeanTest.assertEqual (← copied.padmask.toUInt64Array) #[1, 0]
  LeanTest.assertEqual (← copied.coord.toFloatArray) (← packed.coord.toBFloat16.toFloatArray)
  let rejected ← try
    let _ ← castTreeBFloat16 (Tensor.ones #[2])
    pure false
  catch _ => pure true
  LeanTest.assertTrue rejected "A fixed Float32 leaf must not silently become BF16"

@[test] def testBranchingStaticCopyRejectsImplicitCastAndBroadcast : IO Unit := do
  let dst ← data.cloneInferenceIO (Tensor.fullInt #[2] 7)
  let fractional := Tensor.fromFloatArray #[1.75, 2.25]
  let dtypeError ← try
    let _ ← copyIntoChecked? dst.forgetDType fractional.forgetDType
    pure none
  catch error => pure (some error.toString)
  LeanTest.assertTrue (dtypeError.any (·.containsSubstr "dtype mismatch"))
    "Static buffer copy must reject an implicit Float32-to-Int64 cast"
  LeanTest.assertEqual (← dst.toUInt64Array) #[7, 7]
  let shapeError ← try
    let _ ← copyIntoChecked? dst (Tensor.fullInt #[1] 3)
    pure none
  catch error => pure (some error.toString)
  LeanTest.assertTrue (shapeError.any (·.containsSubstr "shape mismatch"))
    "Static buffer copy must reject native copy_ broadcasting"
  LeanTest.assertEqual (← dst.toUInt64Array) #[7, 7]
  let copied ← copyIntoChecked? dst (Tensor.castLike dst fractional)
  LeanTest.assertEqual (← copied.toUInt64Array) #[1, 2]

@[test] def testBranchingMuonPreservesBF16Rounding : IO Unit := do
  let param := (Tensor.fromFloatArray #[1, 2, 3, 4, 5, 6]).reshapeChecked #[3, 2]
  let grad := (Tensor.fromFloatArray #[0.3, -0.2, 0.7, 0.4, -0.5, 0.9]).reshapeChecked #[3, 2]
  let state := initMoleculeMuonState param
  let momentum := grad * (0.05 : Float)
  let nesterov := grad * (0.05 : Float) + momentum * (0.95 : Float)
  let native ← Optim.PolarExpress.muonOrthogonalize nesterov 5
  let aspect := Float.sqrt 1.5
  let lr := 0.0137
  let scalarDirection := (Tensor.Preserving.mulScalar native (lr * aspect)).toFloat32
  let (scalarResult, _) ← moleculeMuonStep param grad state lr 0.0 0.95 5
  LeanTest.assertEqual (← scalarResult.toFloatArray) (← (param - scalarDirection).toFloatArray)
  let lrTensor := (Tensor.fromFloatArray #[lr]).reshapeChecked #[]
  let tensorDirection := (mulTensorScalar (Tensor.Preserving.mulScalar native aspect) lrTensor).toFloat32
  let (tensorResult, _) ← moleculeMuonStepT param grad state lrTensor.forgetDType 0.0 0.95 5
  LeanTest.assertEqual (← tensorResult.toFloatArray) (← (param - tensorDirection).toFloatArray)
  LeanTest.assertEqual scalarResult.actualSpec.dtype .Float32

@[test] def testBranchingModelConfigMismatchIsRejected : IO Unit := do
  let cfg : BranchingTrainConfig := { maxLen := 2 }
  let model : BranchingMoleculeModel 1 2 (Tensor { shape := #[], dtype := .Float32 }) := {
    forward := fun _ _ _ _ _ => throw <| IO.userError "model should not run"
  }
  let rejected ← try
    let param := Tensor.zeros #[]
    let _ ← trainStepMolecule cfg model param ((Optim.adamw (α := Tensor { shape := #[], dtype := .Float32 })).init param) fixture 0.01
    pure false
  catch error => pure (error.toString.containsSubstr "maxLen mismatch")
  LeanTest.assertTrue rejected "Mismatched model/config sequence lengths must fail before forward"

@[noinline, nospecialize]
private def roundTrip {α : Type} [TensorStruct α] (params : α) (dir : String) : IO α := do
  checkpoint.saveParams params dir
  checkpoint.loadParams params dir

@[test] def testBranchingMoleculeTypedTrainingAndCheckpoint : IO Unit := IO.FS.withTempDir fun dir => do
  let cfg : BranchingTrainConfig := { maxLen := 2, gradClip := 1.0 }
  for bf16 in #[false, true] do
    let initial ← MoleculeTransformerParams.init 2 1 4 4
    let params ← if bf16 then castTreeBFloat16 initial else pure initial
    let model := moleculeTransformerModel (maxLen := 2) (vocab := 2) (heads := 1) (headDim := 4) (mlp := 4)
    let (trained, state, report) ← trainStepMoleculeMuon cfg model params
      (initMoleculeMuonState params) fixture 0.01 (bf16 := bf16)
    LeanTest.assertEqual state.step 1
    LeanTest.assertTrue report.total.isFinite
    LeanTest.assertEqual trained.coordW.actualSpec.dtype (if bf16 then .BFloat16 else .Float32)
    LeanTest.assertTrue (Tensor.item (Tensor.sumAll (Tensor.abs (Tensor.Promoting.sub trained.coordW params.coordW))) > 0)
    let prediction ← moleculeTransformerPrediction (maxLen := 2) (vocab := 2) 0 trained 0.25 fixture.Xt[0]!
    LeanTest.assertEqual prediction.coordTargets.size 1
    let loaded ← roundTrip trained dir.toString
    let ⟨_, packed0⟩ ← packBranchingMolecule cfg fixture
    let packed := if bf16 then packed0.castBFloat16 else packed0
    let (a, b, c, d) ← model.forward trained packed.coord packed.label packed.t packed.padmask
    let (a', b', c', d') ← model.forward loaded packed.coord packed.label packed.t packed.padmask
    LeanTest.assertEqual (← a.toFloatArray) (← a'.toFloatArray)
    LeanTest.assertEqual (← b.toFloatArray) (← b'.toFloatArray)
    LeanTest.assertEqual (← c.toFloatArray) (← c'.toFloatArray)
    LeanTest.assertEqual (← d.toFloatArray) (← d'.toFloatArray)

@[test] def testBranchingFullTransformerRotaryAndBF16 : IO Unit := do
  for bf16 in #[false, true] do
    let initial ← FullMoleculeTransformerParams.init 2 4 1 4 4 2 2
    let params ← if bf16 then castTreeBFloat16 initial else pure initial
    let ⟨_, packed0⟩ ← packBranchingMolecule { maxLen := 2 } fixture
    let packed := if bf16 then packed0.castBFloat16 else packed0
    let (a, _, _, _) ← params.forward packed.coord packed.label packed.t packed.padmask
    let (cos, sin) := Tensor.rotaryFreqs 2 4
    let (b, _, _, _) ← params.forwardFreqs packed.coord packed.label packed.t packed.padmask cos.forgetDType sin.forgetDType
    LeanTest.assertEqual (← a.toFloatArray) (← b.toFloatArray)
    LeanTest.assertEqual a.actualSpec.dtype (if bf16 then .BFloat16 else .Float32)
    for value in (← a.toFloatArray) do LeanTest.assertTrue value.isFinite
    let prediction ← fullMoleculeTransformerPrediction (maxLen := 2) (vocab := 2) 0 params 0.25 fixture.Xt[0]!
    LeanTest.assertEqual prediction.coordTargets.size 1
