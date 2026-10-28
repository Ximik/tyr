import Tyr.AD.TensorStructSchema
import Tyr.Checkpoint
import Tyr.Typed.Neural
import Tyr.Model.KittenTTS.TypedOps
import LeanTest

open torch

@[test] def testShapePackedTensorTraversalAndCheckpoint : IO Unit := IO.FS.withTempDir fun dir => do
  let values : Array (ShapePackedTensor .Float32 .CPU) := #[
    (Tensor.fromFloatArray #[1, 2]).packShape,
    ((Tensor.fromFloatArray #[3, 4, 5, 6]).reshape #[2, 2]).packShape ]
  let mapped := FloatingTensorStruct.map (fun t => Tensor.addScalar t 1) values
  let moved ← TensorStruct.mapM (fun t => pure (t.to t.device)) mapped
  let shapes := TensorStruct.fold (fun t acc => acc.push t.runtimeShape) #[] moved
  LeanTest.assertEqual shapes #[#[2], #[2, 2]]
  let summed := FloatingTensorStruct.zipWith Tensor.add values moved
  LeanTest.assertEqual (← summed[0]!.2.toFloatArray) #[3, 5]
  LeanTest.assertEqual (← summed[1]!.2.toFloatArray) #[7, 9, 11, 13]
  checkpoint.saveParams summed dir.toString
  let loaded ← checkpoint.loadParams values dir.toString
  for i in [:loaded.size] do
    LeanTest.assertEqual loaded[i]!.1 values[i]!.1
    LeanTest.assertEqual (← loaded[i]!.2.toFloatArray) (← summed[i]!.2.toFloatArray)

@[test] def testKittenSignedCumsumBoundary : IO Unit := do
  let input := (Tensor.fromFloatArray #[1, 2, 3, 4, 5, 6]).reshape #[2, 3]
  let output := kittentts.cumsumTensor input (-1)
  LeanTest.assertEqual (← output.toFloatArray) #[1, 3, 6, 4, 9, 15]

@[test] def testShapePackedTensorADRebuild : IO Unit := do
  let value : ShapePackedTensor .Float32 .CPU := ((Tensor.ones #[2, 3]).packShape)
  let leaves := TensorStructFlatten.flatten value
  LeanTest.assertEqual leaves.size 1
  let rebuilt ← match TensorStructFlatten.rebuildFrom value leaves with
    | .ok result => pure result
    | .error message => throw (IO.userError message)
  LeanTest.assertEqual rebuilt.1 #[2, 3]
  let wrong : ShapePackedTensor .Float32 .CPU := ((Tensor.ones #[3, 2]).packShape)
  match TensorStructFlatten.rebuildFrom value (TensorStructFlatten.flatten wrong) with
  | .ok _ => LeanTest.fail "AD rebuild must reject a different shape with the same element count"
  | .error _ => pure ()
