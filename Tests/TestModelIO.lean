/-
  Tests/TestModelIO.lean

  Tests for Model Parameter serialization and structure traversal using TensorStruct.
-/
import Tests.Assertions
import Tyr.Typed.Neural
import Tyr.Checkpoint
import Examples.NanoChat.ModdedGPT
import LeanTest

open torch
open torch.moddedGpt

/-- Helper to flatten model to a list of tensors -/
def flattenModel [TensorStruct α] (model : α) : List (Sigma Tensor) :=
  TensorStruct.fold (fun {s} t acc => (⟨s, t⟩ : Sigma Tensor) :: acc) [] model |>.reverse

/-- Helper to count elements in flattened model -/
def countElements (flat : List (Sigma Tensor)) : Nat :=
  flat.foldl (fun acc ⟨s, _⟩ => acc + s.shape.foldl (fun p d => p * d.toNat) 1) 0

@[test]
def testModdedGPTStructure : IO Unit := do
  -- Create tiny config
  let cfg : moddedGpt.Config := {
    vocabSize := 128
    nLayer := 2
    nHead := 2
    headDim := 16
    modelDim := 32
    maxSeqLen := 64
    blockSize := 32
    numValueEmbeds := 1
  }

  -- Init model
  let params ← ModdedGPTParams.init cfg

  -- Flatten
  let flat := flattenModel params
  
  -- Verify count
  -- 1. embed: 1
  -- 2. smearGate.weight: 1
  -- 3. valueEmbeds[0]: 1
  -- blocks (2 layers, each has attn + mlp):
  --   4-7. blocks[0].attn (wQ, wK, wV, wO): 4
  --   8-9. blocks[0].mlp (cFc, cProj): 2
  --   10-13. blocks[1].attn: 4
  --   14-15. blocks[1].mlp: 2
  -- 16. lmHead.weight: 1
  -- 17. scalars.values: 1
  -- Total: 1 + 1 + 1 + 6 + 6 + 1 + 1 = 17
  let numTensors := flat.length
  LeanTest.assertEqual numTensors 17 "Should have 17 tensors in the model structure"

  -- Verify total elements
  let totalElems := countElements flat
  LeanTest.assertTrue (totalElems > 0) "Total elements should be positive"

@[test]
def testParameterUpdate : IO Unit := do
  -- Verify we can map over parameters to update them (simulating optimizer step or loading)
  let cfg : moddedGpt.Config := {
    vocabSize := 10
    nLayer := 1
    nHead := 1
    headDim := 4
    modelDim := 4
    maxSeqLen := 10
    blockSize := 10
    numValueEmbeds := 1
  }
  
  let params ← ModdedGPTParams.init cfg
  
  -- Apply update: p = p + 1
  let updatedParams := FloatingTensorStruct.map (fun t => Tensor.addScalar t 1.0) params
  
  -- Check one tensor to verify update
  let embedSumOriginal := Tensor.item (Tensor.sumAll params.embed)
  let embedSumUpdated := Tensor.item (Tensor.sumAll updatedParams.embed)
  
  let diff := embedSumUpdated - embedSumOriginal
  let expectedDiff := (cfg.vocabSize * cfg.modelDim).toFloat -- adding 1.0 to each element
  
  LeanTest.assertTrue (Float.abs (diff - expectedDiff) < 1e-4) "Parameters should be updated"

@[test]
def testStateDictSimulation : IO Unit := do
  -- Simulate saving to a state dict (list of tensors) and loading back
  let cfg : moddedGpt.Config := {
    vocabSize := 10
    nLayer := 1
    nHead := 1
    headDim := 4
    modelDim := 4
    maxSeqLen := 10
    blockSize := 10
    numValueEmbeds := 1
  }
  
  let params ← ModdedGPTParams.init cfg
  
  -- "Save" to list
  let stateList := flattenModel params
  
  -- "Load" from list (Mock implementation of loading logic)
  -- We use mapM with a stateful iterator to consume the list
  let loadFromList (list : List (Sigma Tensor)) (model : ModdedGPTParams cfg)
      : IO (ModdedGPTParams cfg) := do
    let ref ← IO.mkRef list
    TensorStruct.mapM (fun {σ} _oldT => do
      let remaining ← ref.get
      match remaining with
      | [] => throw $ IO.userError "Not enough tensors in state list"
      | ⟨_s', t⟩ :: rest => 
        let tensor ← match Tensor.ofRaw (σ := σ) t.raw with
          | .ok tensor => pure tensor
          | .error message => throw (IO.userError message)
        ref.set rest
        pure tensor
    ) model

  let loadedParams ← loadFromList stateList params
  let malformed : Sigma Tensor := ⟨_, Tensor.zeros #[1]⟩
  Tests.assertThrows (loadFromList (malformed :: stateList.drop 1) params) (some "shape")
  
  -- Verify loaded matches original
  let diff := FloatingTensorStruct.zipWith Tensor.sub params loadedParams
  let totalDiff := TensorStruct.fold (fun {_s} t acc => acc + Tensor.item (Tensor.sumAll (Tensor.abs t))) 0.0 diff
  
  LeanTest.assertTrue (totalDiff < 1e-6) "Loaded parameters should match saved ones"
