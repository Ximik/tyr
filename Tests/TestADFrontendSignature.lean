import Tyr.AD.Frontend.API
import Tyr.Module.Derive
import LeanTest

namespace Tests.ADFrontendSignature

open torch
open LeanTest
open Tyr.AD.Frontend

private def allclose {σ : StaticSpec} (lhs rhs : Tensor σ) : Bool :=
  torch.allclose (s := σ.shape) lhs.raw rhs.raw

structure Model where
  weight : DTensor #[2, 4] .Float32
  bias : DTensor #[2] .Float32
  cache : Frozen { shape := #[2], dtype := .Float32 }
  name : Static String
  deriving TensorStruct, ToTensorStructSchema, TensorStructFlatten

structure InputBundle where
  x : DTensor #[4] .Float32
  scale : Option (DTensor #[1] .Float32)
  flag : Bool
  deriving TensorStruct, ToTensorStructSchema, TensorStructFlatten

structure OutputBundle where
  logits : DTensor #[2] .Float32
  label : Static String
  deriving TensorStruct, ToTensorStructSchema, TensorStructFlatten

private def sampleModel : Model :=
  { weight := Tensor.full #[2, 4] 2.0
    bias := Tensor.full #[2] 3.0
    cache := (Tensor.full #[2] 4.0 : Frozen { shape := #[2], dtype := .Float32 })
    name := "model" }

private def sampleInput : InputBundle :=
  { x := Tensor.full #[4] 5.0
    scale := some (Tensor.full #[1] 6.0)
    flag := true }

private def inputWithoutScale : InputBundle :=
  { x := Tensor.full #[4] 5.0
    scale := none
    flag := true }

private def sampleOutput : OutputBundle :=
  { logits := Tensor.full #[2] 7.0
    label := "out" }

private def replacementOutput : OutputBundle :=
  { logits := Tensor.full #[2] 9.0
    label := "replacement" }

private def replacementModel : Model :=
  { weight := Tensor.full #[2, 4] 8.0
    bias := Tensor.full #[2] 9.0
    cache := (Tensor.full #[2] 10.0 : Frozen { shape := #[2], dtype := .Float32 })
    name := "replacement-model" }

private def replacementInput : InputBundle :=
  { x := Tensor.full #[4] 11.0
    scale := some (Tensor.full #[1] 12.0)
    flag := false }

private def unwrapOrFail {α : Type} (result : Except String α) : IO α :=
  match result with
  | .ok value => pure value
  | .error err => LeanTest.fail err

private def sampleSignature : FrontendADSignature :=
  { params := #[FrontendBoundary.ofValue sampleModel .diffAndFrozen]
    inputs := #[FrontendBoundary.ofValue sampleInput .diffOnly]
    outputs := #[FrontendBoundary.ofValue sampleOutput .diffOnly] }

private def replacementInvarLeaves : Array TensorLeafValue :=
  TensorStructFlatten.flatten replacementModel .diffAndFrozen ++
    TensorStructFlatten.flatten replacementInput .diffOnly

private def fakeStructuredFrontend : StructuredFrontendFunction Model InputBundle OutputBundle :=
  { signature := sampleSignature
    evalFlat := fun invarLeaves => do
      if invarLeaves.size != sampleSignature.paramLeafCount + sampleSignature.inputLeafCount then
        throw <| s!"Expected {sampleSignature.paramLeafCount + sampleSignature.inputLeafCount} flattened invar leaves, " ++
          s!"got {invarLeaves.size}."
      pure {
        outputTemplate := sampleOutput
        outputLeaves := TensorStructFlatten.flatten replacementOutput .diffOnly
      }
    linearizeFlat := fun invarLeaves => do
      if invarLeaves.size != sampleSignature.paramLeafCount + sampleSignature.inputLeafCount then
        throw <| s!"Expected {sampleSignature.paramLeafCount + sampleSignature.inputLeafCount} flattened invar leaves, " ++
          s!"got {invarLeaves.size}."
      pure {
        outputTemplate := sampleOutput
        outputLeaves := TensorStructFlatten.flatten replacementOutput .diffOnly
        pullback := fun outputCotangentLeaves => do
          if outputCotangentLeaves.size != sampleSignature.outputLeafCount then
            throw <| s!"Expected {sampleSignature.outputLeafCount} flattened output cotangent leaves, " ++
              s!"got {outputCotangentLeaves.size}."
          pure replacementInvarLeaves
      } }

private def sampleScalarLoss : DTensor #[] .Float32 :=
  Tensor.full #[] 13.0

private def replacementScalarLoss : DTensor #[] .Float32 :=
  Tensor.full #[] 17.0

private def scalarLossSignature : FrontendADSignature :=
  { params := #[FrontendBoundary.ofValue sampleModel .diffAndFrozen]
    inputs := #[FrontendBoundary.ofValue sampleInput .diffOnly]
    outputs := #[FrontendBoundary.ofValue sampleScalarLoss .diffOnly] }

private def fakeScalarLossFrontend : StructuredFrontendFunction Model InputBundle (DTensor #[] .Float32) :=
  { signature := scalarLossSignature
    evalFlat := fun invarLeaves => do
      if invarLeaves.size != scalarLossSignature.paramLeafCount + scalarLossSignature.inputLeafCount then
        throw <| s!"Expected {scalarLossSignature.paramLeafCount + scalarLossSignature.inputLeafCount} flattened invar leaves, " ++
          s!"got {invarLeaves.size}."
      pure {
        outputTemplate := sampleScalarLoss
        outputLeaves := TensorStructFlatten.flatten replacementScalarLoss .diffOnly
      }
    linearizeFlat := fun invarLeaves => do
      if invarLeaves.size != scalarLossSignature.paramLeafCount + scalarLossSignature.inputLeafCount then
        throw <| s!"Expected {scalarLossSignature.paramLeafCount + scalarLossSignature.inputLeafCount} flattened invar leaves, " ++
          s!"got {invarLeaves.size}."
      pure {
        outputTemplate := sampleScalarLoss
        outputLeaves := TensorStructFlatten.flatten replacementScalarLoss .diffOnly
        pullback := fun outputCotangentLeaves => do
          if outputCotangentLeaves.size != scalarLossSignature.outputLeafCount then
            throw <| s!"Expected {scalarLossSignature.outputLeafCount} flattened output cotangent leaves, " ++
              s!"got {outputCotangentLeaves.size}."
          pure replacementInvarLeaves
      } }

private def assertStructuredParamGrad (grad : Model) : IO Unit := do
  LeanTest.assertTrue (grad.name == sampleModel.name)
    "Structured gradient APIs should preserve static parameter fields from the template."
  LeanTest.assertTrue (allclose grad.weight replacementModel.weight)
    "Structured gradient APIs should rebuild differentiable parameter leaves."
  LeanTest.assertTrue (allclose grad.bias replacementModel.bias)
    "Structured gradient APIs should rebuild all selected parameter leaves."
  LeanTest.assertTrue (allclose grad.cache.get replacementModel.cache.get)
    "Structured gradient APIs should rebuild frozen parameter leaves."

private def assertStructuredInputCotangent (cotangent : InputBundle) : IO Unit := do
  LeanTest.assertTrue (cotangent.flag == sampleInput.flag)
    "Structured gradient APIs should preserve static input fields from the template."
  LeanTest.assertTrue (allclose cotangent.x replacementInput.x)
    "Structured gradient APIs should rebuild differentiable input cotangents."
  match cotangent.scale, replacementInput.scale with
  | some lhs, some rhs =>
      LeanTest.assertTrue (allclose lhs rhs)
        "Structured gradient APIs should rebuild optional differentiable input cotangents."
  | _, _ =>
      LeanTest.fail "Expected optional differentiable input cotangent leaf to remain present after reconstruction."

@[test]
def testFrontendBoundaryMatchesFlattenSelection : IO Unit := do
  let boundary := FrontendBoundary.ofValue sampleModel .diffAndFrozen
  LeanTest.assertEqual boundary.selectedLeafCount 3
    "Model boundary should keep diff and frozen tensor leaves."
  LeanTest.assertEqual boundary.renderedSelectedLeafPaths
    #["weight", "bias", "cache"]
    "Boundary-selected paths should follow declaration order."
  let leaves ← unwrapOrFail <| boundary.flattenValue sampleModel
  LeanTest.assertEqual leaves.size 3
    "Validated flattening should produce one leaf per selected schema entry."
  let expectedRoles : Array TensorLeafRole := #[.diff, .diff, .frozen]
  LeanTest.assertTrue ((leaves.map (·.role)) == expectedRoles)
    s!"Flattened roles should track the selected schema, got {reprStr (leaves.map (·.role))}"
  LeanTest.assertTrue ((sampleSignature.paramBindings.map (·.dtype)) ==
      #[some "Float32", some "Float32", some "Float32"])
    "Frontend bindings must retain the typed and frozen leaf dtypes."

@[test]
def testFrontendBoundaryRejectsWrongDTypeAndInvalidRuntimeShape : IO Unit := do
  let boundary := FrontendBoundary.ofValue sampleModel .diffAndFrozen
  let leaves := TensorStructFlatten.flatten sampleModel
  let wrongDType := leaves.set! 0 (TensorLeafValue.ofTensor .diff (Tensor.fullInt #[2, 4] 1))
  let invalidShape : DTensor #[2, 4] .Float32 := Tensor.assumeSpec (Tensor.zeros #[8]).raw
  let wrongShape := leaves.set! 0 (TensorLeafValue.ofTensor .diff invalidShape)
  for (badLeaves, detail) in #[(wrongDType, "dtype"), (wrongShape, "shape")] do
    match boundary.validateFlattenedLeaves badLeaves with
    | .ok _ => LeanTest.fail s!"Frontend boundary accepted invalid {detail}."
    | .error message =>
        LeanTest.assertTrue (message.containsSubstr "weight" && message.containsSubstr detail)
          s!"Boundary diagnostic should identify field and {detail}: {message}"

@[test]
def testFrontendBoundaryRejectsDynamicLeafMismatch : IO Unit := do
  let boundary := FrontendBoundary.ofValue sampleInput .diffOnly
  match boundary.flattenValue inputWithoutScale with
  | .ok _ =>
      LeanTest.fail "Boundary validation should reject option-driven leaf-count drift."
  | .error err =>
      LeanTest.assertTrue (err.containsSubstr "leaf-count mismatch")
        s!"Mismatch diagnostics should explain the structural drift, got: {err}"

@[test]
def testFrontendSignatureBuildsStableBindingsAndJVars : IO Unit := do
  let sig : FrontendADSignature :=
    { params := #[FrontendBoundary.ofValue sampleModel .diffAndFrozen]
      inputs := #[FrontendBoundary.ofValue sampleInput .diffOnly]
      outputs := #[FrontendBoundary.ofValue sampleOutput .diffOnly] }

  LeanTest.assertEqual sig.paramLeafCount 3
    "Parameter boundary should contribute three selected leaves."
  LeanTest.assertEqual sig.inputLeafCount 2
    "Input boundary should contribute the differentiable tensor leaves."
  LeanTest.assertEqual sig.outputLeafCount 1
    "Output boundary should contribute only differentiable leaves."
  LeanTest.assertEqual sig.renderedInvarPaths
    #["weight", "bias", "cache", "x", "scale"]
    "Signature invar ordering should concatenate params then inputs."
  LeanTest.assertEqual sig.renderedOutputPaths
    #["logits"]
    "Signature output ordering should follow output boundary leaves."

  let invars := sig.invars 10
  LeanTest.assertEqual (invars.map (·.id))
    #[10, 11, 12, 13, 14]
    "Input-side JVar ids should be contiguous from the requested start id."
  let expectedParticipation : Array Tyr.AD.JaxprLike.DiffParticipation := #[
    .diff,
    .diff,
    .frozen,
    .diff,
    .diff
  ]
  LeanTest.assertTrue ((invars.map (·.metaInfo.participation)) == expectedParticipation)
    s!"JVar participation should match binding roles, got {reprStr (invars.map (·.metaInfo.participation))}"
  let expectedShapes : Array (Option (Array Nat)) := #[
    some #[2, 4],
    some #[2],
    some #[2],
    some #[4],
    some #[1]
  ]
  LeanTest.assertTrue ((invars.map (·.metaInfo.shape)) == expectedShapes)
    s!"JVar shape metadata should mirror frontend-selected leaves, got {reprStr (invars.map (·.metaInfo.shape))}"

  let outvars := sig.outvars 30
  LeanTest.assertEqual (outvars.map (·.id)) #[30]
    "Output-side JVar ids should honor the requested start id."
  LeanTest.assertTrue ((outvars.map (·.metaInfo.shape)) == #[some #[2]])
    s!"Output JVar metadata should preserve output tensor shape, got {reprStr (outvars.map (·.metaInfo.shape))}"

@[test]
def testFrontendSignatureRebuildsStructuredOutputFromFlatLeaves : IO Unit := do
  let sig : FrontendADSignature :=
    { outputs := #[FrontendBoundary.ofValue sampleOutput .diffOnly] }
  let rebuilt ← unwrapOrFail <|
    sig.rebuildOutputValue 0 sampleOutput (TensorStructFlatten.flatten replacementOutput .diffOnly)
  LeanTest.assertTrue (rebuilt.label == sampleOutput.label)
    "Structured output reconstruction should preserve static fields from the template."
  LeanTest.assertTrue (allclose rebuilt.logits replacementOutput.logits)
    "Structured output reconstruction should restore the flattened tensor payload."

@[test]
def testFrontendSignatureRebuildsStructuredInvarValues : IO Unit := do
  let sig : FrontendADSignature :=
    { params := #[FrontendBoundary.ofValue sampleModel .diffAndFrozen]
      inputs := #[FrontendBoundary.ofValue sampleInput .diffOnly] }
  let invarLeaves :=
    TensorStructFlatten.flatten replacementModel .diffAndFrozen ++
    TensorStructFlatten.flatten replacementInput .diffOnly

  let rebuiltModel ← unwrapOrFail <|
    sig.rebuildParamValue 0 sampleModel invarLeaves
  LeanTest.assertTrue (rebuiltModel.name == sampleModel.name)
    "Structured param reconstruction should preserve static fields from the template."
  LeanTest.assertTrue (allclose rebuiltModel.weight replacementModel.weight)
    "Structured param reconstruction should restore differentiable tensor fields."
  LeanTest.assertTrue (allclose rebuiltModel.bias replacementModel.bias)
    "Structured param reconstruction should restore all selected differentiable leaves."
  LeanTest.assertTrue (allclose rebuiltModel.cache.get replacementModel.cache.get)
    "Structured param reconstruction should also restore frozen tensor leaves."

  let rebuiltInput ← unwrapOrFail <|
    sig.rebuildInputValue 0 sampleInput invarLeaves
  LeanTest.assertTrue (rebuiltInput.flag == sampleInput.flag)
    "Structured input reconstruction should preserve static fields from the template."
  LeanTest.assertTrue (allclose rebuiltInput.x replacementInput.x)
    "Structured input reconstruction should restore differentiable tensor leaves."
  match rebuiltInput.scale, replacementInput.scale with
  | some lhs, some rhs =>
      LeanTest.assertTrue (allclose lhs rhs)
        "Structured input reconstruction should restore optional differentiable leaves."
  | _, _ =>
      LeanTest.fail "Expected optional differentiable input leaf to remain present after reconstruction."

@[test]
def testFrontendCompanionBuildsStructuredGrad : IO Unit := do
  let sig : FrontendADSignature :=
    { params := #[FrontendBoundary.ofValue sampleModel .diffAndFrozen]
      inputs := #[FrontendBoundary.ofValue sampleInput .diffOnly]
      outputs := #[FrontendBoundary.ofValue sampleOutput .diffOnly] }
  let invarLeaves :=
    TensorStructFlatten.flatten replacementModel .diffAndFrozen ++
    TensorStructFlatten.flatten replacementInput .diffOnly
  let grad ← unwrapOrFail <| sig.rebuildSingleParamGrad sampleModel invarLeaves
  LeanTest.assertTrue (grad.params.name == sampleModel.name)
    "Structured grad reconstruction should preserve static parameter fields from the template."
  LeanTest.assertTrue (allclose grad.params.weight replacementModel.weight)
    "Structured grad reconstruction should rebuild differentiable parameter leaves."
  LeanTest.assertTrue (allclose grad.params.bias replacementModel.bias)
    "Structured grad reconstruction should rebuild all selected parameter leaves."
  LeanTest.assertTrue (allclose grad.params.cache.get replacementModel.cache.get)
    "Structured grad reconstruction should also rebuild frozen parameter leaves."

@[test]
def testFrontendCompanionBuildsStructuredPullback : IO Unit := do
  let sig : FrontendADSignature :=
    { params := #[FrontendBoundary.ofValue sampleModel .diffAndFrozen]
      inputs := #[FrontendBoundary.ofValue sampleInput .diffOnly]
      outputs := #[FrontendBoundary.ofValue sampleOutput .diffOnly] }
  let replacementInvarLeaves :=
    TensorStructFlatten.flatten replacementModel .diffAndFrozen ++
    TensorStructFlatten.flatten replacementInput .diffOnly
  let structured ← unwrapOrFail <|
    sig.buildSingleStructuredPullback
      sampleModel
      sampleInput
      sampleOutput
      (TensorStructFlatten.flatten replacementOutput .diffOnly)
      (fun outputCotangentLeaves => do
        if outputCotangentLeaves.size != 1 then
          throw "Expected one flattened output cotangent leaf."
        pure replacementInvarLeaves)
  LeanTest.assertTrue (structured.output.label == sampleOutput.label)
    "Structured pullback should preserve static output fields from the template."
  LeanTest.assertTrue (allclose structured.output.logits replacementOutput.logits)
    "Structured pullback should rebuild the flat primal output."
  let cotangents ← unwrapOrFail <| structured.pullback replacementOutput
  LeanTest.assertTrue (cotangents.params.name == sampleModel.name)
    "Structured pullback should preserve static parameter fields from the template."
  LeanTest.assertTrue (allclose cotangents.params.weight replacementModel.weight)
    "Structured pullback should rebuild differentiable parameter cotangents."
  LeanTest.assertTrue (allclose cotangents.params.bias replacementModel.bias)
    "Structured pullback should rebuild all parameter cotangent leaves."
  LeanTest.assertTrue (allclose cotangents.params.cache.get replacementModel.cache.get)
    "Structured pullback should rebuild frozen parameter cotangent leaves."
  LeanTest.assertTrue (cotangents.inputs.flag == sampleInput.flag)
    "Structured pullback should preserve static input fields from the template."
  LeanTest.assertTrue (allclose cotangents.inputs.x replacementInput.x)
    "Structured pullback should rebuild differentiable input cotangents."
  match cotangents.inputs.scale, replacementInput.scale with
  | some lhs, some rhs =>
      LeanTest.assertTrue (allclose lhs rhs)
        "Structured pullback should rebuild optional differentiable input cotangents."
  | _, _ =>
      LeanTest.fail "Expected optional differentiable input cotangent leaf to remain present after reconstruction."

@[test]
def testStructuredFrontendFunctionCallRebuildsStructuredOutput : IO Unit := do
  let output ← unwrapOrFail <| fakeStructuredFrontend.call sampleModel sampleInput
  LeanTest.assertTrue (output.label == sampleOutput.label)
    "Structured frontend call should preserve static output fields from the template."
  LeanTest.assertTrue (allclose output.logits replacementOutput.logits)
    "Structured frontend call should rebuild the flat backend output."

@[test]
def testStructuredFrontendFunctionLinearizeAndVJP : IO Unit := do
  let structured ← unwrapOrFail <| fakeStructuredFrontend.linearize sampleModel sampleInput
  LeanTest.assertTrue (structured.output.label == sampleOutput.label)
    "Structured frontend linearize should preserve static output fields from the template."
  LeanTest.assertTrue (allclose structured.output.logits replacementOutput.logits)
    "Structured frontend linearize should rebuild the flat primal output."
  let pullbackCotangents ← unwrapOrFail <| structured.pullback replacementOutput
  let directCotangents ← unwrapOrFail <| fakeStructuredFrontend.vjp sampleModel sampleInput replacementOutput
  assertStructuredParamGrad pullbackCotangents.params
  assertStructuredInputCotangent pullbackCotangents.inputs
  assertStructuredParamGrad directCotangents.params
  assertStructuredInputCotangent directCotangents.inputs

@[test]
def testStructuredFrontendFunctionValueAndGrad : IO Unit := do
  let (output, grad) ← unwrapOrFail <| fakeScalarLossFrontend.valueAndGrad sampleModel sampleInput
  LeanTest.assertTrue (allclose output replacementScalarLoss)
    "Structured frontend value-and-grad should rebuild the scalar primal output."
  assertStructuredParamGrad grad.params
  let gradOnly ← unwrapOrFail <| fakeScalarLossFrontend.grad sampleModel sampleInput
  assertStructuredParamGrad gradOnly.params

@[test]
def testStructuredFrontendFunctionGradRejectsNonScalarOutput : IO Unit := do
  match fakeStructuredFrontend.grad sampleModel sampleInput with
  | .ok _ =>
      LeanTest.fail "Structured frontend grad should reject non-scalar outputs."
  | .error err =>
      LeanTest.assertTrue (err.containsSubstr "scalar output")
        s!"Non-scalar grad diagnostics should explain the scalar requirement, got: {err}"

end Tests.ADFrontendSignature
