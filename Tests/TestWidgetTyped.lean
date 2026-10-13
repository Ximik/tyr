import LeanTest
import Tyr.Widget

namespace Tests.WidgetTyped

open torch

private structure DisplayModel where
  weight : Tensor ⟨#[2, 3], .Float64, .exact .CPU⟩
  tokens : Option (DTensor #[2] .Int64)
  deriving ToModuleDisplay

@[test]
def testIndexedTensorWidgetMetadataAndGradients : IO Unit := do
  let weight : Tensor ⟨#[2, 3], .Float64, .exact .CPU⟩ :=
    (Tensor.onesLike (Tensor.zerosFor _)).setRequiresGrad true
  let loss := Tensor.sumAll (Tensor.Preserving.mulScalar weight 3.0)
  Tensor.backwardLoss loss
  let props := tensorToPropsNamed weight "weight" #["out", "in"]
  LeanTest.assertEqual props.shape #[2, 3] "Widget shape"
  LeanTest.assertEqual props.dtype "Float64" "Widget dtype"
  LeanTest.assertEqual props.device "cpu" "Widget device"
  LeanTest.assertEqual props.numel 6 "Widget element count"
  LeanTest.assertEqual props.values (Array.replicate 6 1.0) "Widget values"
  LeanTest.assertEqual props.gradValues (some (Array.replicate 6 3.0)) "Widget gradients"
  LeanTest.assertTrue (props.requiresGrad && props.gradStats.isSome) "Gradient metadata"
  LeanTest.assertEqual props.name (some "weight") "Widget name"
  LeanTest.assertEqual props.axisNames (some #["out", "in"]) "Widget axes"
  let model : DisplayModel := { weight, tokens := some (Tensor.zerosFor _) }
  match (toModuleDisplayProps model "model").root with
  | .group name children =>
    LeanTest.assertEqual name "model" "Derived model name"
    LeanTest.assertEqual children.size 2 "Derived field count"
    match (children[1]? : Option (String × ModuleNode)) with
    | Option.some (_, ModuleNode.tensor tokens) =>
      LeanTest.assertEqual tokens.dtype "Int64" "Optional typed field dtype"
      LeanTest.assertTrue (!tokens.requiresGrad && tokens.gradValues.isNone)
        "Integer tensors have no gradient display"
    | _ => LeanTest.fail "Expected an optional tensor display"
  | _ => LeanTest.fail "Expected a derived module group"

end Tests.WidgetTyped
