import Tyr.AD.Frontend.Companion
import Tyr.Typed.Ops

/-!
# Tyr.AD.Frontend.API

Backend-agnostic structured frontend AD APIs over flat leaf callbacks.

This is the first user-facing layer above `FrontendADSignature`: a backend can
evaluate and linearize in flat leaf space, and this module rebuilds ordinary
Lean values for `call`, `linearize`, `vjp`, and scalar-loss `grad`.
-/

namespace Tyr.AD.Frontend

open torch

/-- Flat frontend primal result together with a structured output template. -/
structure FlatFrontendEvalResult (Outputs : Type) where
  outputTemplate : Outputs
  outputLeaves : Array TensorLeafValue

/-- Flat frontend linearization result with a pullback over flat leaf buffers. -/
structure FlatFrontendLinearizedResult (Outputs : Type) extends FlatFrontendEvalResult Outputs where
  pullback : Array TensorLeafValue → Except String (Array TensorLeafValue)

/--
Structured frontend companion bundle.

This initial API is intentionally single-boundary at the Lean surface:
`Params`, `Inputs`, and `Outputs` are each one structured value. Frontends that
need multiple user-facing values can package them in tuples/structures.
-/
structure StructuredFrontendFunction (Params Inputs Outputs : Type) where
  signature : FrontendADSignature
  evalFlat : Array TensorLeafValue → Except String (FlatFrontendEvalResult Outputs)
  linearizeFlat : Array TensorLeafValue → Except String (FlatFrontendLinearizedResult Outputs)

namespace StructuredFrontendFunction

private def requireSingleBoundary (label : String) (count : Nat) : Except String Unit := do
  if count != 1 then
    throw <| s!"Structured frontend API expects exactly one {label} boundary, " ++
      s!"signature has {count}."

private def flattenSingleBoundaryValue
    {α : Type}
    [ToTensorStructSchema α]
    [TensorStructFlatten α]
    (label : String)
    (boundaries : Array FrontendBoundary)
    (x : α) :
    Except String (Array TensorLeafValue) := do
  requireSingleBoundary label boundaries.size
  let some boundary := boundaries[0]? | throw <|
    s!"Internal structured frontend {label} boundary lookup failure."
  boundary.flattenValue x

private def flattenSingleInvars
    {Params Inputs : Type}
    [ToTensorStructSchema Params]
    [TensorStructFlatten Params]
    [ToTensorStructSchema Inputs]
    [TensorStructFlatten Inputs]
    (sig : FrontendADSignature)
    (params : Params)
    (inputs : Inputs) :
    Except String (Array TensorLeafValue) := do
  let paramLeaves ← flattenSingleBoundaryValue "param" sig.params params
  let inputLeaves ← flattenSingleBoundaryValue "input" sig.inputs inputs
  pure (paramLeaves ++ inputLeaves)

private def buildScalarUnitCotangent
    {Outputs : Type}
    [ToTensorStructSchema Outputs]
    [TensorStructFlatten Outputs]
    (sig : FrontendADSignature)
    (outputTemplate : Outputs)
    (flatOutputLeaves : Array TensorLeafValue) :
    Except String Outputs := do
  requireSingleBoundary "output" sig.outputs.size
  let some boundary := sig.outputs[0]? | throw "Internal structured output boundary lookup failure."
  boundary.validateValue outputTemplate
  boundary.validateFlattenedLeaves flatOutputLeaves
  if flatOutputLeaves.size != 1 then
    throw <| s!"Structured scalar grad expects exactly one selected output leaf, got {flatOutputLeaves.size}."
  let some outputLeaf := flatOutputLeaves[0]? | throw <|
    "Internal structured scalar grad error: missing single output leaf after count validation."
  if outputLeaf.role != .diff then
    throw <| s!"Structured scalar grad requires a differentiable output leaf, got {reprStr outputLeaf.role}."
  if outputLeaf.shape != #[] then
    throw <| s!"Structured scalar grad requires scalar output shape [], got {reprStr outputLeaf.shape}."
  let unitLeaf :=
    match outputLeaf.payload with
    | ⟨_, t⟩ => TensorLeafValue.ofTensor .diff (Tensor.onesLike t)
  boundary.rebuildValue outputTemplate #[unitLeaf]

/-- Execute the frontend and rebuild the structured primal output. -/
def call
    {Params Inputs Outputs : Type}
    [ToTensorStructSchema Params]
    [TensorStructFlatten Params]
    [ToTensorStructSchema Inputs]
    [TensorStructFlatten Inputs]
    [ToTensorStructSchema Outputs]
    [TensorStructFlatten Outputs]
    (fn : StructuredFrontendFunction Params Inputs Outputs)
    (params : Params)
    (inputs : Inputs) :
    Except String Outputs := do
  requireSingleBoundary "output" fn.signature.outputs.size
  let invarLeaves ← flattenSingleInvars fn.signature params inputs
  let flatResult ← fn.evalFlat invarLeaves
  fn.signature.rebuildOutputValue 0 flatResult.outputTemplate flatResult.outputLeaves

/-- Execute the frontend and return a structured pullback. -/
def linearize
    {Params Inputs Outputs : Type}
    [ToTensorStructSchema Params]
    [TensorStructFlatten Params]
    [ToTensorStructSchema Inputs]
    [TensorStructFlatten Inputs]
    [ToTensorStructSchema Outputs]
    [TensorStructFlatten Outputs]
    (fn : StructuredFrontendFunction Params Inputs Outputs)
    (params : Params)
    (inputs : Inputs) :
    Except String (StructuredPullback Params Inputs Outputs) := do
  let invarLeaves ← flattenSingleInvars fn.signature params inputs
  let flatResult ← fn.linearizeFlat invarLeaves
  fn.signature.buildSingleStructuredPullback
    params
    inputs
    flatResult.outputTemplate
    flatResult.outputLeaves
    flatResult.pullback

/-- Execute a structured VJP against an explicit structured output cotangent. -/
def vjp
    {Params Inputs Outputs : Type}
    [ToTensorStructSchema Params]
    [TensorStructFlatten Params]
    [ToTensorStructSchema Inputs]
    [TensorStructFlatten Inputs]
    [ToTensorStructSchema Outputs]
    [TensorStructFlatten Outputs]
    (fn : StructuredFrontendFunction Params Inputs Outputs)
    (params : Params)
    (inputs : Inputs)
    (outputCotangent : Outputs) :
    Except String (StructuredVJPResult Params Inputs) := do
  let structured ← fn.linearize params inputs
  structured.pullback outputCotangent

/--
Execute a scalar-loss style structured gradient, returning both the primal
output and the structured parameter gradient.
-/
def valueAndGrad
    {Params Inputs Outputs : Type}
    [ToTensorStructSchema Params]
    [TensorStructFlatten Params]
    [ToTensorStructSchema Inputs]
    [TensorStructFlatten Inputs]
    [ToTensorStructSchema Outputs]
    [TensorStructFlatten Outputs]
    (fn : StructuredFrontendFunction Params Inputs Outputs)
    (params : Params)
    (inputs : Inputs) :
    Except String (Outputs × StructuredGradResult Params) := do
  let invarLeaves ← flattenSingleInvars fn.signature params inputs
  let flatResult ← fn.linearizeFlat invarLeaves
  let structured ← fn.signature.buildSingleStructuredPullback
    params
    inputs
    flatResult.outputTemplate
    flatResult.outputLeaves
    flatResult.pullback
  let outputCotangent ← buildScalarUnitCotangent
    fn.signature
    flatResult.outputTemplate
    flatResult.outputLeaves
  let vjpResult ← structured.pullback outputCotangent
  pure (structured.output, { params := vjpResult.params })

/-- Structured parameter gradient for single-scalar-output frontends. -/
def grad
    {Params Inputs Outputs : Type}
    [ToTensorStructSchema Params]
    [TensorStructFlatten Params]
    [ToTensorStructSchema Inputs]
    [TensorStructFlatten Inputs]
    [ToTensorStructSchema Outputs]
    [TensorStructFlatten Outputs]
    (fn : StructuredFrontendFunction Params Inputs Outputs)
    (params : Params)
    (inputs : Inputs) :
    Except String (StructuredGradResult Params) := do
  let (_, gradResult) ← fn.valueAndGrad params inputs
  pure gradResult

end StructuredFrontendFunction

end Tyr.AD.Frontend
