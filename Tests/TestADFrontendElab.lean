import Tyr
import LeanTest

namespace Tests.ADFrontendElab

open Lean
open LeanTest
open Tyr.AD.Frontend
open Tyr.AD.JaxprLike

def runCoreM (x : CoreM α) : IO α := do
  Lean.initSearchPath (← Lean.findSysroot)
  let env ← Lean.importModules
    #[{ module := `Tests.TestADFrontendElab }]
    {}
    (loadExts := true)
  let ctx : Core.Context := { fileName := "<test>", fileMap := default }
  let state : Core.State := { env := env }
  let eio := x.run ctx state
  let res ← EIO.toBaseIO eio
  match res with
  | .ok (value, _) => pure value
  | .error msg =>
      throw (IO.userError (← msg.toMessageData.toString))

opaque frontendPadPrimitive (_base : torch.DTensor #[2] .Float32) (_padv : torch.DTensor #[1] .Float32) : torch.DTensor #[6] .Float32 :=
  torch.Tensor.assumeSpec (torch.zeros #[6])

def ordinaryPadWrapper (base : torch.DTensor #[2] .Float32) (padv : torch.DTensor #[1] .Float32) : torch.DTensor #[6] .Float32 :=
  frontendPadPrimitive base padv

opaque frontendReshapePrimitive (x : torch.DTensor #[6] .Float32) : torch.DTensor #[6] .Float32 := x

def unsupportedCompositeWrapper (base : torch.DTensor #[2] .Float32) (padv : torch.DTensor #[1] .Float32) : torch.DTensor #[6] .Float32 :=
  frontendReshapePrimitive (frontendPadPrimitive base padv)

opaque frontendFrozenPadPrimitive (_base : torch.DTensor #[2] .Float32)
    (_padv : torch.Frozen { shape := #[1], dtype := .Float32 }) : torch.DTensor #[6] .Float32 :=
  torch.Tensor.assumeSpec (torch.zeros #[6])

def frozenPadWrapper (base : torch.DTensor #[2] .Float32)
    (padv : torch.Frozen { shape := #[1], dtype := .Float32 }) : torch.DTensor #[6] .Float32 :=
  frontendFrozenPadPrimitive base padv

private def padPrimitiveSpecs : Array PrimitiveFrontendSpec := #[
  {
    leanConst := ``Tests.ADFrontendElab.frontendPadPrimitive
    op := padAliasOpName
    sourceOp := some `Graphax.pad_p
    fixedParams := #[
      OpParam.mkNats .padLow #[1],
      OpParam.mkNats .padHigh #[2],
      OpParam.mkNats .padInterior #[1]
    ]
  },
  {
    leanConst := ``Tests.ADFrontendElab.frontendReshapePrimitive
    op := reshapeAliasOpName
    sourceOp := some `Graphax.reshape_p
  }
]

private def hasEntry (entries : Array Tyr.AD.Sparse.SparseEntry) (src dst : Nat) : Bool :=
  entries.any fun sparseEntry =>
    Tyr.AD.Sparse.SparseEntry.src sparseEntry == src &&
      Tyr.AD.Sparse.SparseEntry.dst sparseEntry == dst

@[test]
unsafe def testDeriveSinglePrimitiveFrontendRegistration : IO Unit := do
  let registration ← runCoreM <|
    deriveSinglePrimitiveFrontendRegistration ``Tests.ADFrontendElab.ordinaryPadWrapper padPrimitiveSpecs
  let jaxpr := registration.jaxpr
  LeanTest.assertEqual jaxpr.invars.size 2
    "Single-primitive frontend derivation should emit one invar per tensor parameter."
  LeanTest.assertEqual jaxpr.eqns.size 1
    "Single-primitive frontend derivation should emit one equation."
  LeanTest.assertEqual jaxpr.outvars.size 1
    "Single-primitive frontend derivation should emit one output var."
  LeanTest.assertTrue ((jaxpr.invars.map (·.metaInfo.shape)) == #[some #[2], some #[1]])
    s!"Derived frontend invar shapes should come from tensor parameter types, got {reprStr (jaxpr.invars.map (·.metaInfo.shape))}"
  LeanTest.assertTrue ((jaxpr.outvars.map (·.metaInfo.shape)) == #[some #[6]])
    s!"Derived frontend output shape should come from the result type, got {reprStr (jaxpr.outvars.map (·.metaInfo.shape))}"
  let eqn := jaxpr.eqns[0]!
  LeanTest.assertEqual eqn.op padAliasOpName
    "Derived frontend equation should use the registered normalized frontend op."
  LeanTest.assertEqual eqn.sourceOpName `Graphax.pad_p
    "Derived frontend equation should preserve the higher-level source primitive."
  LeanTest.assertEqual (eqn.params.findNats? .padLow) (some #[1])
    "Derived frontend equation should preserve fixed primitive params."
  LeanTest.assertEqual (eqn.params.findNats? .padHigh) (some #[2])
    "Derived frontend equation should preserve fixed primitive params."
  LeanTest.assertEqual (eqn.params.findNats? .padInterior) (some #[1])
    "Derived frontend equation should preserve fixed primitive params."

@[test]
unsafe def testDeriveFrontendFrozenTensorSpec : IO Unit := do
  let specs := padPrimitiveSpecs.map fun spec =>
    if spec.leanConst == ``Tests.ADFrontendElab.frontendPadPrimitive then
      { spec with leanConst := ``Tests.ADFrontendElab.frontendFrozenPadPrimitive }
    else spec
  let registration ← runCoreM <|
    deriveSinglePrimitiveFrontendRegistration ``Tests.ADFrontendElab.frozenPadWrapper specs
  LeanTest.assertTrue ((registration.jaxpr.invars.map (·.metaInfo.shape)) == #[some #[2], some #[1]])
    "Frozen parameter shapes must come from their tensor specification."
  LeanTest.assertTrue ((registration.jaxpr.invars.map (·.metaInfo.participation)) == #[.diff, .frozen])
    "Frozen typed parameters must keep their AD participation role."
  LeanTest.assertTrue ((registration.jaxpr.invars.map (·.metaInfo.dtype)) == #[some "Float32", some "Float32"])
    "Typed and frozen parameters must retain their dtype in frontend metadata."

/--
Registration itself currently trips a Lean IR interpreter assertion under
`lean --run`, so this stays as a compile-only smoke helper for now instead of an
executable `@[test]`.
-/
unsafe def registrationSmokeCheck : IO Unit := do
  let (registration, stored) ← runCoreM do
    let registration ←
      registerDerivedSinglePrimitiveFrontend ``Tests.ADFrontendElab.ordinaryPadWrapper padPrimitiveSpecs
    let some stored ← getRegisteredFrontendRegistration? ``Tests.ADFrontendElab.ordinaryPadWrapper
      | throwError "expected derived frontend registration to be stored in the environment"
    pure (registration, stored)
  LeanTest.assertEqual stored.jaxpr.eqns.size 1
    "Stored derived frontend registration should keep the synthesized single-op jaxpr."
  LeanTest.assertEqual stored.jaxpr.eqns[0]!.op padAliasOpName
    "Stored derived frontend registration should keep the normalized frontend op."
  LeanTest.assertEqual stored.jaxpr.eqns[0]!.sourceOpName `Graphax.pad_p
    "Stored derived frontend registration should keep the higher-level source op."
  LeanTest.assertTrue (reprStr stored == reprStr registration)
    "Stored derived frontend registration should match the registration returned by the helper."

@[test]
unsafe def testDeriveSinglePrimitiveFrontendRejectsCompositeBodies : IO Unit := do
  let result ← try
    let _ ← runCoreM <|
      deriveSinglePrimitiveFrontendRegistration ``Tests.ADFrontendElab.unsupportedCompositeWrapper padPrimitiveSpecs
    pure (none : Option String)
  catch e =>
    pure (some e.toString)
  match result with
  | none =>
      LeanTest.fail "Single-primitive frontend derivation should reject composite wrapper bodies."
  | some err =>
      LeanTest.assertTrue (err.containsSubstr "direct parameters")
        s!"Composite-body rejection should explain the supported subset, got: {err}"

end Tests.ADFrontendElab
