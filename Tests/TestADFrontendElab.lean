import Tyr
import LeanTest
import Std.Sync.Mutex

namespace Tests.ADFrontendElab

open Lean
open LeanTest
open Tyr.AD.Frontend
open Tyr.AD.JaxprLike

-- Imported compacted regions outlive an Environment value. Retain one immutable
-- base rather than importing the entire module graph again for every test.
-- Keep the regions alive because returned frontend expressions may refer to them.
private initialize importedEnvironment : Std.Mutex (Option Environment) ← Std.Mutex.new none

private def getImportedEnvironment : IO Environment :=
  importedEnvironment.atomically do
    if let some env := ← get then
      return env
    Lean.initSearchPath (← Lean.findSysroot)
    let env ← Lean.importModules
      #[{ module := `Tests.TestADFrontendElab }] {} (loadExts := true)
    set (some env)
    return env

def runCoreM (x : CoreM α) : IO α := do
  let env ← getImportedEnvironment
  -- CoreM.run uses this baseline; the default zero charges earlier tests too.
  let initHeartbeats ← IO.getNumHeartbeats
  let ctx : Core.Context := { fileName := "<test>", fileMap := default, initHeartbeats }
  let state : Core.State := { env := env }
  let eio := x.run ctx state
  let res ← EIO.toBaseIO eio
  match res with
  | .ok (value, _) => pure value
  | .error msg =>
      throw (IO.userError (← msg.toMessageData.toString))

@[test]
unsafe def testImportedEnvironmentReusePreservesTestIsolation : IO Unit := do
  let original ← runCoreM do
    let env ← getEnv
    setEnv (env.setMainModule `testOnlyEnvironmentMutation)
    return env
  let next ← runCoreM getEnv
  assertTrue (ptrEq original next) "Repeated tests must share their immutable imported base"
  assertEqual original.mainModule next.mainModule
    "A test's environment updates must not enter the shared base"

opaque frontendPadPrimitive (_base : torch.Tensor { shape := #[2], dtype := .Float32, device := .CPU }) (_padv : torch.Tensor { shape := #[1], dtype := .Float32, device := .CPU }) : torch.Tensor { shape := #[6], dtype := .Float32, device := .CPU } :=
  torch.Tensor.zeros #[6]

def ordinaryPadWrapper (base : torch.Tensor { shape := #[2], dtype := .Float32, device := .CPU }) (padv : torch.Tensor { shape := #[1], dtype := .Float32, device := .CPU }) : torch.Tensor { shape := #[6], dtype := .Float32, device := .CPU } :=
  frontendPadPrimitive base padv

opaque frontendReshapePrimitive (x : torch.Tensor { shape := #[6], dtype := .Float32, device := .CPU }) : torch.Tensor { shape := #[6], dtype := .Float32, device := .CPU } := x

def unsupportedCompositeWrapper (base : torch.Tensor { shape := #[2], dtype := .Float32, device := .CPU }) (padv : torch.Tensor { shape := #[1], dtype := .Float32, device := .CPU }) : torch.Tensor { shape := #[6], dtype := .Float32, device := .CPU } :=
  frontendReshapePrimitive (frontendPadPrimitive base padv)

opaque frontendFrozenPadPrimitive (_base : torch.Tensor { shape := #[2], dtype := .Float32, device := .CPU })
    (_padv : torch.Frozen { shape := #[1], dtype := .Float32, device := .CPU }) : torch.Tensor { shape := #[6], dtype := .Float32, device := .CPU } :=
  torch.Tensor.zeros #[6]

def frozenPadWrapper (base : torch.Tensor { shape := #[2], dtype := .Float32, device := .CPU })
    (padv : torch.Frozen { shape := #[1], dtype := .Float32, device := .CPU }) : torch.Tensor { shape := #[6], dtype := .Float32, device := .CPU } :=
  frontendFrozenPadPrimitive base padv

def TensorTypeAlias (spec : torch.StaticSpec) : Type := torch.Tensor spec
abbrev AliasedParameter := TensorTypeAlias { shape := #[2], dtype := .Float64, device := .CPU }
def FrozenTypeAlias : Type := torch.Frozen { shape := #[1], dtype := .Int64, device := .CPU }

opaque frontendAliasedPrimitive (_base : AliasedParameter)
    (_padv : (let type := FrozenTypeAlias; type)) :
    TensorTypeAlias { shape := #[6], dtype := .Float64, device := .CPU } :=
  torch.Tensor.zerosFor { shape := #[6], dtype := .Float64, device := .CPU }

def aliasedWrapper (base : AliasedParameter) (padv : FrozenTypeAlias) :
    TensorTypeAlias { shape := #[6], dtype := .Float64, device := .CPU } :=
  frontendAliasedPrimitive base padv

opaque frontendRawPrimitive (tensor : torch.RawTensor) : torch.RawTensor := tensor

def rawWrapper (tensor : torch.RawTensor) : torch.RawTensor :=
  frontendRawPrimitive tensor

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

@[test]
unsafe def testDeriveFrontendAliasedTensorSpecs : IO Unit := do
  let registration ← runCoreM <|
    deriveSinglePrimitiveFrontendRegistration ``Tests.ADFrontendElab.aliasedWrapper
      #[{ leanConst := ``Tests.ADFrontendElab.frontendAliasedPrimitive, op := padAliasOpName }]
  let jaxpr := registration.jaxpr
  LeanTest.assertTrue ((jaxpr.invars.map (·.metaInfo.shape)) == #[some #[2], some #[1]])
    "Alias normalization must stop at the public Tensor/Frozen type heads."
  LeanTest.assertTrue ((jaxpr.invars.map (·.metaInfo.participation)) == #[.diff, .frozen])
    "Tensor and Frozen aliases must retain distinct AD participation roles."
  LeanTest.assertTrue ((jaxpr.invars.map (·.metaInfo.dtype)) == #[some "Float64", some "Int64"])
    "Alias normalization must preserve each parameter's own static specification."
  LeanTest.assertTrue ((jaxpr.outvars.map (·.metaInfo.shape)) == #[some #[6]])
  LeanTest.assertTrue ((jaxpr.outvars.map (·.metaInfo.dtype)) == #[some "Float64"])

@[test]
unsafe def testDeriveFrontendStartsFreshHeartbeatBudget : IO Unit := do
  let previous ← IO.getNumHeartbeats
  try
    IO.addHeartbeats (Core.getMaxHeartbeats {} + 1)
    testDeriveFrontendAliasedTensorSpecs
  finally
    IO.setNumHeartbeats previous

@[test]
unsafe def testDeriveFrontendRejectsRawHandles : IO Unit := do
  let result ← try
    let _ ← runCoreM <|
      deriveSinglePrimitiveFrontendRegistration ``Tests.ADFrontendElab.rawWrapper
        #[{ leanConst := ``Tests.ADFrontendElab.frontendRawPrimitive, op := padAliasOpName }]
    pure (none : Option String)
  catch error =>
    pure (some error.toString)
  match result with
  | none => LeanTest.fail "A raw handle has no indexed tensor metadata."
  | some message => LeanTest.assertTrue (message.containsSubstr "only supports tensor-like parameters")

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
