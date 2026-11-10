import LeanTest
import Tyr.EventSkeleton.Examples.HydroelasticDemos

namespace Tests.EventSkeletonHydroelasticDemosExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.HydroelasticDemos

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertOk {α : Type} (res : Except String α) (label : String) : IO α := do
  match res with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

private def assertError {α : Type} (res : Except String α) (label : String) :
    IO String := do
  match res with
  | .ok _ => LeanTest.fail s!"{label}: expected error, got ok"
  | .error msg => pure msg

private def finiteArray (xs : Array Float) : Bool :=
  xs.all (fun x => x.isFinite)

private def finiteArrays (xss : Array (Array Float)) : Bool :=
  xss.all finiteArray

private def hasMoveLabel (moves : Array SkeletonMove) (label : String) : Bool :=
  moves.any (fun move => move.label == label)

private def hasMoveLabelWithExactness
    (moves : Array SkeletonMove) (label : String) (exactness : MoveExactness) :
    Bool :=
  moves.any (fun move => move.label == label && move.exactness == exactness)

private def hasAllMoveLabels (moves : Array SkeletonMove) (labels : Array String) :
    Bool :=
  labels.all (hasMoveLabel moves)

private def fullPhysicsMoveLabels : Array String :=
  #[
    "full-physics-step:python ball-paddle hydroelastic full physics",
    "full-physics-step:python nonconvex pepper-table full physics",
    "full-physics-step:spatula slip-control hydroelastic full physics"
  ]

private def contactSupportMoveLabels : Array String :=
  #[
    "contact-support-selection:python ball-paddle hydroelastic full physics",
    "contact-support-selection:python nonconvex pepper-table full physics",
    "contact-support-selection:spatula slip-control hydroelastic full physics"
  ]

@[test]
def testDrakeReferencesAndDefaultsAreRecorded : IO Unit := do
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/hydroelastic/python_ball_paddle/contact_sim_demo.py"))
    "Hydroelastic demo should reference Drake's Python ball-paddle runner"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/hydroelastic/python_nonconvex_mesh/drop_pepper.py"))
    "Hydroelastic demo should reference Drake's pepper/bowl/table runner"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/hydroelastic/spatula_slip_control/spatula_slip_control.cc"))
    "Hydroelastic demo should reference Drake's spatula slip-control runner"

  LeanTest.assertEqual ballPaddleParams.initialQ.size 7
    "Ball-paddle free body should expose quaternion plus xyz positions"
  LeanTest.assertEqual ballPaddleParams.initialV.size 6
    "Ball-paddle free body should expose six generalized velocities"
  LeanTest.assertEqual ballPaddleParams.initialState.size 13
    "Ball-paddle initial state should match 7 q plus 6 v"
  LeanTest.assertTrue (approx ballPaddleParams.paddleTopZ 0.0 1.0e-12)
    s!"Paddle weld and size should put the top surface at z=0, got {ballPaddleParams.paddleTopZ}"
  LeanTest.assertTrue
    (ballPaddleParams.contactModel == ContactModelChoice.hydroelasticWithFallback)
    "Ball-paddle should keep Drake's hydroelastic_with_fallback default"

  LeanTest.assertEqual nonconvexMeshParams.initialQ.size 14
    "Pepper/bowl state should expose two free-body quaternion-position blocks"
  LeanTest.assertEqual nonconvexMeshParams.initialV.size 12
    "Pepper/bowl state should expose two free-body velocity blocks"
  LeanTest.assertEqual nonconvexMeshParams.initialState.size 26
    "Pepper/bowl initial state should match 14 q plus 12 v"
  LeanTest.assertTrue (approx (nonconvexMeshParams.initialV.getD 2 0.0) 150.0 1.0e-12)
    s!"Pepper spin should match the Drake runner, got {nonconvexMeshParams.initialV.getD 2 0.0}"

  LeanTest.assertTrue
    (spatulaSlipParams.contactModel == ContactModelChoice.hydroelastic)
    "Spatula demo should keep Drake's hydroelastic default"
  LeanTest.assertTrue (spatulaSlipParams.contactApproximation == "lagged")
    "Spatula demo should keep Drake's lagged contact approximation flag"
  LeanTest.assertTrue (approx spatulaSlipParams.period 3.0 1.0e-12)
    s!"Spatula square-wave period should be 3s, got {spatulaSlipParams.period}"
  LeanTest.assertTrue (approx spatulaSlipParams.torsionalFrictionRadius 0.015 1.0e-12)
    s!"Spatula torsional friction radius should match the handle radius, got {spatulaSlipParams.torsionalFrictionRadius}"

  assertOk validateDocumentationImageAssets?
    "Hydroelastic README image asset catalog"
  LeanTest.assertTrue
    (documentationImagePaths.any (fun path =>
      path == "../drake/examples/hydroelastic/spatula_slip_control/images/spatula_3.jpg"))
    "Spatula documentation image sidecars should be recorded"

@[test]
def testHydroelasticPatchSupportAdaptsToFullPhysicsContactSupport : IO Unit := do
  let support := ballPaddleSupport ballPaddleParams { z := 0.015 }
  assertOk (support.validateGeometry?) "ball-paddle patch geometry"
  assertOk (support.validateJacobianWidth? 6) "ball-paddle free-body patch width"
  let selectedIds ← assertOk support.selectedIds? "ball-paddle selected patch ids"
  LeanTest.assertEqual selectedIds #[2000]
    "Ball-paddle support should retain the active hydroelastic patch"

  let contactSupport := support.equivalentContactSupport
  assertOk (contactSupport.validateJacobianWidth? 6)
    "equivalent contact support should preserve free-body Jacobian width"
  let contactIds ← assertOk contactSupport.selectedIds? "equivalent selected contact ids"
  LeanTest.assertEqual contactIds selectedIds
    "Patch ids should remain stable across the hydroelastic-to-contact adapter"
  let forces ← assertOk support.selectedContactForces?
    "selected hydroelastic patch force scalars"
  LeanTest.assertEqual forces.size 1
    "One active patch should produce one scalar force record"
  LeanTest.assertEqual forces[0]!.candidateId 2000
    "Force candidate id should match the retained hydroelastic patch"
  LeanTest.assertTrue (forces[0]!.normalForce > 0.0)
    s!"Active ball-paddle patch should produce positive normal force, got {forces[0]!.normalForce}"

@[test]
def testBallPaddleFullPhysicsUsesMassMatrixAndHydroelasticForce : IO Unit := do
  let result ← assertOk (ballPaddleFullPhysics? ballPaddleParams { z := 0.015 })
    "ball-paddle full physics"
  LeanTest.assertEqual result.support.totalCandidates 1
    "Ball-paddle full physics should receive one dynamic contact candidate"
  LeanTest.assertEqual result.contactForces.size 1
    "Ball-paddle full physics should solve one selected contact force"
  LeanTest.assertTrue (result.generalizedContactForce.getD 5 0.0 > 0.0)
    s!"Ball-paddle patch should push along translational z, got {result.generalizedContactForce}"
  LeanTest.assertTrue (result.derivative.vdot.getD 5 0.0 > 0.0)
    s!"Hydroelastic force should dominate gravity at this penetration, got vdot={result.derivative.vdot}"
  LeanTest.assertTrue (finiteArray result.derivative.qdot && finiteArray result.derivative.vdot)
    s!"Ball-paddle derivative should remain finite, got {reprStr result.derivative}"
  LeanTest.assertTrue (result.move.kind == SkeletonMoveKind.intervalAdjoint)
    "Full-physics solve should contribute an interval adjoint move"

@[test]
def testNonconvexPepperTableFullPhysicsUsesDynamicPatchProvider : IO Unit := do
  let support := pepperTableSupport nonconvexMeshParams { y := -0.15, z := -0.002 }
  assertOk (support.validateGeometry?) "pepper-table patch geometry"
  assertOk (support.validateJacobianWidth? 12) "pepper-table free-body patch width"
  let ids ← assertOk support.selectedIds? "pepper-table selected ids"
  LeanTest.assertEqual ids #[2100]
    "Pepper-table support should retain the active table patch"

  let result ← assertOk
    (pepperTableFullPhysics? nonconvexMeshParams { y := -0.15, z := -0.002 })
    "pepper-table full physics"
  LeanTest.assertEqual result.contactForces.size 1
    "Pepper-table full physics should solve one selected patch force"
  LeanTest.assertTrue (result.generalizedContactForce.getD 5 0.0 > 0.0)
    s!"Pepper-table contact should push pepper translational z, got {result.generalizedContactForce}"
  LeanTest.assertTrue (approx (result.derivative.qdot.getD 2 0.0) 150.0 1.0e-12)
    s!"Full-physics qdot should preserve the Drake pepper spin, got {result.derivative.qdot}"
  LeanTest.assertTrue (finiteArray result.derivative.vdot)
    s!"Pepper-table derivative should remain finite, got {reprStr result.derivative}"

@[test]
def testSpatulaControllerAndFullPhysicsBoundary : IO Unit := do
  LeanTest.assertEqual (spatulaSlipParams.actuation 0.0) #[6.5, -6.5]
    "Square-wave high phase should add +/-5N to the constant gripper force"
  LeanTest.assertEqual (spatulaSlipParams.actuation 1.49) #[6.5, -6.5]
    "Square wave should still be high before the 1.5s duty boundary"
  LeanTest.assertEqual (spatulaSlipParams.actuation 1.5) #[1.5, -1.5]
    "Square wave should drop exactly at the 1.5s duty boundary"
  LeanTest.assertEqual (spatulaSlipParams.actuation 3.1) #[6.5, -6.5]
    "Square wave should repeat at the next period"

  let support := spatulaPatchSupport spatulaSlipParams
  assertOk (support.validateJacobianWidth? 3)
    "spatula finger patches should expose two finger rows plus a yaw-slip row"
  let ids ← assertOk support.selectedIds? "spatula selected ids"
  LeanTest.assertEqual ids #[2200, 2201]
    "Both spatula finger patches should be retained"
  let normalForce ← assertOk (spatulaSelectedNormalForceSum? support)
    "spatula selected normal force sum"
  LeanTest.assertTrue (approx normalForce 6.0 1.0e-12)
    s!"Both spatula patches should contribute pressure-derived normal force, got {normalForce}"

  let result ← assertOk (spatulaFullPhysics? spatulaSlipParams 0.0)
    "spatula full physics"
  LeanTest.assertEqual result.contactForces.size 2
    "Spatula full physics should receive both finger patch forces"
  LeanTest.assertTrue (result.generalizedContactForce.getD 0 0.0 > 0.0)
    s!"Left finger contact should push coordinate 0, got {result.generalizedContactForce}"
  LeanTest.assertTrue (result.generalizedContactForce.getD 1 0.0 > 0.0)
    s!"Right finger contact should push coordinate 1, got {result.generalizedContactForce}"
  LeanTest.assertTrue (approx (result.generalizedPrimitiveForce.getD 2 0.0) 0.0 1.0e-12)
    s!"Zero slip should stay in the torsional stiction deadband, got {result.generalizedPrimitiveForce}"
  LeanTest.assertTrue (finiteArray result.derivative.vdot)
    s!"Spatula full-physics derivative should remain finite, got {reprStr result.derivative}"

  let slipping := { spatulaSlipParams with spatulaSlipRate := 0.4 }
  let slipResult ← assertOk (spatulaFullPhysics? slipping 0.0)
    "spatula slipping full physics"
  LeanTest.assertTrue (slipResult.generalizedPrimitiveForce.getD 2 0.0 < 0.0)
    s!"Positive slip should receive opposing torsional friction, got {slipResult.generalizedPrimitiveForce}"
  LeanTest.assertTrue (slipResult.derivative.vdot.getD 2 0.0 < 0.0)
    s!"Pressure-dependent torsional friction should decelerate positive slip, got {slipResult.derivative.vdot}"
  LeanTest.assertTrue (approx (slipResult.derivative.qdot.getD 2 0.0) 0.4 1.0e-12)
    s!"Full-physics qdot should preserve the sampled slip rate, got {slipResult.derivative.qdot}"

@[test]
def testHydroelasticFullPhysicsPrimitiveProvidersRecomputeStateAndTime :
    IO Unit := do
  let ballProvider :=
    ballPaddleFullPhysicsPrimitiveProvider
      "ball-paddle hydroelastic dynamic provider test"
  let ballVelocity := #[0.0, 0.0, 0.0, 0.0, 0.0, -0.7]
  let ballAirborne :=
    ballPaddlePhysicsState ballPaddleParams { z := 0.30 } ballVelocity
  let ballContact :=
    ballPaddlePhysicsState ballPaddleParams { z := 0.015 } ballVelocity
  let ballAirSupport ← assertOk (ballProvider.supportAt? ballAirborne)
    "airborne ball-paddle support"
  let ballAirIds ← assertOk ballAirSupport.selectedIds?
    "airborne ball-paddle selected ids"
  LeanTest.assertEqual ballAirIds (#[] : Array Nat)
    "Airborne ball-paddle state should not retain the hydroelastic patch"
  let ballContactSupport ← assertOk (ballProvider.supportAt? ballContact)
    "contact ball-paddle support"
  let ballContactIds ← assertOk ballContactSupport.selectedIds?
    "contact ball-paddle selected ids"
  LeanTest.assertEqual ballContactIds #[2000]
    "Contact ball-paddle state should retain the ball-paddle patch"
  let ballPrimitives ← assertOk (ballProvider.validPrimitivesAt? ballContact)
    "ball-paddle provider primitives"
  LeanTest.assertTrue (approx (ballPrimitives.qdot.getD 5 0.0) (-0.7) 1.0e-12)
    s!"Ball-paddle provider should preserve current vertical qdot, got {ballPrimitives.qdot}"
  let ballResult ← assertOk (ballProvider.solveAt? ballContact 5600)
    "ball-paddle provider solve"
  LeanTest.assertTrue (ballResult.derivative.vdot.getD 5 0.0 > 0.0)
    s!"Dynamic ball-paddle patch should push through the provider, got {ballResult.derivative.vdot}"
  LeanTest.assertTrue
    (ballResult.move.label ==
      "full-physics-step:ball-paddle hydroelastic dynamic provider test")
    s!"Provider solve should carry the provider label, got {ballResult.move.label}"

  let pepperProvider :=
    pepperTableFullPhysicsPrimitiveProvider
      "pepper-table hydroelastic dynamic provider test"
  let pepperVelocity :=
    #[0.0, 0.0, 80.0, 0.0, 0.0, -0.3,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  let pepperAirborne :=
    pepperTablePhysicsState nonconvexMeshParams { y := -0.15, z := 0.20 }
      pepperVelocity
  let pepperContact :=
    pepperTablePhysicsState nonconvexMeshParams { y := -0.15, z := -0.002 }
      pepperVelocity
  let pepperAirSupport ← assertOk (pepperProvider.supportAt? pepperAirborne)
    "airborne pepper-table support"
  let pepperAirIds ← assertOk pepperAirSupport.selectedIds?
    "airborne pepper-table selected ids"
  LeanTest.assertEqual pepperAirIds (#[] : Array Nat)
    "Airborne pepper-table state should not retain the table patch"
  let pepperContactSupport ← assertOk (pepperProvider.supportAt? pepperContact)
    "contact pepper-table support"
  let pepperContactIds ← assertOk pepperContactSupport.selectedIds?
    "contact pepper-table selected ids"
  LeanTest.assertEqual pepperContactIds #[2100]
    "Contact pepper-table state should retain the table patch"
  let pepperResult ← assertOk (pepperProvider.solveAt? pepperContact 5601)
    "pepper-table provider solve"
  LeanTest.assertTrue (approx (pepperResult.derivative.qdot.getD 2 0.0) 80.0 1.0e-12)
    s!"Pepper-table provider should preserve current spin qdot, got {pepperResult.derivative.qdot}"
  LeanTest.assertTrue (pepperResult.generalizedContactForce.getD 5 0.0 > 0.0)
    s!"Pepper-table provider should push the pepper upward, got {pepperResult.generalizedContactForce}"

  let spatulaProvider :=
    spatulaSlipFullPhysicsPrimitiveProvider
      "spatula hydroelastic dynamic provider test"
  let highPrimitives ← assertOk
    (spatulaProvider.validPrimitivesAt? (spatulaSlipPhysicsState spatulaSlipParams 0.0))
    "spatula high-phase primitives"
  LeanTest.assertEqual highPrimitives.actuationForces #[6.5, -6.5, 0.0]
    "Spatula provider should recompute high-phase square-wave actuation"
  let lowPrimitives ← assertOk
    (spatulaProvider.validPrimitivesAt? (spatulaSlipPhysicsState spatulaSlipParams 1.5))
    "spatula low-phase primitives"
  LeanTest.assertEqual lowPrimitives.actuationForces #[1.5, -1.5, 0.0]
    "Spatula provider should recompute low-phase square-wave actuation"
  let slipping := { spatulaSlipParams with spatulaSlipRate := 0.4 }
  let slipResult ← assertOk
    (spatulaProvider.solveAt? (spatulaSlipPhysicsState slipping 0.0) 5602)
    "spatula provider slipping solve"
  LeanTest.assertTrue (slipResult.generalizedPrimitiveForce.getD 2 0.0 < 0.0)
    s!"Spatula provider should recompute slip-opposing torsional friction, got {slipResult.generalizedPrimitiveForce}"
  LeanTest.assertTrue (approx (slipResult.derivative.qdot.getD 2 0.0) 0.4 1.0e-12)
    s!"Spatula provider should preserve current slip qdot, got {slipResult.derivative.qdot}"

  let badBallMsg ← assertError
    (ballProvider.validPrimitivesAt?
      (ballPaddlePhysicsState ballPaddleParams { z := 0.015 } #[0.0]))
    "bad ball-paddle velocity"
  LeanTest.assertTrue (badBallMsg.contains "velocity size")
    s!"Bad ball-paddle velocity should report a dimension error, got {badBallMsg}"

@[test]
def testEndToEndHydroelasticDemosTraceCarriesFullPhysicsMoves : IO Unit := do
  let result ← assertOk (buildEndToEnd? ballPaddleParams nonconvexMeshParams spatulaSlipParams)
    "hydroelastic demos end-to-end"
  assertOk result.trace.validate? "hydroelastic demo trace validation"
  LeanTest.assertEqual result.ballPaddleInitialState.size 13
    "End-to-end result should carry the ball-paddle initial state"
  LeanTest.assertEqual result.pepperBowlInitialState.size 26
    "End-to-end result should carry the pepper/bowl initial state"
  LeanTest.assertEqual result.spatulaActuationSamples.size 5
    "End-to-end result should carry the sampled square-wave controller outputs"
  LeanTest.assertEqual
    (result.documentationImages.map (fun asset => asset.fullPath))
    documentationImagePaths
    "End-to-end result should carry the hydroelastic README image sidecars"
  LeanTest.assertTrue (hasMoveLabel result.moves "hydroelastic patch-to-contact full-physics adapter")
    "End-to-end move list should include the patch-to-contact adapter"
  LeanTest.assertTrue (hasMoveLabel result.moves "spatula pressure-dependent torsional friction primitive")
    "End-to-end move list should include the spatula torsional friction primitive"
  LeanTest.assertTrue
    (hasAllMoveLabels result.moves fullPhysicsMoveLabels)
    "Each hydroelastic demo should contribute a full-physics interval move"
  LeanTest.assertTrue
    (contactSupportMoveLabels.all (fun label =>
      hasMoveLabelWithExactness result.moves label MoveExactness.controlledApproximation))
    "Each hydroelastic demo should expose threshold-selected patch support separately from the exact solve"
  LeanTest.assertTrue
    (finiteArrays #[
      result.ballPaddleFullPhysics.derivative.vdot,
      result.pepperTableFullPhysics.derivative.vdot,
      result.spatulaFullPhysics.derivative.vdot
    ])
    "Each hydroelastic full-physics derivative should be finite in the end-to-end result"
  LeanTest.assertTrue
    (hasMoveLabel result.moves "hydroelastic README image documentation boundary")
    "End-to-end result should keep README image assets as a non-physics documentation boundary"

end Tests.EventSkeletonHydroelasticDemosExample
