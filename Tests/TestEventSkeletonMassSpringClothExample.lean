import LeanTest
import Tyr.EventSkeleton.Examples.MassSpringCloth

namespace Tests.EventSkeletonMassSpringClothExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.MassSpringCloth

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertOk {α : Type} (res : Except String α) (label : String) : IO α := do
  match res with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

private def assertError {α : Type} (res : Except String α) (label : String) : IO String := do
  match res with
  | .ok _ => LeanTest.fail s!"{label}: expected error"
  | .error msg => pure msg

private def assertSome {α : Type} (x : Option α) (label : String) : IO α := do
  match x with
  | some value => pure value
  | none => LeanTest.fail s!"{label}: expected some"

private def assertArrayApprox
    (actual expected : Array Float)
    (tol : Float)
    (label : String) : IO Unit := do
  let diff := FloatArray.maxAbsDiff actual expected
  LeanTest.assertTrue (diff < tol)
    s!"{label}: max abs diff {diff}, actual={actual}, expected={expected}"

@[test]
def testDrakeReferencesAndDefaultsAreRecorded : IO Unit := do
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/mass_spring_cloth/cloth_spring_model.h"))
    "Example should reference Drake's cloth spring system declaration"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/mass_spring_cloth/cloth_spring_model.cc"))
    "Example should reference Drake's cloth spring system implementation"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/mass_spring_cloth/cloth_spring_model_params.h"))
    "Example should reference Drake's cloth parameter defaults"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/mass_spring_cloth/cloth_spring_model_params.cc"))
    "Example should reference Drake's cloth parameter coordinate implementation"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/mass_spring_cloth/run_cloth_spring_model.cc"))
    "Example should reference Drake's simulation driver flags"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/mass_spring_cloth/cloth_spring_model_geometry.h"))
    "Example should reference Drake's per-particle geometry declaration"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/mass_spring_cloth/cloth_spring_model_geometry.cc"))
    "Example should reference Drake's per-particle geometry registration"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/mass_spring_cloth/test/cloth_spring_model_test.cc"))
    "Example should reference Drake's continuous and discrete cloth simulation tests"

  LeanTest.assertEqual parameterCoordinateNames #["mass", "k", "d", "gravity"]
    "Parameter names should preserve Drake's cloth model parameters"
  LeanTest.assertEqual params.nx 20
    "Default nx should match Drake's run_cloth_spring_model flag"
  LeanTest.assertEqual params.ny 20
    "Default ny should match Drake's run_cloth_spring_model flag"
  LeanTest.assertTrue (approx params.spacing 0.05 1.0e-12)
    s!"Default h should be 0.05, got {params.spacing}"
  LeanTest.assertTrue (approx params.dt 0.01 1.0e-12)
    s!"Default dt should be 0.01, got {params.dt}"
  LeanTest.assertTrue (approx params.physical.mass 1.0 1.0e-12)
    s!"Default mass should be 1.0, got {params.physical.mass}"
  LeanTest.assertTrue (approx params.physical.stiffness 100.0 1.0e-12)
    s!"Default stiffness should be 100.0, got {params.physical.stiffness}"
  LeanTest.assertTrue (approx params.physical.damping 10.0 1.0e-12)
    s!"Default damping should be 10.0, got {params.physical.damping}"
  LeanTest.assertTrue (approx params.physical.gravityZ (-9.81) 1.0e-12)
    s!"Default gravity should be -9.81, got {params.physical.gravityZ}"
  LeanTest.assertTrue (approx (visualParticleRadius params) 0.04 1.0e-12)
    s!"Drake visual sphere radius should be 0.8*h, got {visualParticleRadius params}"
  LeanTest.assertEqual (particleFrameName 7) "particle7"
    "Geometry frame names should preserve Drake's particle{i} naming"

@[test]
def testClothSpringModelGeometryProviderMatchesDrakeSceneGraphRegistration : IO Unit := do
  let p : ClothParams := { params with nx := 3, ny := 2, spacing := 0.5 }
  let result ← assertOk (buildClothSpringModelGeometry? p (defaultState p))
    "ClothSpringModelGeometry provider"
  assertOk result.provider.validate? "ClothSpringModelGeometry SceneGraph provider"
  assertOk (result.poses.validate? result.provider) "ClothSpringModelGeometry pose output"
  LeanTest.assertEqual result.inputPortName "particle_positions"
    "ClothSpringModelGeometry should declare Drake's particle_positions input"
  LeanTest.assertEqual result.inputPortSize 18
    "ClothSpringModelGeometry input should have 3*num_particles entries"
  LeanTest.assertEqual result.outputPortName "geometry_pose"
    "ClothSpringModelGeometry should declare an abstract output port named geometry_pose"
  LeanTest.assertTrue (approx result.particleRadius (0.8 * p.spacing) 1.0e-12)
    s!"Particle visual radius should be 0.8*h, got {result.particleRadius}"
  LeanTest.assertEqual result.provider.sources.size 1
    "ClothSpringModelGeometry should register one SceneGraph source"
  LeanTest.assertEqual result.provider.frames.size p.particleCount
    "ClothSpringModelGeometry should register one frame per particle"
  LeanTest.assertEqual result.provider.geometries.size p.particleCount
    "ClothSpringModelGeometry should register one sphere per particle"
  LeanTest.assertTrue result.provider.anchoredGeometries.isEmpty
    "Cloth particles should be frame-attached, not anchored"
  LeanTest.assertTrue (result.provider.shapeNames.all (fun name => name == "sphere"))
    s!"Cloth geometry should register only spheres, got {result.provider.shapeNames}"

  let frame0 ← assertSome (result.provider.frameById? (particleFrameId 0))
    "cloth particle0 frame lookup"
  let frame5 ← assertSome (result.provider.frameById? (particleFrameId 5))
    "cloth particle5 frame lookup"
  LeanTest.assertEqual frame0.name "particle0"
    "First particle frame should preserve Drake's particle0 name"
  LeanTest.assertEqual frame5.name "particle5"
    "Last particle frame should preserve Drake's particle index name"

  let sphere0 ← assertSome (result.provider.geometryById? (particleGeometryId 0))
    "cloth particle0 geometry lookup"
  LeanTest.assertEqual sphere0.name "sphere_visual"
    "Particle sphere geometry should preserve Drake's sphere_visual name"
  LeanTest.assertEqual sphere0.frameId? (some (particleFrameId 0))
    "Particle sphere should attach to its particle frame"
  LeanTest.assertTrue (sphere0.X_FG == ScenePose3.identity)
    s!"Particle sphere should use identity X_FG, got {reprStr sphere0.X_FG}"
  LeanTest.assertTrue (sphere0.properties.diffuseRgba? ==
      some { r := 1.0, g := 0.0, b := 1.0, a := 1.0 })
    s!"Particle sphere should carry Drake's magenta diffuse color, got {reprStr sphere0.properties.diffuseRgba?}"
  LeanTest.assertTrue (sphere0.hasRole .illustration)
    "Particle sphere should carry the illustration role"
  match sphere0.shape with
  | .sphere radius =>
      LeanTest.assertTrue (approx radius (0.8 * p.spacing) 1.0e-12)
        s!"Particle sphere should have radius 0.8*h, got {radius}"
  | other => LeanTest.fail s!"Cloth particle geometry should be a sphere, got {reprStr other}"

@[test]
def testClothSpringModelGeometryPoseOutputMatchesParticlePositions : IO Unit := do
  let p : ClothParams := { params with nx := 2, ny := 2, spacing := 0.5 }
  let x : ClothState :=
    {
      q := #[
        0.0, 0.0, 0.1,
        0.0, 0.5, 0.2,
        0.5, 0.0, 0.3,
        0.5, 0.5, 0.4
      ]
      v := ParticleSpringSystem.zeroVelocities p.particleCount
    }
  let result ← assertOk (buildClothSpringModelGeometry? p x)
    "ClothSpringModelGeometry pose output"
  let pose0 ← assertSome (result.poses.poseForFrame? (particleFrameId 0))
    "cloth particle0 pose"
  let pose3 ← assertSome (result.poses.poseForFrame? (particleFrameId 3))
    "cloth particle3 pose"
  LeanTest.assertTrue (pose0.translation == { x := 0.0, y := 0.0, z := 0.1 })
    s!"Particle 0 pose should copy particle_positions[0:3], got {reprStr pose0.translation}"
  LeanTest.assertTrue (pose3.translation == { x := 0.5, y := 0.5, z := 0.4 })
    s!"Particle 3 pose should copy particle_positions[9:12], got {reprStr pose3.translation}"
  LeanTest.assertTrue (pose0.rotationAxis == SceneVec3.unitZ &&
      approx pose0.rotationAngle 0.0 1.0e-12 &&
      pose3.rotationAxis == SceneVec3.unitZ &&
      approx pose3.rotationAngle 0.0 1.0e-12)
    s!"Particle poses should be pure translations, got {reprStr pose0}, {reprStr pose3}"

@[test]
def testClothSpringModelGeometryGraphRecordsExactSceneGraphBoundary : IO Unit := do
  let p : ClothParams := { params with nx := 2, ny := 2, spacing := 0.5 }
  let result ← assertOk (buildClothSpringModelGeometry? p (defaultState p))
    "ClothSpringModelGeometry graph"
  LeanTest.assertEqual result.moves.size 2
    "ClothSpringModelGeometry should expose registration and pose-output local moves"
  LeanTest.assertTrue (result.moves.all (fun move =>
      move.kind == SkeletonMoveKind.localSchurBlock &&
      move.exactness == MoveExactness.exact))
    "ClothSpringModelGeometry moves should be exact local SceneGraph blocks"
  LeanTest.assertTrue (result.moves.any (fun move =>
      move.targets == #[clothGeometryProviderVertex] &&
      move.writes == #[clothGeometryProviderVertex] &&
      move.label.contains "Register particle frames"))
    "ClothSpringModelGeometry graph should record provider registration"
  LeanTest.assertTrue (result.moves.any (fun move =>
      move.targets == #[clothGeometryPoseOutputVertex] &&
      move.reads == #[clothGeometryStateInputVertex, clothGeometryProviderVertex] &&
      move.writes == #[clothGeometryPoseOutputVertex] &&
      move.label.contains "OutputGeometryPose"))
    "ClothSpringModelGeometry graph should record particle_positions-to-FramePoseVector output"

@[test]
def testGridTopologyMatchesDrakeOrdering : IO Unit := do
  let p : ClothParams := { params with nx := 3, ny := 2, spacing := 0.5 }
  let ss := springs p
  LeanTest.assertEqual p.particleCount 6
    "Particle count should be nx*ny"
  LeanTest.assertEqual p.pinnedParticles #[0, 1]
    "Drake fixes the bottom-left and top-left particles for ny=2"
  LeanTest.assertEqual p.stretchingSpringCount 7
    "Stretching springs should cover y-neighbors and x-neighbors"
  LeanTest.assertEqual p.shearingSpringCount 4
    "Shearing springs should add two diagonals per grid cell"
  LeanTest.assertEqual p.springCount 11
    "Total spring count should include stretch and shear springs"
  LeanTest.assertEqual ss.size 11
    "Generated spring topology should match the counted topology"

  LeanTest.assertEqual ss[0]!.particle0 0
    "First y-direction spring should start at particle 0"
  LeanTest.assertEqual ss[0]!.particle1 1
    "First y-direction spring should connect particle 0 to 1"
  LeanTest.assertTrue (approx ss[0]!.restLength 0.5 1.0e-12)
    s!"Stretching spring rest length should be h, got {ss[0]!.restLength}"

  LeanTest.assertEqual ss[3]!.particle0 0
    "First x-direction spring should start after all y springs"
  LeanTest.assertEqual ss[3]!.particle1 2
    "First x-direction spring should connect particle 0 to particle ny"
  LeanTest.assertTrue (approx ss[3]!.restLength 0.5 1.0e-12)
    s!"X stretching spring rest length should be h, got {ss[3]!.restLength}"

  LeanTest.assertEqual ss[7]!.particle0 0
    "First shear spring should start after all stretching springs"
  LeanTest.assertEqual ss[7]!.particle1 3
    "First shear spring should connect the lower-left cell diagonal"
  LeanTest.assertTrue (approx ss[7]!.restLength (Float.sqrt 2.0 * 0.5) 1.0e-12)
    s!"Shear spring rest length should be sqrt(2)*h, got {ss[7]!.restLength}"
  LeanTest.assertEqual ss[8]!.particle0 1
    "Second shear spring should connect the opposite diagonal"
  LeanTest.assertEqual ss[8]!.particle1 2
    "Second shear spring should connect particle 1 to particle ny"

@[test]
def testInitialStateMatchesDrakeGrid : IO Unit := do
  let p : ClothParams := { params with nx := 3, ny := 2, spacing := 0.5 }
  let x := defaultState p
  LeanTest.assertEqual x.q.size 18
    "Flat position array should hold 3 coordinates per particle"
  LeanTest.assertEqual x.v.size 18
    "Flat velocity array should hold 3 coordinates per particle"
  LeanTest.assertEqual (x.particlePosition 0) #[0.0, 0.0, 0.0]
    "Particle 0 should be the lower-left grid point"
  LeanTest.assertEqual (x.particlePosition 1) #[0.0, 0.5, 0.0]
    "Particle index i*ny+j should advance along y first"
  LeanTest.assertEqual (x.particlePosition 2) #[0.5, 0.0, 0.0]
    "Particle 2 should be the first particle in the second x column"
  LeanTest.assertEqual (x.particlePosition 5) #[1.0, 0.5, 0.0]
    "Last particle should sit at ((nx-1)*h, (ny-1)*h, 0)"
  LeanTest.assertTrue (x.v.all (fun value => approx value 0.0 1.0e-12))
    s!"Initial velocities should be zero, got {x.v}"

@[test]
def testRestStateHasGravityOnlyAwayFromPinnedCorners : IO Unit := do
  let p : ClothParams := { params with nx := 3, ny := 3, spacing := 0.5 }
  let x := defaultState p
  let dx ← assertOk (derivative? p x) "cloth derivative at rest"
  LeanTest.assertTrue (dx.qdot.all (fun value => approx value 0.0 1.0e-12))
    s!"At rest, qdot should be zero, got {dx.qdot}"
  LeanTest.assertTrue (approx dx.springForces.elasticEnergy 0.0 1.0e-12)
    s!"Grid starts at spring rest lengths, got energy {dx.springForces.elasticEnergy}"
  LeanTest.assertTrue (approx dx.springForces.dampingPower 0.0 1.0e-12)
    s!"Zero velocities should produce no damping power, got {dx.springForces.dampingPower}"

  LeanTest.assertEqual (ParticleSpringSystem.particleState 0 dx.vdot) #[0.0, 0.0, 0.0]
    "Bottom-left pinned particle should have zero acceleration"
  LeanTest.assertEqual (ParticleSpringSystem.particleState 2 dx.vdot) #[0.0, 0.0, 0.0]
    "Top-left pinned particle should have zero acceleration for ny=3"
  LeanTest.assertTrue (approx (dx.vdot.getD 14 0.0) (-9.81) 1.0e-12)
    s!"Interior particle z acceleration should be gravity, got {dx.vdot.getD 14 0.0}"
  LeanTest.assertTrue (approx (dx.vdot.getD 12 99.0) 0.0 1.0e-12)
    s!"Interior particle x acceleration should have no spring residual, got {dx.vdot.getD 12 0.0}"
  LeanTest.assertTrue (approx (dx.vdot.getD 13 99.0) 0.0 1.0e-12)
    s!"Interior particle y acceleration should have no spring residual, got {dx.vdot.getD 13 0.0}"

@[test]
def testFullPhysicsPrimitiveMatchesContinuousClothDerivative : IO Unit := do
  let p : ClothParams := { params with nx := 3, ny := 3, spacing := 0.5 }
  let x := defaultState p
  let dx ← assertOk (derivative? p x) "cloth derivative through full physics"
  let (primitiveForce, forceResult) ← assertOk (primitiveGeneralizedForce? p x)
    "cloth primitive generalized force"
  let (fullPhysics, primitiveForceResult) ← assertOk (solveFullPhysics? p x 4904)
    "cloth primitive full physics"
  let m := p.physical.massPerParticle p.particleCount

  LeanTest.assertEqual fullPhysics.equation.massMatrix (massMatrix p)
    "Cloth full physics should expose one diagonal mass entry per particle coordinate"
  LeanTest.assertEqual fullPhysics.contactForces.size 0
    "Cloth force elements should not masquerade as contact forces"
  LeanTest.assertEqual fullPhysics.support.totalCandidates 0
    "Cloth spring graph should not create contact candidates"
  LeanTest.assertTrue (fullPhysics.supportMove.exactness == MoveExactness.exact)
    "Empty full-support selection should be exact"
  assertArrayApprox fullPhysics.generalizedPrimitiveForce primitiveForce 1.0e-12
    "Full physics should expose spring+damping+gravity as a primitive force contribution"
  assertArrayApprox primitiveForceResult.forces forceResult.forces 1.0e-12
    "Full physics primitive should reuse the ParticleSpringSystem force accumulator"
  assertArrayApprox fullPhysics.derivative.qdot dx.qdot 1.0e-12
    "Primitive full physics qdot should match the cloth derivative"
  assertArrayApprox fullPhysics.derivative.vdot dx.vdot 1.0e-12
    "Primitive full physics acceleration should match the cloth derivative"
  assertArrayApprox (ParticleSpringSystem.particleState 0 fullPhysics.generalizedPrimitiveForce)
    #[0.0, 0.0, 0.0] 1.0e-12
    "Pinned bottom-left particle should have zero primitive generalized force"
  assertArrayApprox (ParticleSpringSystem.particleState 2 fullPhysics.generalizedPrimitiveForce)
    #[0.0, 0.0, 0.0] 1.0e-12
    "Pinned top-left particle should have zero primitive generalized force"
  LeanTest.assertTrue (approx (fullPhysics.generalizedPrimitiveForce.getD 14 0.0)
      (m * p.physical.gravityZ) 1.0e-12)
    s!"Free center particle should receive mass-scaled gravity force, got {fullPhysics.generalizedPrimitiveForce.getD 14 0.0}"

@[test]
def testFullPhysicsPrimitiveProviderRecomputesClothState : IO Unit := do
  let p : ClothParams := { params with nx := 3, ny := 3, spacing := 0.5 }
  let provider := fullPhysicsPrimitiveProvider p
    "mass-spring cloth provider test"
  let x0 := defaultState p
  let saggedQ := ParticleSpringSystem.setParticleState x0.q 4
    #[0.5, 0.5, -0.1]
  let movingV :=
    ParticleSpringSystem.setParticleState
      (ParticleSpringSystem.setParticleState x0.v 0 #[10.0, 10.0, 10.0])
      4 #[0.2, -0.1, 0.3]
  let x1 : ClothState := { q := saggedQ, v := movingV }

  let primitive0 ← assertOk (provider.validPrimitivesAt? x0)
    "cloth provider primitive at rest"
  let primitive1 ← assertOk (provider.validPrimitivesAt? x1)
    "cloth provider primitive after sagging center particle"
  let result1 ← assertOk (provider.solveAt? x1 4905)
    "cloth provider solve after sagging center particle"
  let (directResult1, _) ← assertOk
    (solveFullPhysics? p x1 4906 "mass-spring cloth direct provider parity")
    "direct cloth solve after sagging center particle"

  LeanTest.assertEqual primitive0.generalizedForceContributions.size 1
    "Cloth provider should expose spring+damping+gravity as one primitive force contribution"
  LeanTest.assertEqual primitive1.generalizedForceContributions.size 1
    "Cloth provider should keep the force contribution when state changes"
  assertArrayApprox (ParticleSpringSystem.particleState 0 primitive1.qdot)
    #[0.0, 0.0, 0.0] 1.0e-12
    "Provider qdot should clamp the pinned bottom-left particle"
  assertArrayApprox (ParticleSpringSystem.particleState 2 primitive1.qdot)
    #[0.0, 0.0, 0.0] 1.0e-12
    "Provider qdot should clamp the pinned top-left particle"
  assertArrayApprox (ParticleSpringSystem.particleState 4 primitive1.qdot)
    #[0.2, -0.1, 0.3] 1.0e-12
    "Provider qdot should preserve the free center-particle velocity"
  LeanTest.assertTrue
    (FloatArray.maxAbsDiff primitive0.generalizedForceContributions[0]!.force
      primitive1.generalizedForceContributions[0]!.force > 1.0e-6)
    "Sagging and moving the center particle should recompute spring/damping/generalized force"
  assertArrayApprox result1.derivative.vdot directResult1.derivative.vdot 1.0e-12
    "Provider solve should match direct full-physics solve acceleration"
  assertArrayApprox result1.generalizedPrimitiveForce directResult1.generalizedPrimitiveForce 1.0e-12
    "Provider solve should expose the same recomputed spring+damping+gravity primitive force"
  LeanTest.assertEqual result1.move.targets #[4905]
    "Provider solve should target the supplied full-physics interval vertex"
  LeanTest.assertEqual result1.contactForces.size 0
    "Cloth provider should stay contact-free while using primitive force elements"

@[test]
def testDiscreteUpdateBoundaryMatchesDrakeImplicitDampingScheme : IO Unit := do
  let p : ClothParams := { params with nx := 3, ny := 3, spacing := 0.5, dt := 0.01 }
  let x := defaultState p
  let step ← assertOk
    (discreteStep? p x { accuracy := 2.220446049250313e-16 })
    "cloth discrete update at rest"

  LeanTest.assertEqual step.linearSystemSize p.positionDim
    "Discrete implicit damping system should have one velocity unknown per position coordinate"
  LeanTest.assertEqual step.maxIterations p.positionDim
    "Drake defaults the CG max iterations to the linear system size"
  LeanTest.assertTrue (approx step.accuracy 2.220446049250313e-16 1.0e-24)
    s!"Discrete solver accuracy should preserve the requested tight tolerance, got {step.accuracy}"
  LeanTest.assertEqual step.dampingMatrix.size p.positionDim
    "Implicit damping matrix should be square with state velocity dimension"
  LeanTest.assertEqual (step.dampingMatrix.getD 0 #[]).size p.positionDim
    "Implicit damping matrix rows should have state velocity dimension"
  LeanTest.assertTrue
    (approx ((step.dampingMatrix.getD 0 #[]).getD 0 0.0) 1.0 1.0e-12 &&
      approx ((step.dampingMatrix.getD 1 #[]).getD 1 0.0) 1.0 1.0e-12 &&
      approx ((step.dampingMatrix.getD 2 #[]).getD 2 0.0) 1.0 1.0e-12)
    "Pinned bottom-left particle should have identity damping block"
  LeanTest.assertTrue
    (approx ((step.dampingMatrix.getD 6 #[]).getD 6 0.0) 1.0 1.0e-12 &&
      approx ((step.dampingMatrix.getD 7 #[]).getD 7 0.0) 1.0 1.0e-12 &&
      approx ((step.dampingMatrix.getD 8 #[]).getD 8 0.0) 1.0 1.0e-12)
    "Pinned top-left particle should have identity damping block for ny=3"

  assertArrayApprox (ParticleSpringSystem.particleState 0 step.vHat) #[0.0, 0.0, 0.0] 1.0e-12
    "Pinned bottom-left velocity prediction should be clamped to zero"
  assertArrayApprox (ParticleSpringSystem.particleState 2 step.vHat) #[0.0, 0.0, 0.0] 1.0e-12
    "Pinned top-left velocity prediction should be clamped to zero"
  assertArrayApprox (ParticleSpringSystem.particleState 4 step.vHat) #[0.0, 0.0, -0.0981] 1.0e-12
    "Free center particle should receive explicit gravity in v_hat"
  assertArrayApprox step.dampingDv (Array.replicate p.positionDim 0.0) 1.0e-12
    "At the flat rest state, implicit damping should not change velocity"
  assertArrayApprox (ParticleSpringSystem.particleState 4 step.nextState.q)
    #[0.5, 0.5, -0.000981] 1.0e-12
    "Discrete position update should use q_n + dt * v_{n+1}"
  LeanTest.assertTrue (step.move.kind == SkeletonMoveKind.localSchurBlock &&
      step.move.exactness == MoveExactness.exact &&
      step.move.label.contains "implicit damping")
    "Discrete update should expose the implicit damping solve as an exact local Schur block"

@[test]
def testDiscreteDampingSolveSatisfiesAssembledLinearSystem : IO Unit := do
  let p : ClothParams := { params with nx := 3, ny := 3, spacing := 0.5, dt := 0.01 }
  let x0 := defaultState p
  let v :=
    ParticleSpringSystem.setParticleState x0.v 4 #[1.0, 0.0, 0.0]
  let x := { x0 with v := v }
  let step ← assertOk (discreteStep? p x)
    "cloth discrete update with nonzero damping"
  let lhs := FloatMatrix.matVec step.dampingMatrix step.dampingDv
  let rhs := FloatArray.scale p.dt step.dampingForces
  assertArrayApprox lhs rhs 1.0e-9
    "Implicit damping update should solve H * dv = dt * damping_force"
  LeanTest.assertTrue
    (FloatArray.maxAbsDiff step.dampingDv (Array.replicate p.positionDim 0.0) > 1.0e-9)
    "A nonzero relative velocity should produce a nonzero implicit damping correction"
  assertArrayApprox (ParticleSpringSystem.particleState 0 step.dampingDv) #[0.0, 0.0, 0.0] 1.0e-12
    "Pinned bottom-left damping correction should stay zero"
  assertArrayApprox (ParticleSpringSystem.particleState 2 step.dampingDv) #[0.0, 0.0, 0.0] 1.0e-12
    "Pinned top-left damping correction should stay zero"

@[test]
def testDiscreteUpdateRejectsNearlyCoincidentSpringParticles : IO Unit := do
  let p : ClothParams := { params with nx := 2, ny := 2, spacing := 0.5, dt := 0.01 }
  let x0 := defaultState p
  let q :=
    ParticleSpringSystem.setParticleState x0.q 1
      (ParticleSpringSystem.particleState 0 x0.q)
  let msg ← assertError (discreteStep? p { x0 with q := q })
    "cloth discrete update with overlapping spring particles"
  LeanTest.assertTrue (msg.contains "nearly coincident")
    s!"Coincident spring particles should be rejected like Drake, got {msg}"

@[test]
def testEndToEndTraceAndShortRolloutExecute : IO Unit := do
  let p : ClothParams := { params with nx := 3, ny := 3, spacing := 0.2, dt := 0.01 }
  let result ← assertOk (buildEndToEnd? p) "mass-spring cloth end-to-end"
  let _ ← assertOk result.trace.validate? "mass-spring cloth trace validation"
  LeanTest.assertEqual result.moves.size 4
    "Accepted interval plus full-physics support and mass solve should produce four moves"
  LeanTest.assertTrue (result.moves.any (fun move =>
      move.label == "full-physics-step:mass-spring cloth primitive full physics"))
    "End-to-end cloth result should expose the primitive full-physics mass solve"
  LeanTest.assertTrue result.initialState.isFinite
    s!"Initial cloth state should be finite, got {reprStr result.initialState}"
  LeanTest.assertTrue result.oneStepState.isFinite
    s!"One-step cloth state should be finite, got {reprStr result.oneStepState}"
  LeanTest.assertTrue result.discreteStep.nextState.isFinite
    s!"Discrete cloth step should be finite, got {reprStr result.discreteStep.nextState}"
  LeanTest.assertTrue result.rolloutState.isFinite
    s!"Short rollout cloth state should be finite, got {reprStr result.rolloutState}"
  LeanTest.assertEqual result.oneStepState.q result.initialState.q
    "The first explicit Euler position step should be unchanged because initial velocity is zero"
  LeanTest.assertTrue (approx (result.oneStepState.v.getD 14 0.0) (-0.0981) 1.0e-12)
    s!"One step should integrate gravity into the center particle velocity, got {result.oneStepState.v.getD 14 0.0}"
  LeanTest.assertTrue (result.rolloutState.q != result.initialState.q)
    "A two-step rollout should move after the first step creates velocity"

end Tests.EventSkeletonMassSpringClothExample
