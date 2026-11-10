import Tyr.EventSkeleton.Contact
import Tyr.EventSkeleton.Physics

/-!
# Tyr.EventSkeleton.Manipulator

Dense manipulator-equation helpers for Drake-style examples.

The primitive here is the standard second-order form

`M(q) vdot = tau - bias(q, v)`.

Examples can build `M`, generalized forces, and bias terms however they like:
from hand-written equations, a URDF provider, or a future multibody compiler.
-/

namespace Tyr.EventSkeleton

structure ManipulatorEquation where
  massMatrix : Array (Array Float)
  qdot : Array Float
  generalizedForces : Array Float
  biasForces : Array Float := #[]
  label : String := ""
  deriving Repr, Inhabited

structure ManipulatorDerivative where
  qdot : Array Float
  vdot : Array Float
  rhs : Array Float
  massMatrix : Array (Array Float)
  label : String := ""
  deriving Repr, Inhabited

structure CouplerConstraint where
  constrainedName : String := ""
  referenceName : String := ""
  gearRatio : Float := 1.0
  offset : Float := 0.0
  label : String := ""
  deriving Repr, Inhabited

structure JointActuatorModel where
  gearRatio : Float := 1.0
  rotorInertia : Float := 0.0
  label : String := ""
  deriving Repr, Inhabited

structure JointTorqueControllerGains where
  stiffness : Array Float
  dampingRatio : Array Float
  label : String := ""
  deriving Repr, Inhabited

structure JointTorqueControllerInput where
  estimatedState : Array Float
  desiredState : Array Float
  commandedTorque : Array Float
  gravityCompensationTorque : Array Float
  massMatrixDiagonal : Array Float
  label : String := ""
  deriving Repr, Inhabited

structure JointTorqueControllerOutput where
  position : Array Float
  velocity : Array Float
  desiredPosition : Array Float
  desiredVelocity : Array Float
  commandedTorque : Array Float
  gravityCompensationTorque : Array Float
  springTorque : Array Float
  dampingGains : Array Float
  dampingTorque : Array Float
  controlTorque : Array Float
  deriving Repr, Inhabited

structure JointPidGains where
  kp : Array Float
  kd : Array Float
  ki : Array Float := #[]
  label : String := ""
  deriving Repr, Inhabited

structure JointPidInput where
  estimatedState : Array Float
  desiredState : Array Float
  integralError : Array Float := #[]
  label : String := ""
  deriving Repr, Inhabited

structure JointPidOutput where
  position : Array Float
  velocity : Array Float
  desiredPosition : Array Float
  desiredVelocity : Array Float
  positionError : Array Float
  velocityError : Array Float
  integralError : Array Float
  feedback : Array Float
  deriving Repr, Inhabited

/-!
## Full multibody plant primitive

Some Drake examples are not small closed-form manipulators.  The metadata below
records the parser-built `MultibodyPlant` boundary, while `FullPhysicsPrimitives`
below is the preferred implementation path for the plant dynamics.  An opaque
`simulatorMove` remains useful as a fallback boundary, but ports should compile
mass, bias, actuation, and contact terms into primitive dynamics whenever those
terms are available.
-/

inductive DiscreteContactApproximation where
  | sap
  | similar
  | lagged
  deriving Repr, BEq, Inhabited

namespace DiscreteContactApproximation

def label : DiscreteContactApproximation → String
  | .sap => "sap"
  | .similar => "similar"
  | .lagged => "lagged"

def fromString? (s : String) : Except String DiscreteContactApproximation :=
  if s == "sap" then
    .ok .sap
  else if s == "similar" then
    .ok .similar
  else if s == "lagged" then
    .ok .lagged
  else
    .error s!"unsupported discrete contact approximation {s}"

end DiscreteContactApproximation

structure MultibodyPlantConfigPrimitive where
  timeStep : Float := 0.0
  penetrationAllowance : Float := 1.0e-3
  stictionTolerance : Float := 1.0e-3
  contactApproximation : DiscreteContactApproximation := .sap
  deriving Repr, Inhabited

namespace MultibodyPlantConfigPrimitive

def isDiscrete (config : MultibodyPlantConfigPrimitive) : Bool :=
  config.timeStep > 0.0

def isContinuous (config : MultibodyPlantConfigPrimitive) : Bool :=
  config.timeStep == 0.0

def validate? (config : MultibodyPlantConfigPrimitive) : Except String Unit := do
  if !config.timeStep.isFinite || config.timeStep < 0.0 then
    .error s!"multibody plant time_step must be nonnegative and finite, got {config.timeStep}"
  if !config.penetrationAllowance.isFinite || config.penetrationAllowance < 0.0 then
    .error s!"multibody plant penetration allowance must be nonnegative and finite, got {config.penetrationAllowance}"
  if !config.stictionTolerance.isFinite || config.stictionTolerance <= 0.0 then
    .error s!"multibody plant stiction tolerance must be positive and finite, got {config.stictionTolerance}"

end MultibodyPlantConfigPrimitive

inductive FloatingBaseCoordinateConvention where
  | quaternion
  | rpy
  deriving Repr, BEq, Inhabited

namespace FloatingBaseCoordinateConvention

def positionCount : FloatingBaseCoordinateConvention → Nat
  | .quaternion => 7
  | .rpy => 6

def velocityCount : FloatingBaseCoordinateConvention → Nat
  | .quaternion => 6
  | .rpy => 6

end FloatingBaseCoordinateConvention

structure FloatingBaseModelInstance where
  bodyName : String
  convention : FloatingBaseCoordinateConvention := .quaternion
  floatingPositionsStart : Nat := 0
  floatingVelocitiesStartInV : Nat := 0
  deriving Repr, Inhabited

namespace FloatingBaseModelInstance

def validate? (base : FloatingBaseModelInstance)
    (numPositions numVelocities : Nat) : Except String Unit := do
  if base.floatingPositionsStart + base.convention.positionCount > numPositions then
    .error s!"floating base {base.bodyName}: position span starts at {base.floatingPositionsStart} and exceeds num_positions={numPositions}"
  if base.floatingVelocitiesStartInV + base.convention.velocityCount > numVelocities then
    .error s!"floating base {base.bodyName}: velocity span starts at {base.floatingVelocitiesStartInV} and exceeds num_velocities={numVelocities}"

end FloatingBaseModelInstance

structure FullMultibodyPlantModel where
  modelName : String
  modelUri : String
  numPositions : Nat
  numVelocities : Nat
  numActuatedDofs : Nat
  floatingBases : Array FloatingBaseModelInstance := #[]
  finalized : Bool := true
  label : String := ""
  deriving Repr, Inhabited

namespace FullMultibodyPlantModel

def stateDim (model : FullMultibodyPlantModel) : Nat :=
  model.numPositions + model.numVelocities

def validate? (model : FullMultibodyPlantModel) : Except String Unit := do
  if model.modelUri.isEmpty then
    .error s!"full multibody plant model {model.label}: model URI is empty"
  if model.numPositions == 0 then
    .error s!"full multibody plant model {model.label}: num_positions must be positive"
  if model.numVelocities == 0 then
    .error s!"full multibody plant model {model.label}: num_velocities must be positive"
  if model.numActuatedDofs > model.numVelocities then
    .error s!"full multibody plant model {model.label}: num_actuated_dofs {model.numActuatedDofs} exceeds num_velocities {model.numVelocities}"
  if !model.finalized then
    .error s!"full multibody plant model {model.label}: plant must be finalized before simulation"
  for base in model.floatingBases do
    base.validate? model.numPositions model.numVelocities

end FullMultibodyPlantModel

/-!
## Parser-produced multibody model quantities

The parser is also a primitive boundary.  Drake examples often test only the
quantities produced by `Parser.AddModelsFromUrl` followed by `plant.Finalize`:
model instances, joints, bodies, actuators, and per-instance state dimensions.
Recording that output here keeps parser-backed full-physics examples factored
through reusable observations instead of hard-coding them in each port.
-/

structure ParsedModelInstanceQuantities where
  name : String := ""
  modelUri : String := ""
  numPositions : Nat
  numVelocities : Nat
  deriving Repr, BEq, Inhabited

namespace ParsedModelInstanceQuantities

def validate? (inst : ParsedModelInstanceQuantities)
    (label : String := "") : Except String Unit := do
  if inst.name.isEmpty then
    .error s!"parsed model instance {label}: name cannot be empty"
  if inst.modelUri.isEmpty then
    .error s!"parsed model instance {inst.name}: model URI cannot be empty"

end ParsedModelInstanceQuantities

structure ParsedMultibodyPlantQuantities where
  modelUris : Array String := #[]
  builtInModelInstances : Nat := 2
  numModelInstances : Nat := 0
  numActuators : Nat := 0
  numJoints : Nat := 0
  numBodies : Nat := 0
  modelInstances : Array ParsedModelInstanceQuantities := #[]
  finalized : Bool := true
  label : String := ""
  deriving Repr, BEq, Inhabited

namespace ParsedMultibodyPlantQuantities

def expectedMinimumModelInstances (quantities : ParsedMultibodyPlantQuantities) :
    Nat :=
  quantities.builtInModelInstances + quantities.modelInstances.size

def validate? (quantities : ParsedMultibodyPlantQuantities) :
    Except String Unit := do
  if quantities.modelUris.isEmpty then
    .error s!"parsed multibody plant {quantities.label}: no model URIs"
  for uri in quantities.modelUris do
    if uri.isEmpty then
      .error s!"parsed multibody plant {quantities.label}: empty model URI"
  if quantities.numModelInstances < quantities.expectedMinimumModelInstances then
    .error s!"parsed multibody plant {quantities.label}: num_model_instances {quantities.numModelInstances} is smaller than built-ins plus parsed models {quantities.expectedMinimumModelInstances}"
  if quantities.numBodies == 0 then
    .error s!"parsed multibody plant {quantities.label}: num_bodies must include at least world"
  if !quantities.finalized then
    .error s!"parsed multibody plant {quantities.label}: plant must be finalized before quantities are used"
  for inst in quantities.modelInstances do
    inst.validate? quantities.label

def parserMove
    (target : VertexId)
    (label : String := "Parser.AddModelsFromUrl + MultibodyPlant.Finalize") :
    SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[target]
    exactness := .exact
    label := label
  }

end ParsedMultibodyPlantQuantities

structure HalfSpaceContactEnvironment where
  visualName : String := "GroundVisualGeometry"
  collisionName : String := "GroundCollisionGeometry"
  friction : CoulombFriction := { staticFriction := 1.0, dynamicFriction := 1.0 }
  deriving Repr, Inhabited

namespace HalfSpaceContactEnvironment

def validate? (ground : HalfSpaceContactEnvironment) : Except String Unit := do
  ground.friction.validate? ground.collisionName

end HalfSpaceContactEnvironment

structure FullMultibodyPlantStep where
  model : FullMultibodyPlantModel
  config : MultibodyPlantConfigPrimitive
  q0 : Array Float
  v0 : Array Float
  actuation : Array Float
  t0 : Float := 0.0
  t1 : Float
  ground? : Option HalfSpaceContactEnvironment := none
  label : String := ""
  deriving Repr, Inhabited

namespace FullMultibodyPlantStep

def initialState (step : FullMultibodyPlantStep) : Array Float :=
  step.q0 ++ step.v0

def hasContactEnvironment (step : FullMultibodyPlantStep) : Bool :=
  step.ground?.isSome

private def validateFiniteVector? (xs : Array Float) (label : String) :
    Except String Unit := do
  for i in [:xs.size] do
    if !(xs[i]!).isFinite then
      .error s!"{label}[{i}] must be finite, got {xs[i]!}"

def validate? (step : FullMultibodyPlantStep) : Except String Unit := do
  step.model.validate?
  step.config.validate?
  if step.q0.size != step.model.numPositions then
    .error s!"full plant step {step.label}: q0 size {step.q0.size} != num_positions {step.model.numPositions}"
  if step.v0.size != step.model.numVelocities then
    .error s!"full plant step {step.label}: v0 size {step.v0.size} != num_velocities {step.model.numVelocities}"
  if step.actuation.size != step.model.numActuatedDofs then
    .error s!"full plant step {step.label}: actuation size {step.actuation.size} != num_actuated_dofs {step.model.numActuatedDofs}"
  validateFiniteVector? step.q0 s!"full plant step {step.label}: q0"
  validateFiniteVector? step.v0 s!"full plant step {step.label}: v0"
  validateFiniteVector? step.actuation s!"full plant step {step.label}: actuation"
  if !step.t0.isFinite || !step.t1.isFinite then
    .error s!"full plant step {step.label}: times must be finite"
  if step.t1 < step.t0 then
    .error s!"full plant step {step.label}: t1 {step.t1} is before t0 {step.t0}"
  match step.ground? with
  | some ground => ground.validate?
  | none => pure ()

def simulatorMove (intervalVertex : VertexId) (label : String := "full-multibody-plant-advance") :
    SkeletonMove :=
  {
    kind := .intervalAdjoint
    targets := #[intervalVertex]
    exactness := .exact
    label := label
  }

end FullMultibodyPlantStep

structure GeneralizedActuationMap where
  velocityDim : Nat
  actuatorVelocityIndices : Array Nat
  label : String := ""
  deriving Repr, Inhabited

namespace GeneralizedActuationMap

private def natRangeFrom (start count : Nat) : Array Nat := Id.run do
  let mut out : Array Nat := #[]
  for i in [:count] do
    out := out.push (start + i)
  return out

private def validateFiniteVector? (xs : Array Float) (label : String) :
    Except String Unit := do
  for i in [:xs.size] do
    if !(xs[i]!).isFinite then
      .error s!"{label}[{i}] must be finite, got {xs[i]!}"

def identity (velocityDim : Nat)
    (label : String := "identity generalized actuation map") :
    GeneralizedActuationMap :=
  {
    velocityDim := velocityDim
    actuatorVelocityIndices := natRangeFrom 0 velocityDim
    label := label
  }

def contiguousOffset (velocityDim actuatorCount offset : Nat)
    (label : String := "contiguous generalized actuation map") :
    GeneralizedActuationMap :=
  {
    velocityDim := velocityDim
    actuatorVelocityIndices := natRangeFrom offset actuatorCount
    label := label
  }

def validate? (map : GeneralizedActuationMap)
    (actuatorCount? : Option Nat := none) : Except String Unit := do
  if map.velocityDim == 0 then
    .error s!"generalized actuation map {map.label}: velocity dimension must be positive"
  match actuatorCount? with
  | some actuatorCount =>
      if map.actuatorVelocityIndices.size != actuatorCount then
        .error s!"generalized actuation map {map.label}: actuator index count {map.actuatorVelocityIndices.size} != actuator count {actuatorCount}"
  | none => pure ()
  for i in [:map.actuatorVelocityIndices.size] do
    let idx := map.actuatorVelocityIndices[i]!
    if idx >= map.velocityDim then
      .error s!"generalized actuation map {map.label}: actuator index {idx} is outside velocity dimension {map.velocityDim}"
    for j in [:map.actuatorVelocityIndices.size] do
      if i < j && idx == map.actuatorVelocityIndices[j]! then
        .error s!"generalized actuation map {map.label}: duplicate actuator velocity index {idx}"

def generalizedForces? (map : GeneralizedActuationMap)
    (actuation : Array Float) : Except String (Array Float) := do
  map.validate? (some actuation.size)
  validateFiniteVector? actuation s!"generalized actuation map {map.label}: actuation"
  let mut tau := Array.replicate map.velocityDim 0.0
  for i in [:actuation.size] do
    tau := tau.set! (map.actuatorVelocityIndices[i]!) (actuation[i]!)
  pure tau

def generalizedForcesFromStep? (map : GeneralizedActuationMap)
    (step : FullMultibodyPlantStep) : Except String (Array Float) := do
  step.validate?
  if map.velocityDim != step.model.numVelocities then
    .error s!"generalized actuation map {map.label}: velocity dimension {map.velocityDim} != plant velocities {step.model.numVelocities}"
  map.generalizedForces? step.actuation

end GeneralizedActuationMap

namespace ManipulatorDerivative

def stateDerivative (d : ManipulatorDerivative) : Array Float :=
  d.qdot ++ d.vdot

end ManipulatorDerivative

namespace CouplerConstraint

def validate? (constraint : CouplerConstraint) : Except String Unit := do
  if !constraint.gearRatio.isFinite then
    .error s!"coupler constraint {constraint.label}: gear ratio must be finite, got {constraint.gearRatio}"
  else if !constraint.offset.isFinite then
    .error s!"coupler constraint {constraint.label}: offset must be finite, got {constraint.offset}"
  else
    .ok ()

def constrainedFromReference
    (constraint : CouplerConstraint)
    (referencePosition : Float) : Float :=
  constraint.gearRatio * referencePosition + constraint.offset

def referenceFromConstrained?
    (constraint : CouplerConstraint)
    (constrainedPosition : Float) : Except String Float := do
  constraint.validate?
  if constraint.gearRatio == 0.0 then
    .error s!"coupler constraint {constraint.label}: cannot invert zero gear ratio"
  pure ((constrainedPosition - constraint.offset) / constraint.gearRatio)

def positionResidual
    (constraint : CouplerConstraint)
    (constrainedPosition referencePosition : Float) : Float :=
  constrainedPosition - constraint.constrainedFromReference referencePosition

def velocityResidual
    (constraint : CouplerConstraint)
    (constrainedVelocity referenceVelocity : Float) : Float :=
  constrainedVelocity - constraint.gearRatio * referenceVelocity

/--
Virtual-work conversion for a symmetric grasp with equal and opposite normal
forces on the coupled bodies.  This is the relation used by Drake's simple
gripper example: `U = G * (1 - rho)`.
-/
def opposingGripActuationForce
    (constraint : CouplerConstraint)
    (gripForce : Float) : Float :=
  gripForce * (1.0 - constraint.gearRatio)

end CouplerConstraint

namespace JointActuatorModel

def validate? (model : JointActuatorModel) : Except String Unit := do
  if !model.gearRatio.isFinite || model.gearRatio <= 0.0 then
    .error s!"joint actuator model {model.label}: gear ratio must be positive and finite, got {model.gearRatio}"
  if !model.rotorInertia.isFinite || model.rotorInertia < 0.0 then
    .error s!"joint actuator model {model.label}: rotor inertia must be nonnegative and finite, got {model.rotorInertia}"

def reflectedInertia? (model : JointActuatorModel) : Except String Float := do
  model.validate?
  pure (model.rotorInertia * model.gearRatio * model.gearRatio)

end JointActuatorModel

namespace JointTorqueControllerGains

def dof (gains : JointTorqueControllerGains) : Nat :=
  gains.stiffness.size

def validate? (gains : JointTorqueControllerGains) : Except String Unit := do
  let n := gains.dof
  if n == 0 then
    .error s!"joint torque controller gains {gains.label}: empty stiffness vector"
  if gains.dampingRatio.size != n then
    .error s!"joint torque controller gains {gains.label}: damping ratio size {gains.dampingRatio.size} != stiffness size {n}"
  for i in [:n] do
    let k := gains.stiffness[i]!
    let d := gains.dampingRatio[i]!
    if !k.isFinite || k < 0.0 then
      .error s!"joint torque controller gains {gains.label}: stiffness[{i}] must be nonnegative and finite, got {k}"
    if !d.isFinite || d < 0.0 then
      .error s!"joint torque controller gains {gains.label}: dampingRatio[{i}] must be nonnegative and finite, got {d}"

end JointTorqueControllerGains

namespace JointTorqueControllerInput

def dof (input : JointTorqueControllerInput) : Nat :=
  input.massMatrixDiagonal.size

private def validateVectorSize?
    (xs : Array Float)
    (expected : Nat)
    (field label : String) : Except String Unit := do
  if xs.size != expected then
    .error s!"joint torque controller input {label}: {field} size {xs.size} != expected {expected}"
  for i in [:xs.size] do
    let x := xs[i]!
    if !x.isFinite then
      .error s!"joint torque controller input {label}: {field}[{i}] must be finite, got {x}"

def validate? (input : JointTorqueControllerInput)
    (gains : JointTorqueControllerGains) : Except String Unit := do
  gains.validate?
  let n := input.dof
  if n != gains.dof then
    .error s!"joint torque controller input {input.label}: mass matrix diagonal size {n} != gains dof {gains.dof}"
  validateVectorSize? input.estimatedState (2 * n) "estimatedState" input.label
  validateVectorSize? input.desiredState (2 * n) "desiredState" input.label
  validateVectorSize? input.commandedTorque n "commandedTorque" input.label
  validateVectorSize? input.gravityCompensationTorque n "gravityCompensationTorque" input.label
  validateVectorSize? input.massMatrixDiagonal n "massMatrixDiagonal" input.label
  for i in [:n] do
    let h := input.massMatrixDiagonal[i]!
    if h < 0.0 then
      .error s!"joint torque controller input {input.label}: massMatrixDiagonal[{i}] must be nonnegative, got {h}"

def positions (input : JointTorqueControllerInput) : Array Float :=
  input.estimatedState.extract 0 input.dof

def velocities (input : JointTorqueControllerInput) : Array Float :=
  input.estimatedState.extract input.dof (2 * input.dof)

def desiredPositions (input : JointTorqueControllerInput) : Array Float :=
  input.desiredState.extract 0 input.dof

def desiredVelocities (input : JointTorqueControllerInput) : Array Float :=
  input.desiredState.extract input.dof (2 * input.dof)

def evaluate? (input : JointTorqueControllerInput)
    (gains : JointTorqueControllerGains) : Except String JointTorqueControllerOutput := do
  input.validate? gains
  let n := input.dof
  let q := input.positions
  let v := input.velocities
  let qd := input.desiredPositions
  let vd := input.desiredVelocities
  let mut spring : Array Float := #[]
  let mut dampingGains : Array Float := #[]
  let mut damping : Array Float := #[]
  let mut control := input.commandedTorque
  control := FloatArray.add control input.gravityCompensationTorque
  for i in [:n] do
    let k := gains.stiffness[i]!
    let ratio := gains.dampingRatio[i]!
    let springTorque := k * (qd.getD i 0.0 - q.getD i 0.0)
    let criticalDamping := 2.0 * Float.sqrt (input.massMatrixDiagonal[i]! * k)
    let gain := ratio * criticalDamping
    let dampingTorque := -gain * v.getD i 0.0
    spring := spring.push springTorque
    dampingGains := dampingGains.push gain
    damping := damping.push dampingTorque
  control := FloatArray.add control spring
  control := FloatArray.add control damping
  pure {
    position := q
    velocity := v
    desiredPosition := qd
    desiredVelocity := vd
    commandedTorque := input.commandedTorque
    gravityCompensationTorque := input.gravityCompensationTorque
    springTorque := spring
    dampingGains := dampingGains
    dampingTorque := damping
    controlTorque := control
  }

end JointTorqueControllerInput

namespace JointPidGains

def dof (gains : JointPidGains) : Nat :=
  gains.kp.size

def kiVector (gains : JointPidGains) : Array Float :=
  if gains.ki.isEmpty then
    Array.replicate gains.dof 0.0
  else
    gains.ki

def validate? (gains : JointPidGains) : Except String Unit := do
  let n := gains.dof
  if n == 0 then
    .error s!"joint PID gains {gains.label}: empty kp vector"
  if gains.kd.size != n then
    .error s!"joint PID gains {gains.label}: kd size {gains.kd.size} != kp size {n}"
  if !gains.ki.isEmpty && gains.ki.size != n then
    .error s!"joint PID gains {gains.label}: ki size {gains.ki.size} != kp size {n}"
  let ki := gains.kiVector
  for i in [:n] do
    let kp := gains.kp[i]!
    let kd := gains.kd[i]!
    let kiVal := ki[i]!
    if !kp.isFinite || kp < 0.0 then
      .error s!"joint PID gains {gains.label}: kp[{i}] must be nonnegative and finite, got {kp}"
    if !kd.isFinite || kd < 0.0 then
      .error s!"joint PID gains {gains.label}: kd[{i}] must be nonnegative and finite, got {kd}"
    if !kiVal.isFinite || kiVal < 0.0 then
      .error s!"joint PID gains {gains.label}: ki[{i}] must be nonnegative and finite, got {kiVal}"

end JointPidGains

namespace JointPidInput

private def validateVectorSize?
    (xs : Array Float)
    (expected : Nat)
    (field label : String) : Except String Unit := do
  if xs.size != expected then
    .error s!"joint PID input {label}: {field} size {xs.size} != expected {expected}"
  for i in [:xs.size] do
    let x := xs[i]!
    if !x.isFinite then
      .error s!"joint PID input {label}: {field}[{i}] must be finite, got {x}"

def integralVector (input : JointPidInput) (n : Nat) : Array Float :=
  if input.integralError.isEmpty then
    Array.replicate n 0.0
  else
    input.integralError

def validate? (input : JointPidInput)
    (gains : JointPidGains) : Except String Unit := do
  gains.validate?
  let n := gains.dof
  validateVectorSize? input.estimatedState (2 * n) "estimatedState" input.label
  validateVectorSize? input.desiredState (2 * n) "desiredState" input.label
  if !input.integralError.isEmpty then
    validateVectorSize? input.integralError n "integralError" input.label

def positions (input : JointPidInput) (n : Nat) : Array Float :=
  input.estimatedState.extract 0 n

def velocities (input : JointPidInput) (n : Nat) : Array Float :=
  input.estimatedState.extract n (2 * n)

def desiredPositions (input : JointPidInput) (n : Nat) : Array Float :=
  input.desiredState.extract 0 n

def desiredVelocities (input : JointPidInput) (n : Nat) : Array Float :=
  input.desiredState.extract n (2 * n)

def evaluate? (input : JointPidInput)
    (gains : JointPidGains) : Except String JointPidOutput := do
  input.validate? gains
  let n := gains.dof
  let q := input.positions n
  let v := input.velocities n
  let qd := input.desiredPositions n
  let vd := input.desiredVelocities n
  let integral := input.integralVector n
  let ki := gains.kiVector
  let mut positionError : Array Float := #[]
  let mut velocityError : Array Float := #[]
  let mut feedback : Array Float := #[]
  for i in [:n] do
    let qErr := qd[i]! - q[i]!
    let vErr := vd[i]! - v[i]!
    positionError := positionError.push qErr
    velocityError := velocityError.push vErr
    feedback :=
      feedback.push
        (gains.kp[i]! * qErr + gains.kd[i]! * vErr + ki[i]! * integral[i]!)
  pure {
    position := q
    velocity := v
    desiredPosition := qd
    desiredVelocity := vd
    positionError := positionError
    velocityError := velocityError
    integralError := integral
    feedback := feedback
  }

end JointPidInput

namespace ManipulatorEquation

def dof (eq : ManipulatorEquation) : Nat :=
  eq.qdot.size

def biasVector? (eq : ManipulatorEquation) : Except String (Array Float) := do
  let n := eq.dof
  if eq.biasForces.isEmpty then
    pure (Array.replicate n 0.0)
  else if eq.biasForces.size == n then
    pure eq.biasForces
  else
    .error s!"manipulator equation {eq.label}: bias size {eq.biasForces.size} != dof {n}"

def validate? (eq : ManipulatorEquation) : Except String Unit := do
  let n := eq.dof
  if n == 0 then
    .error s!"manipulator equation {eq.label}: empty velocity vector"
  DenseLinearAlgebra.validateSquare? eq.massMatrix n s!"manipulator mass matrix {eq.label}"
  if eq.generalizedForces.size != n then
    .error s!"manipulator equation {eq.label}: generalized force size {eq.generalizedForces.size} != dof {n}"
  discard eq.biasVector?

def rhs? (eq : ManipulatorEquation) : Except String (Array Float) := do
  eq.validate?
  let bias ← eq.biasVector?
  pure (FloatArray.sub eq.generalizedForces bias)

def solve? (eq : ManipulatorEquation) : Except String ManipulatorDerivative := do
  let rhs ← eq.rhs?
  let vdot ← DenseLinearAlgebra.solveLinear? eq.massMatrix rhs
  pure {
    qdot := eq.qdot
    vdot := vdot
    rhs := rhs
    massMatrix := eq.massMatrix
    label := eq.label
  }

end ManipulatorEquation

/-!
## Full physics assembly from existing primitives

This layer is deliberately a composition, not a new backend.  A model-specific
provider dynamically computes contact candidates, runtime support selection
chooses the active constraints, a force provider returns scalar contact forces,
and the shared manipulator equation solves the mass-matrix dynamics.
-/

/--
One already-evaluated generalized force produced by another primitive.

This keeps non-actuator effects such as bushings, spring graphs, force-density
integrals, and exact constraint-force blocks visible at the full-physics
boundary instead of folding them into actuator torques.
-/
structure GeneralizedForceContribution where
  force : Array Float := #[]
  source : String := ""
  label : String := ""
  deriving Repr, Inhabited

namespace GeneralizedForceContribution

def ofForce
    (force : Array Float)
    (label : String := "")
    (source : String := "") : GeneralizedForceContribution :=
  { force := force, source := source, label := label }

def validate? (velocityDim : Nat) (contribution : GeneralizedForceContribution) :
    Except String Unit := do
  if contribution.force.size != velocityDim then
    .error s!"generalized force contribution {contribution.label}: force size {contribution.force.size} != velocity dimension {velocityDim}"
  for i in [:contribution.force.size] do
    if !(contribution.force[i]!).isFinite then
      .error s!"generalized force contribution {contribution.label}: force[{i}] must be finite, got {contribution.force[i]!}"

end GeneralizedForceContribution

def sumGeneralizedForceContributions
    (velocityDim : Nat)
    (contributions : Array GeneralizedForceContribution) : Array Float :=
  contributions.foldl
    (fun acc contribution => FloatArray.add acc contribution.force)
    (Array.replicate velocityDim 0.0)

/-!
## Bilateral acceleration constraints

Closed kinematic loops, welds, couplers, and ideal ball constraints are not
contacts: their multipliers are signed and they enforce rows of `J vdot =
target`.  The primitive below is the dense local Schur complement for those
constraints.  It is intentionally small, but it keeps the full-physics boundary
honest: examples can expose loop closures as constraints instead of encoding
them as zero-force contact candidates.
-/

structure BilateralConstraintPrimitive where
  id : Nat := 0
  jacobian : Array (Array Float) := #[]
  targetAcceleration : Array Float := #[]
  label : String := ""
  deriving Repr, Inhabited

structure BilateralConstraintSolve where
  constraints : Array BilateralConstraintPrimitive := #[]
  jacobian : Array (Array Float) := #[]
  targetAcceleration : Array Float := #[]
  delassus : Array (Array Float) := #[]
  multiplierRhs : Array Float := #[]
  multipliers : Array Float := #[]
  generalizedConstraintForce : Array Float := #[]
  freeAcceleration : Array Float := #[]
  acceleration : Array Float := #[]
  constraintAccelerationBefore : Array Float := #[]
  constraintAccelerationAfter : Array Float := #[]
  label : String := ""
  deriving Repr, Inhabited

namespace BilateralConstraintPrimitive

def rowCount (constraint : BilateralConstraintPrimitive) : Nat :=
  constraint.jacobian.size

private def validateFiniteVector? (xs : Array Float) (label : String) :
    Except String Unit := do
  for i in [:xs.size] do
    if !(xs[i]!).isFinite then
      .error s!"{label}[{i}] must be finite, got {xs[i]!}"

def targetVector? (constraint : BilateralConstraintPrimitive) :
    Except String (Array Float) := do
  if constraint.targetAcceleration.isEmpty then
    pure (Array.replicate constraint.rowCount 0.0)
  else if constraint.targetAcceleration.size == constraint.rowCount then
    validateFiniteVector? constraint.targetAcceleration
      s!"bilateral constraint {constraint.label}: targetAcceleration"
    pure constraint.targetAcceleration
  else
    .error s!"bilateral constraint {constraint.label}: targetAcceleration size {constraint.targetAcceleration.size} != row count {constraint.rowCount}"

def validate? (velocityDim : Nat) (constraint : BilateralConstraintPrimitive) :
    Except String Unit := do
  if constraint.jacobian.isEmpty then
    .error s!"bilateral constraint {constraint.label}: empty Jacobian"
  for i in [:constraint.jacobian.size] do
    let row := constraint.jacobian[i]!
    if row.size != velocityDim then
      .error s!"bilateral constraint {constraint.label}: row {i} width {row.size} != velocity dimension {velocityDim}"
    validateFiniteVector? row s!"bilateral constraint {constraint.label}: jacobian row {i}"
  let _ ← constraint.targetVector?
  pure ()

end BilateralConstraintPrimitive

namespace BilateralConstraintSolve

private def maxAbs (xs : Array Float) : Float :=
  xs.foldl (fun acc x => max acc (Float.abs x)) 0.0

private structure BilateralConstraintRowGroup where
  row : Array Float
  rhs : Float
  count : Nat := 1
  deriving Repr, Inhabited

private def rowMatches (tol : Float) (a b : Array Float) : Bool :=
  a.size == b.size && FloatArray.maxAbsDiff a b <= tol

def solve?
    (massMatrix : Array (Array Float))
    (baseRhs : Array Float)
    (constraints : Array BilateralConstraintPrimitive)
    (label : String := "")
    (zeroRhsTol : Float := 1.0e-10) :
    Except String BilateralConstraintSolve := do
  let n := baseRhs.size
  DenseLinearAlgebra.validateSquare? massMatrix n
    s!"bilateral constraint mass matrix {label}"
  BilateralConstraintPrimitive.validateFiniteVector? baseRhs
    s!"bilateral constraint base rhs {label}"
  let free ← DenseLinearAlgebra.solveLinear? massMatrix baseRhs
  if constraints.isEmpty then
    pure {
      constraints := constraints
      generalizedConstraintForce := Array.replicate n 0.0
      freeAcceleration := free
      acceleration := free
      label := label
    }
  else
    let mut jacobian : Array (Array Float) := #[]
    let mut target : Array Float := #[]
    for constraint in constraints do
      constraint.validate? n
      let localTarget ← constraint.targetVector?
      for row in constraint.jacobian do
        jacobian := jacobian.push row
      for value in localTarget do
        target := target.push value
    let before := FloatMatrix.matVec jacobian free
    let multiplierRhs := FloatArray.sub target before
    let minvJt ← VelocityProjection.massInverseTimesJacobianTranspose?
      massMatrix jacobian
    let delassus := FloatMatrix.matMat jacobian minvJt
    let mut groups : Array BilateralConstraintRowGroup := #[]
    let mut rowGroupIndices : Array Nat := #[]
    for i in [:jacobian.size] do
      let row := jacobian[i]!
      let rhs := multiplierRhs.getD i 0.0
      let mut found? : Option Nat := none
      for groupIndex in [:groups.size] do
        let group := groups[groupIndex]!
        if rowMatches zeroRhsTol group.row row then
          if Float.abs (group.rhs - rhs) > zeroRhsTol then
            .error s!"bilateral constraint {label}: duplicate constraint row {i} has inconsistent RHS {rhs} != {group.rhs}"
          found? := some groupIndex
      match found? with
      | some groupIndex =>
          let group := groups[groupIndex]!
          groups := groups.set! groupIndex { group with count := group.count + 1 }
          rowGroupIndices := rowGroupIndices.push groupIndex
      | none =>
          groups := groups.push { row := row, rhs := rhs, count := 1 }
          rowGroupIndices := rowGroupIndices.push (groups.size - 1)
    let compressedJacobian := groups.map (fun group => group.row)
    let compressedRhs := groups.map (fun group => group.rhs)
    let compressedMultipliers ←
      if maxAbs compressedRhs <= zeroRhsTol then
        pure (Array.replicate compressedJacobian.size 0.0)
      else
        let compressedMinvJt ←
          VelocityProjection.massInverseTimesJacobianTranspose?
            massMatrix compressedJacobian
        let compressedDelassus :=
          FloatMatrix.matMat compressedJacobian compressedMinvJt
        DenseLinearAlgebra.solveLinear? compressedDelassus compressedRhs
    let mut multipliers : Array Float := #[]
    for groupIndex in rowGroupIndices do
      let group := groups[groupIndex]!
      let multiplier := compressedMultipliers.getD groupIndex 0.0
      multipliers := multipliers.push (multiplier / group.count.toFloat)
    let generalizedConstraintForce := FloatMatrix.transposeVec jacobian multipliers
    let correction ←
      DenseLinearAlgebra.solveLinear? massMatrix generalizedConstraintForce
    let acceleration := FloatArray.add free correction
    pure {
      constraints := constraints
      jacobian := jacobian
      targetAcceleration := target
      delassus := delassus
      multiplierRhs := multiplierRhs
      multipliers := multipliers
      generalizedConstraintForce := generalizedConstraintForce
      freeAcceleration := free
      acceleration := acceleration
      constraintAccelerationBefore := before
      constraintAccelerationAfter := FloatMatrix.matVec jacobian acceleration
      label := label
    }

end BilateralConstraintSolve

structure FullPhysicsEquation where
  massMatrix : Array (Array Float)
  qdot : Array Float
  actuationForces : Array Float
  biasForces : Array Float := #[]
  generalizedForceContributions : Array GeneralizedForceContribution := #[]
  contactSupport : ContactSupport := { policy := .fullSupport }
  contactForces : Array ContactForceScalars := #[]
  bilateralConstraints : Array BilateralConstraintPrimitive := #[]
  label : String := ""
  deriving Repr, Inhabited

structure FullPhysicsResult where
  support : ContactSupport
  supportMove : SkeletonMove
  contactForces : Array ContactForceScalars
  generalizedPrimitiveForce : Array Float := #[]
  generalizedContactForce : Array Float
  generalizedConstraintForce : Array Float := #[]
  generalizedForces : Array Float
  constraintSolve? : Option BilateralConstraintSolve := none
  equation : ManipulatorEquation
  derivative : ManipulatorDerivative
  move : SkeletonMove
  deriving Repr, Inhabited

/--
Where selected contact force scalars come from when assembling a full-physics
step from primitive observations.

`.precomputed` is for model-specific exact force laws such as Rod2D's
Hunt-Crossley/Stribeck contact.  `.compliantModel` uses the reusable penalty
force provider in `Contact`.
-/
inductive ContactForceSource where
  | precomputed
  | compliantModel
  deriving Repr, BEq, Inhabited

/--
A realized full-physics input bundle at one state.

This is not a separate physics backend.  Geometry, URDF, or hand-written model
code may compute candidates dynamically in whatever representation is efficient;
the event-skeleton layer only asks for these primitive observations at the
support-selection / `J^T f` boundary.
-/
structure FullPhysicsPrimitives where
  massMatrix : Array (Array Float)
  qdot : Array Float
  actuationForces : Array Float
  biasForces : Array Float := #[]
  generalizedForceContributions : Array GeneralizedForceContribution := #[]
  contactCandidates : Array ContactCandidate := #[]
  sourceContactCandidateCount? : Option Nat := none
  supportPolicy : SupportPolicy := .fullSupport
  contactForceSource : ContactForceSource := .precomputed
  contactForces : Array ContactForceScalars := #[]
  bilateralConstraints : Array BilateralConstraintPrimitive := #[]
  compliantContactModel : CompliantContactModel := {}
  distanceTol : Float := 0.0
  tangentVelocityTol : Float := 1.0e-9
  label : String := ""
  deriving Repr, Inhabited

namespace FullPhysicsPrimitives

def velocityDim (primitive : FullPhysicsPrimitives) : Nat :=
  primitive.qdot.size

def contactCandidateSet (primitive : FullPhysicsPrimitives) : ContactCandidateSet :=
  {
    candidates := primitive.contactCandidates
    sourceCandidateCount? := primitive.sourceContactCandidateCount?
    label := s!"full-physics-contact-candidates:{primitive.label}"
  }

def support (primitive : FullPhysicsPrimitives) : ContactSupport :=
  primitive.contactCandidateSet.selectWithPolicy primitive.supportPolicy
    s!"full-physics-contact-support:{primitive.label}"
    |>.classifyCandidates primitive.distanceTol primitive.tangentVelocityTol

def selectedForceScalars? (primitive : FullPhysicsPrimitives)
    (support : ContactSupport := primitive.support) :
    Except String (Array ContactForceScalars) := do
  match primitive.contactForceSource with
  | .precomputed =>
      let selected ← support.selectedCandidates?
      if primitive.contactForces.size != selected.size then
        .error s!"full physics primitives {primitive.label}: precomputed contact force count {primitive.contactForces.size} != selected contact count {selected.size}"
      pure primitive.contactForces
  | .compliantModel =>
      primitive.compliantContactModel.forcesForSupport? support

def equation? (primitive : FullPhysicsPrimitives) : Except String FullPhysicsEquation := do
  primitive.contactCandidateSet.validate? (some primitive.velocityDim)
  let support := primitive.support
  let contactForces ← primitive.selectedForceScalars? support
  pure {
    massMatrix := primitive.massMatrix
    qdot := primitive.qdot
    actuationForces := primitive.actuationForces
    biasForces := primitive.biasForces
    generalizedForceContributions := primitive.generalizedForceContributions
    contactSupport := support
    contactForces := contactForces
    bilateralConstraints := primitive.bilateralConstraints
    label := primitive.label
  }

end FullPhysicsPrimitives

namespace FullPhysicsEquation

def velocityDim (eq : FullPhysicsEquation) : Nat :=
  eq.qdot.size

private def validateFiniteVector? (xs : Array Float) (label : String) :
    Except String Unit := do
  for i in [:xs.size] do
    if !(xs[i]!).isFinite then
      .error s!"{label}[{i}] must be finite, got {xs[i]!}"

def validate? (eq : FullPhysicsEquation) : Except String Unit := do
  let n := eq.velocityDim
  if n == 0 then
    .error s!"full physics equation {eq.label}: empty generalized velocity vector"
  DenseLinearAlgebra.validateSquare? eq.massMatrix n s!"full physics mass matrix {eq.label}"
  if eq.actuationForces.size != n then
    .error s!"full physics equation {eq.label}: actuation force size {eq.actuationForces.size} != velocity dimension {n}"
  if !eq.biasForces.isEmpty && eq.biasForces.size != n then
    .error s!"full physics equation {eq.label}: bias force size {eq.biasForces.size} != velocity dimension {n}"
  validateFiniteVector? eq.qdot s!"full physics equation {eq.label}: qdot"
  validateFiniteVector? eq.actuationForces s!"full physics equation {eq.label}: actuation"
  validateFiniteVector? eq.biasForces s!"full physics equation {eq.label}: bias"
  for contribution in eq.generalizedForceContributions do
    contribution.validate? n
  eq.contactSupport.validateSelectedIndices?
  eq.contactSupport.validateSourceCandidateCount?
  eq.contactSupport.validateJacobianWidth? n
  let selected ← eq.contactSupport.selectedCandidates?
  if eq.contactForces.size != selected.size then
    .error s!"full physics equation {eq.label}: contact force count {eq.contactForces.size} != selected contact count {selected.size}"
  for i in [:selected.size] do
    let candidate := selected[i]!
    let force := eq.contactForces[i]!
    if force.candidateId != candidate.id then
      .error s!"full physics equation {eq.label}: contact force {i} has candidate id {force.candidateId}, expected {candidate.id}"
  for constraint in eq.bilateralConstraints do
    constraint.validate? n

def generalizedContactForce? (eq : FullPhysicsEquation) :
    Except String (Array Float) := do
  eq.validate?
  let selected ← eq.contactSupport.selectedCandidates?
  let mut total := Array.replicate eq.velocityDim 0.0
  for i in [:selected.size] do
    let generalized := ContactForceScalars.generalizedForce selected[i]! eq.contactForces[i]!
    if generalized.size != eq.velocityDim then
      .error s!"full physics equation {eq.label}: generalized contact force {i} width {generalized.size} != velocity dimension {eq.velocityDim}"
    total := FloatArray.add total generalized
  pure total

def generalizedPrimitiveForce? (eq : FullPhysicsEquation) :
    Except String (Array Float) := do
  eq.validate?
  pure (sumGeneralizedForceContributions eq.velocityDim eq.generalizedForceContributions)

def manipulatorEquation? (eq : FullPhysicsEquation) :
    Except String (ManipulatorEquation × Array Float × Array Float × Array Float) := do
  let contactForce ← eq.generalizedContactForce?
  let primitiveForce ← eq.generalizedPrimitiveForce?
  let generalizedForces :=
    FloatArray.add
      (FloatArray.add eq.actuationForces primitiveForce)
      contactForce
  let manipulator : ManipulatorEquation := {
    massMatrix := eq.massMatrix
    qdot := eq.qdot
    generalizedForces := generalizedForces
    biasForces := eq.biasForces
    label := eq.label
  }
  pure (manipulator, primitiveForce, contactForce, generalizedForces)

def solve? (eq : FullPhysicsEquation) (intervalVertex : VertexId := 0) :
    Except String FullPhysicsResult := do
  let (manipulator, primitiveForce, contactForce, generalizedForces) ← eq.manipulatorEquation?
  let baseRhs ← manipulator.rhs?
  let constraintSolve ←
    BilateralConstraintSolve.solve? manipulator.massMatrix baseRhs
      eq.bilateralConstraints eq.label
  let generalizedConstraintForce := constraintSolve.generalizedConstraintForce
  let totalGeneralizedForces :=
    FloatArray.add generalizedForces generalizedConstraintForce
  let totalRhs := FloatArray.add baseRhs generalizedConstraintForce
  let totalManipulator : ManipulatorEquation :=
    { manipulator with generalizedForces := totalGeneralizedForces }
  let derivative : ManipulatorDerivative := {
    qdot := manipulator.qdot
    vdot := constraintSolve.acceleration
    rhs := totalRhs
    massMatrix := manipulator.massMatrix
    label := manipulator.label
  }
  pure {
    support := eq.contactSupport
    supportMove := {
      kind := .markMarginalize
      targets := #[intervalVertex]
      exactness := eq.contactSupport.policy.defaultExactness
      label := s!"contact-support-selection:{eq.label}"
    }
    contactForces := eq.contactForces
    generalizedPrimitiveForce := primitiveForce
    generalizedContactForce := contactForce
    generalizedConstraintForce := generalizedConstraintForce
    generalizedForces := totalGeneralizedForces
    constraintSolve? :=
      if eq.bilateralConstraints.isEmpty then none else some constraintSolve
    equation := totalManipulator
    derivative := derivative
    move := {
      kind := .intervalAdjoint
      targets := #[intervalVertex]
      exactness := .exact
      label := s!"full-physics-step:{eq.label}"
    }
  }

def fromDynamicContacts?
    (massMatrix : Array (Array Float))
    (qdot actuationForces : Array Float)
    (biasForces : Array Float := #[])
    (candidates : Array ContactCandidate)
    (selectionPolicy : SupportPolicy)
    (forceModel : CompliantContactModel)
    (label : String := "")
    (sourceCandidateCount? : Option Nat := none) :
    Except String FullPhysicsEquation := do
  ({
    massMatrix := massMatrix
    qdot := qdot
    actuationForces := actuationForces
    biasForces := biasForces
    contactCandidates := candidates
    sourceContactCandidateCount? := sourceCandidateCount?
    supportPolicy := selectionPolicy
    contactForceSource := .compliantModel
    compliantContactModel := forceModel
    label := label
  } : FullPhysicsPrimitives).equation?

def fromHydroelasticPatches?
    (massMatrix : Array (Array Float))
    (qdot actuationForces : Array Float)
    (biasForces : Array Float := #[])
    (support : HydroelasticPatchSupport)
    (label : String := "") :
    Except String FullPhysicsEquation := do
  support.validateGeometry?
  let contactSupport := support.equivalentContactSupport
  let contactForces ← support.selectedContactForces?
  pure {
    massMatrix := massMatrix
    qdot := qdot
    actuationForces := actuationForces
    biasForces := biasForces
    contactSupport := contactSupport
    contactForces := contactForces
    label := label
  }

end FullPhysicsEquation

namespace FullPhysicsPrimitives

def solve? (primitive : FullPhysicsPrimitives) (intervalVertex : VertexId := 0) :
    Except String FullPhysicsResult := do
  let equation ← primitive.equation?
  equation.solve? intervalVertex

def validateAgainstPlantStep?
    (primitive : FullPhysicsPrimitives)
    (step : FullMultibodyPlantStep) : Except String Unit := do
  step.validate?
  let equation ← primitive.equation?
  equation.validate?
  let n := step.model.numVelocities
  if primitive.velocityDim != n then
    .error s!"full physics primitives {primitive.label}: qdot size {primitive.velocityDim} != plant velocities {n}"
  if primitive.massMatrix.size != n then
    .error s!"full physics primitives {primitive.label}: mass matrix rows {primitive.massMatrix.size} != plant velocities {n}"
  if primitive.actuationForces.size != n then
    .error s!"full physics primitives {primitive.label}: generalized actuation force size {primitive.actuationForces.size} != plant velocities {n}"
  if primitive.qdot.size != step.v0.size then
    .error s!"full physics primitives {primitive.label}: qdot size {primitive.qdot.size} != plant v0 size {step.v0.size}"
  for i in [:primitive.qdot.size] do
    if primitive.qdot[i]! != step.v0[i]! then
      .error s!"full physics primitives {primitive.label}: qdot[{i}] {primitive.qdot[i]!} != plant v0[{i}] {step.v0[i]!}"

end FullPhysicsPrimitives

/-!
## State-dependent full-physics providers

The model/parser side of a full physics implementation should compute the
primitive bundle at the current state.  This provider is the reusable boundary:
URDF/SDF importers, SceneGraph queries, or hand-written examples can recompute
mass, bias, actuation, and contact candidates dynamically, while the downstream
solver still consumes only `FullPhysicsPrimitives`.
-/

structure FullPhysicsPrimitiveProvider (State : Type) where
  label : String := ""
  primitivesAt? : State → Except String FullPhysicsPrimitives

namespace FullPhysicsPrimitiveProvider

def validPrimitivesAt?
    (provider : FullPhysicsPrimitiveProvider State)
    (state : State) : Except String FullPhysicsPrimitives := do
  let primitives ← provider.primitivesAt? state
  let _ ← primitives.equation?
  pure primitives

def contactCandidateSetAt?
    (provider : FullPhysicsPrimitiveProvider State)
    (state : State) : Except String ContactCandidateSet := do
  let primitives ← provider.primitivesAt? state
  let candidates := primitives.contactCandidateSet
  candidates.validate? (some primitives.velocityDim)
  pure candidates

def supportAt?
    (provider : FullPhysicsPrimitiveProvider State)
    (state : State) : Except String ContactSupport := do
  let primitives ← provider.primitivesAt? state
  let equation ← primitives.equation?
  pure equation.contactSupport

def equationAt?
    (provider : FullPhysicsPrimitiveProvider State)
    (state : State) : Except String FullPhysicsEquation := do
  let primitives ← provider.primitivesAt? state
  primitives.equation?

def solveAt?
    (provider : FullPhysicsPrimitiveProvider State)
    (state : State)
    (intervalVertex : VertexId := 0) : Except String FullPhysicsResult := do
  let equation ← provider.equationAt? state
  equation.solve? intervalVertex

end FullPhysicsPrimitiveProvider

structure FullPlantPrimitivePhysics where
  step : FullMultibodyPlantStep
  primitives : FullPhysicsPrimitives
  intervalVertex : VertexId := 0
  label : String := ""
  deriving Repr, Inhabited

namespace FullPlantPrimitivePhysics

def validate? (physics : FullPlantPrimitivePhysics) : Except String Unit := do
  physics.primitives.validateAgainstPlantStep? physics.step

def solve? (physics : FullPlantPrimitivePhysics) :
    Except String FullPhysicsResult := do
  physics.validate?
  physics.primitives.solve? physics.intervalVertex

end FullPlantPrimitivePhysics

end Tyr.EventSkeleton
