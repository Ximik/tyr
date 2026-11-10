import Tyr.EventSkeleton.Saltation

/-!
# Tyr.EventSkeleton.Physics

Small dense physics kernels used by the event-skeleton examples.

These routines are intentionally plain Lean arrays.  They are not a high
performance multibody backend; they are the reusable numerical primitives that
let examples express mass-matrix solves, contact-space Delassus operators, and
impulse projections without depending on Drake or a separate physics engine.
-/

namespace Tyr.EventSkeleton

namespace FloatMatrix

def transpose (m : Array (Array Float)) : Array (Array Float) := Id.run do
  let cols := colCount m
  let mut out : Array (Array Float) := #[]
  for j in [:cols] do
    let mut row : Array Float := #[]
    for i in [:m.size] do
      row := row.push ((m.getD i #[]).getD j 0.0)
    out := out.push row
  return out

def matMat (a b : Array (Array Float)) : Array (Array Float) := Id.run do
  let bt := transpose b
  let mut out : Array (Array Float) := #[]
  for row in a do
    let mut outRow : Array Float := #[]
    for col in bt do
      outRow := outRow.push (FloatArray.dot row col)
    out := out.push outRow
  return out

def identity (n : Nat) : Array (Array Float) := Id.run do
  let mut rows : Array (Array Float) := #[]
  for i in [:n] do
    let mut row : Array Float := #[]
    for j in [:n] do
      row := row.push (if i == j then 1.0 else 0.0)
    rows := rows.push row
  return rows

def diagonal (diag : Array Float) : Array (Array Float) := Id.run do
  let n := diag.size
  let mut rows : Array (Array Float) := #[]
  for i in [:n] do
    let mut row : Array Float := #[]
    for j in [:n] do
      row := row.push (if i == j then diag[i]! else 0.0)
    rows := rows.push row
  return rows

end FloatMatrix

namespace DenseLinearAlgebra

def validateSquare? (a : Array (Array Float)) (n : Nat) (label : String) :
    Except String Unit := do
  if a.size != n then
    .error s!"{label}: row count {a.size} does not match expected size {n}"
  else
    for i in [:n] do
      let row := a[i]!
      if row.size != n then
        .error s!"{label}: row {i} width {row.size} does not match expected size {n}"

def solveLinear? (a : Array (Array Float)) (b : Array Float) :
    Except String (Array Float) := do
  let n := b.size
  if n == 0 then
    .error "linear solve received an empty system"
  validateSquare? a n "linear solve"

  let mut aug : Array (Array Float) := #[]
  for i in [:n] do
    aug := aug.push (a[i]!.push b[i]!)

  for k in [:n] do
    let mut pivot := k
    let mut pivotAbs := Float.abs ((aug[k]!).getD k 0.0)
    for i in [:n] do
      if i >= k then
        let candidate := Float.abs ((aug[i]!).getD k 0.0)
        if candidate > pivotAbs then
          pivot := i
          pivotAbs := candidate
    if pivotAbs < 1.0e-12 then
      .error s!"linear solve singular pivot at column {k}"
    if pivot != k then
      let rowK := aug[k]!
      aug := aug.set! k aug[pivot]!
      aug := aug.set! pivot rowK
    let pivotRow := aug[k]!
    let pivotVal := pivotRow.getD k 0.0
    for i in [:n] do
      if i != k then
        let factor := ((aug[i]!).getD k 0.0) / pivotVal
        let mut row := aug[i]!
        for j in [:n + 1] do
          row := row.set! j (row.getD j 0.0 - factor * pivotRow.getD j 0.0)
        aug := aug.set! i row

  let mut out : Array Float := #[]
  for i in [:n] do
    let diag := (aug[i]!).getD i 0.0
    if Float.abs diag < 1.0e-12 then
      .error s!"linear solve singular diagonal at row {i}"
    out := out.push (((aug[i]!).getD n 0.0) / diag)
  pure out

end DenseLinearAlgebra

/-!
## Dense linear complementarity problems

Small contact examples need the same mathematical object Drake's rod solver
uses after MLCP reduction:

`z >= 0`, `w = M z + q >= 0`, and `z^T w = 0`.

The active-set enumerator below is deliberately dense and small-scale.  It is
not a replacement for Lemke or SAP; it is the primitive that lets example ports
express the full complementarity boundary without outsourcing physics.
-/

structure LinearComplementaritySolution where
  z : Array Float
  w : Array Float
  activeSet : Array Nat := #[]
  maxComplementarity : Float := 0.0
  deriving Repr, Inhabited

namespace LinearComplementarityProblem

private def validate? (m : Array (Array Float)) (q : Array Float) :
    Except String Unit := do
  DenseLinearAlgebra.validateSquare? m q.size "LCP matrix"

private def activeSubsets (n : Nat) : Array (Array Nat) := Id.run do
  let mut subsets : Array (Array Nat) := #[#[]]
  for i in [:n] do
    let current := subsets
    for subset in current do
      subsets := subsets.push (subset.push i)
  return subsets

private def subvector (xs : Array Float) (indices : Array Nat) : Array Float := Id.run do
  let mut out : Array Float := #[]
  for i in indices do
    out := out.push (xs.getD i 0.0)
  return out

private def submatrix
    (m : Array (Array Float)) (rows cols : Array Nat) : Array (Array Float) :=
  Id.run do
    let mut out : Array (Array Float) := #[]
    for i in rows do
      let src := m.getD i #[]
      let mut row : Array Float := #[]
      for j in cols do
        row := row.push (src.getD j 0.0)
      out := out.push row
    return out

private def fillActive (n : Nat) (active : Array Nat) (values : Array Float) :
    Array Float := Id.run do
  let mut out := Array.replicate n 0.0
  for k in [:active.size] do
    out := out.set! active[k]! (values.getD k 0.0)
  return out

private def minValue (xs : Array Float) : Float :=
  xs.foldl (fun acc x => if x < acc then x else acc) 0.0

private def maxAbsComplementarity (z w : Array Float) : Float := Id.run do
  let n := Nat.max z.size w.size
  let mut out := 0.0
  for i in [:n] do
    let c := Float.abs ((z.getD i 0.0) * (w.getD i 0.0))
    if c > out then
      out := c
  return out

private def clampTinyNonnegative (tol : Float) (xs : Array Float) : Array Float :=
  xs.map (fun x => if x < 0.0 && x >= -tol then 0.0 else x)

private def candidateOk (tol : Float) (z w : Array Float) : Bool :=
  minValue z >= -tol && minValue w >= -tol &&
    maxAbsComplementarity z w <= tol

/--
Solve a small dense LCP by enumerating active sets.

For an active set `A`, the solver enforces `w_A = 0` and `z_not_A = 0`, solves
`M_AA z_A = -q_A`, then checks the global nonnegativity and complementarity
conditions.  This is exponential in problem size, which is acceptable for the
small exact example ports that use it.
-/
def solveByActiveSet?
    (m : Array (Array Float)) (q : Array Float) (tol : Float := 1.0e-8) :
    Except String LinearComplementaritySolution := do
  validate? m q
  let n := q.size
  for active in activeSubsets n do
    let zCandidate? : Except String (Array Float) :=
      if active.isEmpty then
        .ok (Array.replicate n 0.0)
      else
        let maa := submatrix m active active
        let qa := FloatArray.scale (-1.0) (subvector q active)
        match DenseLinearAlgebra.solveLinear? maa qa with
        | .ok zActive => .ok (fillActive n active zActive)
        | .error msg => .error msg
    match zCandidate? with
    | .error _ => pure ()
    | .ok zRaw =>
        let wRaw := FloatArray.add (FloatMatrix.matVec m zRaw) q
        if candidateOk tol zRaw wRaw then
          let z := clampTinyNonnegative tol zRaw
          let w := clampTinyNonnegative tol wRaw
          return {
            z := z
            w := w
            activeSet := active
            maxComplementarity := maxAbsComplementarity z w
          }
  .error "no feasible dense LCP active set found"

end LinearComplementarityProblem

/-!
## Contact-material friction

Drake associates Coulomb coefficients with individual surfaces, then combines
two surfaces into pair coefficients by the harmonic-mean rule used by
`CalcContactFrictionFromSurfaceProperties`.
-/

structure CoulombFriction where
  staticFriction : Float := 0.0
  dynamicFriction : Float := 0.0
  deriving Repr, BEq, Inhabited

namespace CoulombFriction

private def combineCoeff (a b : Float) : Float :=
  let denom := a + b
  if denom == 0.0 then 0.0 else 2.0 * a * b / denom

def combine (a b : CoulombFriction) : CoulombFriction :=
  {
    staticFriction := combineCoeff a.staticFriction b.staticFriction
    dynamicFriction := combineCoeff a.dynamicFriction b.dynamicFriction
  }

def validate? (mu : CoulombFriction) (label : String := "Coulomb friction") :
    Except String Unit := do
  if !(Float.isFinite mu.staticFriction) || mu.staticFriction < 0.0 then
    .error s!"{label}: static friction must be nonnegative and finite, got {mu.staticFriction}"
  if !(Float.isFinite mu.dynamicFriction) || mu.dynamicFriction < 0.0 then
    .error s!"{label}: dynamic friction must be nonnegative and finite, got {mu.dynamicFriction}"
  if mu.dynamicFriction > mu.staticFriction then
    .error s!"{label}: dynamic friction {mu.dynamicFriction} exceeds static friction {mu.staticFriction}"

def frictionless : CoulombFriction := {}

end CoulombFriction

/-!
## Planar contact wrench and static equilibrium

Several Drake examples reduce a multibody contact problem to planar force and
torque balance.  The coordinates here are the brick/body-frame `(y, z)`
components used by `examples/planar_gripper`: forces are expressed in the body
frame and torques are the scalar moment about the `+x` axis.
-/

structure PlanarVec2 where
  y : Float := 0.0
  z : Float := 0.0
  deriving Repr, BEq, Inhabited

namespace PlanarVec2

def add (a b : PlanarVec2) : PlanarVec2 :=
  { y := a.y + b.y, z := a.z + b.z }

def sub (a b : PlanarVec2) : PlanarVec2 :=
  { y := a.y - b.y, z := a.z - b.z }

def scale (s : Float) (v : PlanarVec2) : PlanarVec2 :=
  { y := s * v.y, z := s * v.z }

def dot (a b : PlanarVec2) : Float :=
  a.y * b.y + a.z * b.z

def toArray (v : PlanarVec2) : Array Float :=
  #[v.y, v.z]

def isFinite (v : PlanarVec2) : Bool :=
  Float.isFinite v.y && Float.isFinite v.z

end PlanarVec2

inductive PlanarBoxFace where
  | posY
  | negY
  | posZ
  | negZ
  deriving Repr, BEq, Inhabited

namespace PlanarBoxFace

def label : PlanarBoxFace → String
  | .posY => "+Y"
  | .negY => "-Y"
  | .posZ => "+Z"
  | .negZ => "-Z"

/--
Shift a fingertip sphere center to the brick contact point by moving one
radius inward along the contacted face normal.  This follows Drake's
`BrickStaticEquilibriumNonlinearConstraint`.
-/
def contactPointFromFingerTip
    (face : PlanarBoxFace)
    (fingerTipRadius : Float)
    (tipInBody : PlanarVec2) : PlanarVec2 :=
  match face with
  | .posY => { tipInBody with y := tipInBody.y - fingerTipRadius }
  | .negY => { tipInBody with y := tipInBody.y + fingerTipRadius }
  | .posZ => { tipInBody with z := tipInBody.z - fingerTipRadius }
  | .negZ => { tipInBody with z := tipInBody.z + fingerTipRadius }

def inFrictionCone
    (face : PlanarBoxFace)
    (mu : Float)
    (force : PlanarVec2)
    (tol : Float := 1.0e-12) : Bool :=
  match face with
  | .negY =>
      force.y >= -tol &&
      force.z <= mu * force.y + tol &&
      force.z >= -mu * force.y - tol
  | .posY =>
      force.y <= tol &&
      force.z <= -mu * force.y + tol &&
      force.z >= mu * force.y - tol
  | .negZ =>
      force.z >= -tol &&
      force.y <= mu * force.z + tol &&
      force.y >= -mu * force.z - tol
  | .posZ =>
      force.z <= tol &&
      force.y <= -mu * force.z + tol &&
      force.y >= mu * force.z - tol

end PlanarBoxFace

structure PlanarContactWrench where
  point_B : PlanarVec2 := {}
  force_B : PlanarVec2 := {}
  label : String := ""
  deriving Repr, Inhabited

namespace PlanarContactWrench

def torqueX (wrench : PlanarContactWrench) : Float :=
  wrench.point_B.y * wrench.force_B.z -
    wrench.point_B.z * wrench.force_B.y

end PlanarContactWrench

structure PlanarStaticEquilibrium where
  mass : Float
  gravity : Float := 9.81
  theta : Float := 0.0
  contacts : Array PlanarContactWrench := #[]
  label : String := ""
  deriving Repr, Inhabited

structure PlanarStaticEquilibriumResidual where
  force_B : PlanarVec2
  torqueX : Float
  deriving Repr, Inhabited

namespace PlanarStaticEquilibriumResidual

def toArray (residual : PlanarStaticEquilibriumResidual) : Array Float :=
  #[residual.force_B.y, residual.force_B.z, residual.torqueX]

end PlanarStaticEquilibriumResidual

namespace PlanarStaticEquilibrium

def gravityForceInBody (eq : PlanarStaticEquilibrium) : PlanarVec2 :=
  let mg := eq.mass * eq.gravity
  { y := -mg * Float.sin eq.theta, z := -mg * Float.cos eq.theta }

def validate? (eq : PlanarStaticEquilibrium) : Except String Unit := do
  if !(Float.isFinite eq.mass) || eq.mass < 0.0 then
    .error s!"planar static equilibrium {eq.label}: mass must be nonnegative and finite, got {eq.mass}"
  if !(Float.isFinite eq.gravity) || eq.gravity < 0.0 then
    .error s!"planar static equilibrium {eq.label}: gravity must be nonnegative and finite, got {eq.gravity}"
  if !(Float.isFinite eq.theta) then
    .error s!"planar static equilibrium {eq.label}: theta must be finite, got {eq.theta}"
  for contact in eq.contacts do
    if !contact.point_B.isFinite then
      .error s!"planar static equilibrium {eq.label}: contact point is not finite in {contact.label}"
    if !contact.force_B.isFinite then
      .error s!"planar static equilibrium {eq.label}: contact force is not finite in {contact.label}"

def residual? (eq : PlanarStaticEquilibrium) :
    Except String PlanarStaticEquilibriumResidual := do
  eq.validate?
  let mut force := eq.gravityForceInBody
  let mut torque := 0.0
  for contact in eq.contacts do
    force := force.add contact.force_B
    torque := torque + contact.torqueX
  pure { force_B := force, torqueX := torque }

def residualArray? (eq : PlanarStaticEquilibrium) : Except String (Array Float) := do
  let residual ← eq.residual?
  pure residual.toArray

end PlanarStaticEquilibrium

/--
Result of enforcing `J v_post = target` by the minimum mass-metric impulse
projection

`v_post = v_pre - M^{-1} J^T lambda`.

The `lambda` value solves the contact-space system

`J M^{-1} J^T lambda = J v_pre - target`.
-/
structure VelocityProjection where
  vPost : Array Float
  lambda : Array Float
  correction : Array Float
  delassus : Array (Array Float)
  constraintVelocityBefore : Array Float
  constraintVelocityAfter : Array Float
  target : Array Float
  deriving Repr, Inhabited

namespace VelocityProjection

def validateJacobian? (velocityDim : Nat) (jac : Array (Array Float)) :
    Except String Unit := do
  for i in [:jac.size] do
    let row := jac[i]!
    if row.size != velocityDim then
      .error s!"constraint Jacobian row {i} width {row.size} != velocity dimension {velocityDim}"

def massInverseTimesJacobianTranspose?
    (mass jac : Array (Array Float)) : Except String (Array (Array Float)) := do
  let n := FloatMatrix.colCount mass
  let mut cols : Array (Array Float) := #[]
  for i in [:jac.size] do
    let row := jac[i]!
    let solved ← DenseLinearAlgebra.solveLinear? mass row
    if solved.size != n then
      .error s!"M^-1 J^T column {i} has width {solved.size}, expected {n}"
    cols := cols.push solved
  pure (FloatMatrix.transpose cols)

def project?
    (mass jac : Array (Array Float))
    (vPre : Array Float)
    (target? : Option (Array Float) := none) :
    Except String VelocityProjection := do
  let n := vPre.size
  DenseLinearAlgebra.validateSquare? mass n "mass matrix"
  validateJacobian? n jac
  let target := target?.getD (Array.replicate jac.size 0.0)
  if target.size != jac.size then
    .error s!"projection target size {target.size} != constraint count {jac.size}"
  if jac.isEmpty then
    pure {
      vPost := vPre
      lambda := #[]
      correction := Array.replicate n 0.0
      delassus := #[]
      constraintVelocityBefore := #[]
      constraintVelocityAfter := #[]
      target := target
    }
  else
    let jv := FloatMatrix.matVec jac vPre
    let residual := FloatArray.sub jv target
    let minvJt ← massInverseTimesJacobianTranspose? mass jac
    let delassus := FloatMatrix.matMat jac minvJt
    let lambda ← DenseLinearAlgebra.solveLinear? delassus residual
    let impulse := FloatMatrix.transposeVec jac lambda
    let correction ← DenseLinearAlgebra.solveLinear? mass impulse
    let vPost := FloatArray.sub vPre correction
    pure {
      vPost := vPost
      lambda := lambda
      correction := correction
      delassus := delassus
      constraintVelocityBefore := jv
      constraintVelocityAfter := FloatMatrix.matVec jac vPost
      target := target
    }

end VelocityProjection

/-!
## Normal contact complementarity

Acceleration-level sustained contact solves use the same Delassus operator as
velocity projection, but leave contacts unilateral:

`0 <= f_N ⟂ J_N a + b_N >= 0`,

where `a = M^-1 (tau + J_N^T f_N)`.  This is the small dense primitive behind
full-physics examples that want coupled normal forces without committing to a
particular large-scale contact backend.
-/

structure NormalContactLcpProblem where
  massMatrix : Array (Array Float)
  normalJacobian : Array (Array Float)
  generalizedForces : Array Float
  normalBias : Array Float := #[]
  label : String := ""
  deriving Repr, Inhabited

structure NormalContactLcpResult where
  problem : NormalContactLcpProblem
  delassus : Array (Array Float)
  lcpVector : Array Float
  solution : LinearComplementaritySolution
  normalForces : Array Float
  generalizedNormalForce : Array Float
  freeAcceleration : Array Float
  acceleration : Array Float
  normalMotionBefore : Array Float
  normalMotionAfter : Array Float
  deriving Repr, Inhabited

namespace NormalContactLcpProblem

private def nonnegativeAll (xs : Array Float) (tol : Float) : Bool :=
  xs.all (fun x => x >= -tol)

private def biasVector? (problem : NormalContactLcpProblem) :
    Except String (Array Float) := do
  let m := problem.normalJacobian.size
  if problem.normalBias.isEmpty then
    pure (Array.replicate m 0.0)
  else if problem.normalBias.size == m then
    pure problem.normalBias
  else
    .error s!"normal contact LCP {problem.label}: normal bias size {problem.normalBias.size} != contact count {m}"

def velocityDim (problem : NormalContactLcpProblem) : Nat :=
  problem.generalizedForces.size

def validate? (problem : NormalContactLcpProblem) : Except String Unit := do
  let n := problem.velocityDim
  if n == 0 then
    .error s!"normal contact LCP {problem.label}: empty generalized force vector"
  DenseLinearAlgebra.validateSquare? problem.massMatrix n
    s!"normal contact LCP mass matrix {problem.label}"
  VelocityProjection.validateJacobian? n problem.normalJacobian
  let _ ← problem.biasVector?

def delassus? (problem : NormalContactLcpProblem) :
    Except String (Array (Array Float)) := do
  problem.validate?
  let minvJt ← VelocityProjection.massInverseTimesJacobianTranspose?
    problem.massMatrix problem.normalJacobian
  pure (FloatMatrix.matMat problem.normalJacobian minvJt)

def freeAcceleration? (problem : NormalContactLcpProblem) :
    Except String (Array Float) := do
  problem.validate?
  DenseLinearAlgebra.solveLinear? problem.massMatrix problem.generalizedForces

def lcpVector? (problem : NormalContactLcpProblem) :
    Except String (Array Float × Array Float) := do
  let free ← problem.freeAcceleration?
  let bias ← problem.biasVector?
  pure (FloatArray.add (FloatMatrix.matVec problem.normalJacobian free) bias, free)

def solve? (problem : NormalContactLcpProblem) (tol : Float := 1.0e-8) :
    Except String NormalContactLcpResult := do
  problem.validate?
  let delassus ← problem.delassus?
  let (q, free) ← problem.lcpVector?
  let solution ←
    if q.isEmpty || nonnegativeAll q tol then
      pure {
        z := Array.replicate q.size 0.0
        w := q
        activeSet := #[]
        maxComplementarity := 0.0
      }
    else
      LinearComplementarityProblem.solveByActiveSet? delassus q tol
  let generalizedNormalForce :=
    FloatMatrix.transposeVec problem.normalJacobian solution.z
  let acceleration ←
    DenseLinearAlgebra.solveLinear? problem.massMatrix
      (FloatArray.add problem.generalizedForces generalizedNormalForce)
  let bias ← problem.biasVector?
  pure {
    problem := problem
    delassus := delassus
    lcpVector := q
    solution := solution
    normalForces := solution.z
    generalizedNormalForce := generalizedNormalForce
    freeAcceleration := free
    acceleration := acceleration
    normalMotionBefore := q
    normalMotionAfter :=
      FloatArray.add (FloatMatrix.matVec problem.normalJacobian acceleration) bias
  }

end NormalContactLcpProblem

/-!
## Linear bushing penalty constraints

`LinearBushingRollPitchYaw` is Drake's compliant way to close kinematic loops
when the model is cut into a tree.  The primitive below evaluates the local
spring-damper wrench and maps it into generalized coordinates with `J^T f`.
-/

structure LinearBushingRollPitchYawParams where
  torqueStiffness : Array Float := #[0.0, 0.0, 0.0]
  torqueDamping : Array Float := #[0.0, 0.0, 0.0]
  forceStiffness : Array Float := #[0.0, 0.0, 0.0]
  forceDamping : Array Float := #[0.0, 0.0, 0.0]
  label : String := ""
  deriving Repr, Inhabited

namespace LinearBushingRollPitchYawParams

def fourBarPlanarRevolute
    (forceStiffness forceDamping torqueStiffness torqueDamping : Float) :
    LinearBushingRollPitchYawParams :=
  {
    torqueStiffness := #[torqueStiffness, torqueStiffness, 0.0]
    torqueDamping := #[torqueDamping, torqueDamping, 0.0]
    forceStiffness := #[forceStiffness, forceStiffness, forceStiffness]
    forceDamping := #[forceDamping, forceDamping, forceDamping]
    label := "linear-bushing-rpy-planar-revolute"
  }

private def validate3 (xs : Array Float) (field : String) : Except String Unit :=
  if xs.size == 3 then
    .ok ()
  else
    .error s!"linear bushing {field} size {xs.size} != 3"

def validate? (p : LinearBushingRollPitchYawParams) : Except String Unit := do
  validate3 p.torqueStiffness "torqueStiffness"
  validate3 p.torqueDamping "torqueDamping"
  validate3 p.forceStiffness "forceStiffness"
  validate3 p.forceDamping "forceDamping"

end LinearBushingRollPitchYawParams

structure LinearBushingRollPitchYawState where
  rpyError : Array Float := #[0.0, 0.0, 0.0]
  angularVelocityError : Array Float := #[0.0, 0.0, 0.0]
  translationError : Array Float := #[0.0, 0.0, 0.0]
  translationVelocityError : Array Float := #[0.0, 0.0, 0.0]
  rpyJacobian : Array (Array Float) := #[]
  translationJacobian : Array (Array Float) := #[]
  label : String := ""
  deriving Repr, Inhabited

namespace LinearBushingRollPitchYawState

private def validate3 (xs : Array Float) (field : String) : Except String Unit :=
  if xs.size == 3 then
    .ok ()
  else
    .error s!"linear bushing state {field} size {xs.size} != 3"

def validateJacobianRows? (velocityDim : Nat) (label : String)
    (rows : Array (Array Float)) : Except String Unit := do
  for i in [:rows.size] do
    let row := rows[i]!
    if row.size != velocityDim then
      .error s!"linear bushing {label} row {i} width {row.size} != velocity dimension {velocityDim}"

def validate? (velocityDim : Nat) (s : LinearBushingRollPitchYawState) :
    Except String Unit := do
  validate3 s.rpyError "rpyError"
  validate3 s.angularVelocityError "angularVelocityError"
  validate3 s.translationError "translationError"
  validate3 s.translationVelocityError "translationVelocityError"
  validateJacobianRows? velocityDim "rpyJacobian" s.rpyJacobian
  validateJacobianRows? velocityDim "translationJacobian" s.translationJacobian

end LinearBushingRollPitchYawState

structure LinearBushingRollPitchYawResult where
  torque : Array Float := #[0.0, 0.0, 0.0]
  force : Array Float := #[0.0, 0.0, 0.0]
  generalizedForce : Array Float := #[]
  potentialEnergy : Float := 0.0
  dissipationPower : Float := 0.0
  deriving Repr, Inhabited

namespace LinearBushingRollPitchYaw

private def springDamper
    (stiffness damping error velocity : Array Float) : Array Float := Id.run do
  let mut out : Array Float := #[]
  for i in [:3] do
    out := out.push
      (-(stiffness.getD i 0.0) * error.getD i 0.0 -
        (damping.getD i 0.0) * velocity.getD i 0.0)
  return out

private def springEnergy (stiffness error : Array Float) : Float := Id.run do
  let mut out := 0.0
  for i in [:3] do
    let e := error.getD i 0.0
    out := out + 0.5 * stiffness.getD i 0.0 * e * e
  return out

private def dampingPower (damping velocity : Array Float) : Float := Id.run do
  let mut out := 0.0
  for i in [:3] do
    let v := velocity.getD i 0.0
    out := out + damping.getD i 0.0 * v * v
  return out

def evaluate?
    (velocityDim : Nat)
    (params : LinearBushingRollPitchYawParams)
    (state : LinearBushingRollPitchYawState) :
    Except String LinearBushingRollPitchYawResult := do
  params.validate?
  state.validate? velocityDim
  let torque :=
    springDamper params.torqueStiffness params.torqueDamping
      state.rpyError state.angularVelocityError
  let force :=
    springDamper params.forceStiffness params.forceDamping
      state.translationError state.translationVelocityError
  let torqueGeneralized :=
    FloatMatrix.transposeVec state.rpyJacobian torque
  let forceGeneralized :=
    FloatMatrix.transposeVec state.translationJacobian force
  pure {
    torque := torque
    force := force
    generalizedForce := FloatArray.add torqueGeneralized forceGeneralized
    potentialEnergy :=
      springEnergy params.torqueStiffness state.rpyError +
      springEnergy params.forceStiffness state.translationError
    dissipationPower :=
      dampingPower params.torqueDamping state.angularVelocityError +
      dampingPower params.forceDamping state.translationVelocityError
  }

end LinearBushingRollPitchYaw

/-!
## Particle spring graphs

Drake's mass-spring cloth example is a user-defined dynamical system over a
particle graph.  The primitive below evaluates spring elastic and damping
forces for a graph of 3D particles stored in flat arrays.
-/

structure ParticleSpring where
  particle0 : Nat
  particle1 : Nat
  restLength : Float
  deriving Repr, BEq, Inhabited

structure ParticleSpringParams where
  mass : Float := 1.0
  stiffness : Float := 100.0
  damping : Float := 10.0
  gravityZ : Float := -9.81
  deriving Repr, Inhabited

namespace ParticleSpringParams

def massPerParticle (p : ParticleSpringParams) (particleCount : Nat) : Float :=
  p.mass / particleCount.toFloat

def validate? (particleCount : Nat) (p : ParticleSpringParams) :
    Except String Unit := do
  if particleCount == 0 then
    .error "particle spring system has no particles"
  if !(Float.isFinite p.mass) || p.mass <= 0.0 then
    .error s!"particle spring mass must be positive and finite, got {p.mass}"
  if !(Float.isFinite p.stiffness) || p.stiffness < 0.0 then
    .error s!"particle spring stiffness must be nonnegative and finite, got {p.stiffness}"
  if !(Float.isFinite p.damping) || p.damping < 0.0 then
    .error s!"particle spring damping must be nonnegative and finite, got {p.damping}"
  if !(Float.isFinite p.gravityZ) then
    .error s!"particle spring gravity must be finite, got {p.gravityZ}"

end ParticleSpringParams

structure ParticleSpringForceResult where
  forces : Array Float
  elasticForces : Array Float
  dampingForces : Array Float
  elasticEnergy : Float
  dampingPower : Float
  deriving Repr, Inhabited

namespace ParticleSpringSystem

def stateDim (particleCount : Nat) : Nat :=
  3 * particleCount

def particleState (particleIndex : Nat) (xs : Array Float) : Array Float :=
  let base := 3 * particleIndex
  #[xs.getD base 0.0, xs.getD (base + 1) 0.0, xs.getD (base + 2) 0.0]

private def vec3Add (a b : Array Float) : Array Float :=
  #[
    a.getD 0 0.0 + b.getD 0 0.0,
    a.getD 1 0.0 + b.getD 1 0.0,
    a.getD 2 0.0 + b.getD 2 0.0
  ]

private def vec3Sub (a b : Array Float) : Array Float :=
  #[
    a.getD 0 0.0 - b.getD 0 0.0,
    a.getD 1 0.0 - b.getD 1 0.0,
    a.getD 2 0.0 - b.getD 2 0.0
  ]

private def vec3Scale (s : Float) (v : Array Float) : Array Float :=
  #[s * v.getD 0 0.0, s * v.getD 1 0.0, s * v.getD 2 0.0]

private def vec3Dot (a b : Array Float) : Float :=
  a.getD 0 0.0 * b.getD 0 0.0 +
  a.getD 1 0.0 * b.getD 1 0.0 +
  a.getD 2 0.0 * b.getD 2 0.0

private def vec3Norm (v : Array Float) : Float :=
  Float.sqrt (vec3Dot v v)

def addParticleState (xs : Array Float) (particleIndex : Nat) (delta : Array Float) :
    Array Float := Id.run do
  let mut out := xs
  let base := 3 * particleIndex
  for d in [:3] do
    out := out.set! (base + d) (out.getD (base + d) 0.0 + delta.getD d 0.0)
  return out

def setParticleState (xs : Array Float) (particleIndex : Nat) (value : Array Float) :
    Array Float := Id.run do
  let mut out := xs
  let base := 3 * particleIndex
  for d in [:3] do
    out := out.set! (base + d) (value.getD d 0.0)
  return out

def validateStateSize? (particleCount : Nat) (xs : Array Float) (label : String) :
    Except String Unit := do
  let expected := stateDim particleCount
  if xs.size != expected then
    .error s!"{label} size {xs.size} != particle state dimension {expected}"

def validateSpring? (particleCount : Nat) (spring : ParticleSpring) :
    Except String Unit := do
  if spring.particle0 >= particleCount then
    .error s!"spring particle0 index {spring.particle0} >= particle count {particleCount}"
  if spring.particle1 >= particleCount then
    .error s!"spring particle1 index {spring.particle1} >= particle count {particleCount}"
  if spring.particle0 == spring.particle1 then
    .error s!"spring connects particle {spring.particle0} to itself"
  if !(Float.isFinite spring.restLength) || spring.restLength <= 0.0 then
    .error s!"spring rest length must be positive and finite, got {spring.restLength}"

private def relativeSpringTolerance : Float :=
  2.220446049250313e-15

private def validateCurrentLength? (length restLength : Float) :
    Except String Unit := do
  if length < relativeSpringTolerance * restLength then
    .error "two spring particles are nearly coincident; the state is invalid"

def springForces?
    (params : ParticleSpringParams)
    (q v : Array Float)
    (spring : ParticleSpring) :
    Except String (Array Float × Array Float × Float × Float) := do
  let p0 := spring.particle0
  let p1 := spring.particle1
  let x0 := particleState p0 q
  let x1 := particleState p1 q
  let v0 := particleState p0 v
  let v1 := particleState p1 v
  let dx := vec3Sub x1 x0
  let length := vec3Norm dx
  validateCurrentLength? length spring.restLength
  let n := vec3Scale (1.0 / length) dx
  let extension := length - spring.restLength
  let elastic := vec3Scale (params.stiffness * extension) n
  let relativeVelocity := vec3Sub v1 v0
  let projectedVelocity := vec3Dot relativeVelocity n
  let damping := vec3Scale (params.damping * projectedVelocity) n
  let elasticEnergy := 0.5 * params.stiffness * extension * extension
  let dampingPower := params.damping * projectedVelocity * projectedVelocity
  pure (elastic, damping, elasticEnergy, dampingPower)

def accumulateForces?
    (particleCount : Nat)
    (params : ParticleSpringParams)
    (springs : Array ParticleSpring)
    (q v : Array Float) :
    Except String ParticleSpringForceResult := do
  params.validate? particleCount
  validateStateSize? particleCount q "particle positions"
  validateStateSize? particleCount v "particle velocities"
  let dim := stateDim particleCount
  let mut elasticForces := Array.replicate dim 0.0
  let mut dampingForces := Array.replicate dim 0.0
  let mut elasticEnergy := 0.0
  let mut dampingPower := 0.0
  for spring in springs do
    validateSpring? particleCount spring
    let (elastic, damping, energy, power) ← springForces? params q v spring
    elasticForces := addParticleState elasticForces spring.particle0 elastic
    elasticForces := addParticleState elasticForces spring.particle1 (vec3Scale (-1.0) elastic)
    dampingForces := addParticleState dampingForces spring.particle0 damping
    dampingForces := addParticleState dampingForces spring.particle1 (vec3Scale (-1.0) damping)
    elasticEnergy := elasticEnergy + energy
    dampingPower := dampingPower + power
  pure {
    forces := FloatArray.add elasticForces dampingForces
    elasticForces := elasticForces
    dampingForces := dampingForces
    elasticEnergy := elasticEnergy
    dampingPower := dampingPower
  }

def gridParticleIndex (ny : Nat) (i j : Nat) : Nat :=
  i * ny + j

def initialGridPositions (nx ny : Nat) (h : Float) : Array Float := Id.run do
  let mut out : Array Float := #[]
  for i in [:nx] do
    for j in [:ny] do
      out := out.push (i.toFloat * h)
      out := out.push (j.toFloat * h)
      out := out.push 0.0
  return out

def zeroVelocities (particleCount : Nat) : Array Float :=
  Array.replicate (stateDim particleCount) 0.0

def gridSprings (nx ny : Nat) (h : Float) (useShearingSprings : Bool := true) :
    Array ParticleSpring := Id.run do
  let mut springs : Array ParticleSpring := #[]
  if ny > 1 then
    for i in [:nx] do
      for j in [:ny - 1] do
        let p := gridParticleIndex ny i j
        springs := springs.push { particle0 := p, particle1 := p + 1, restLength := h }
  if nx > 1 then
    for i in [:nx - 1] do
      for j in [:ny] do
        let p := gridParticleIndex ny i j
        springs := springs.push { particle0 := p, particle1 := p + ny, restLength := h }
  if useShearingSprings && nx > 1 && ny > 1 then
    for i in [:nx - 1] do
      for j in [:ny - 1] do
        let p := gridParticleIndex ny i j
        let diagonal := Float.sqrt 2.0 * h
        springs := springs.push { particle0 := p, particle1 := p + ny + 1, restLength := diagonal }
        springs := springs.push { particle0 := p + 1, particle1 := p + ny, restLength := diagonal }
  return springs

def gravityAccelerations (particleCount : Nat) (gravityZ : Float) : Array Float := Id.run do
  let mut out : Array Float := #[]
  for _ in [:particleCount] do
    out := out.push 0.0
    out := out.push 0.0
    out := out.push gravityZ
  return out

def accelerations?
    (particleCount : Nat)
    (params : ParticleSpringParams)
    (springs : Array ParticleSpring)
    (q v : Array Float)
    (pinnedParticles : Array Nat := #[]) :
    Except String (Array Float × ParticleSpringForceResult) := do
  let result ← accumulateForces? particleCount params springs q v
  let m := params.massPerParticle particleCount
  if !(Float.isFinite m) || m <= 0.0 then
    .error s!"mass per particle must be positive and finite, got {m}"
  let mut accel := FloatArray.add (FloatArray.scale (1.0 / m) result.forces)
    (gravityAccelerations particleCount params.gravityZ)
  for p in pinnedParticles do
    if p >= particleCount then
      .error s!"pinned particle index {p} >= particle count {particleCount}"
    accel := setParticleState accel p #[0.0, 0.0, 0.0]
  pure (accel, result)

end ParticleSpringSystem

end Tyr.EventSkeleton
