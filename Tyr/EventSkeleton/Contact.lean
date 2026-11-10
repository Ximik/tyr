import Tyr.EventSkeleton.Physics
import Tyr.EventSkeleton.Trace

/-!
# Tyr.EventSkeleton.Contact

Contact-specific support selection for hybrid event skeletons.

`ContactCandidate` is a primitive view of one possible contact constraint, not
a promised storage layout.  Geometry providers may generate these views from
URDF-style collision geometry, hand-written kinematics, or a packed
structure-of-arrays broadphase.  The physics layer consumes only the primitive
fields here: signed distance, contact velocities, and generalized-velocity
Jacobian rows.
-/

namespace Tyr.EventSkeleton

/-- Coarse contact mode assigned after geometric and velocity tests. -/
inductive ContactMode where
  | separated
  | impacting
  | sticking
  | sliding
  deriving Repr, BEq, Inhabited

/--
One candidate contact point or patch.

The Jacobian rows use generalized-velocity coordinates.  A high-performance
many-contact backend can keep a different internal representation and expose
this shape only at the support-selection / solver boundary.
-/
structure ContactCandidate where
  id : Nat
  bodyA : String := ""
  bodyB : String := ""
  point_W : Array Float := #[]
  normal_W : Array Float := #[]
  signedDistance : Float
  normalVelocity : Float := 0.0
  tangentVelocity : Float := 0.0
  tangentVelocity2 : Float := 0.0
  normalJacobian : Array Float := #[]
  tangentJacobian : Array Float := #[]
  tangentJacobian2 : Array Float := #[]
  mode : ContactMode := .separated
  label : String := ""
  deriving Repr, Inhabited

namespace ContactCandidate

def withinDistance (distanceTol : Float) (candidate : ContactCandidate) : Bool :=
  candidate.signedDistance <= distanceTol

def isClosing (velocityTol : Float) (candidate : ContactCandidate) : Bool :=
  candidate.normalVelocity < velocityTol

def classify
    (distanceTol tangentVelocityTol : Float)
    (candidate : ContactCandidate) : ContactMode :=
  let tangentSpeed :=
    max (Float.abs candidate.tangentVelocity) (Float.abs candidate.tangentVelocity2)
  if !candidate.withinDistance distanceTol then
    .separated
  else if candidate.isClosing 0.0 then
    .impacting
  else if tangentSpeed <= tangentVelocityTol then
    .sticking
  else
    .sliding

def withClassifiedMode
    (distanceTol tangentVelocityTol : Float)
    (candidate : ContactCandidate) : ContactCandidate :=
  { candidate with mode := candidate.classify distanceTol tangentVelocityTol }

def validateJacobianWidth? (width : Nat) (candidate : ContactCandidate) :
    Except String Unit := do
  if candidate.normalJacobian.size != width then
    .error s!"contact candidate {candidate.id}: normal Jacobian width {candidate.normalJacobian.size} != expected {width}"
  else if candidate.tangentJacobian.size != width then
    .error s!"contact candidate {candidate.id}: tangent Jacobian width {candidate.tangentJacobian.size} != expected {width}"
  else if !candidate.tangentJacobian2.isEmpty && candidate.tangentJacobian2.size != width then
    .error s!"contact candidate {candidate.id}: second tangent Jacobian width {candidate.tangentJacobian2.size} != expected {width}"
  else
    .ok ()

def constraintJacobianRows (includeTangent : Bool) (candidate : ContactCandidate) :
    Array (Array Float) := Id.run do
  if includeTangent then
    let mut rows := #[candidate.normalJacobian, candidate.tangentJacobian]
    if !candidate.tangentJacobian2.isEmpty then
      rows := rows.push candidate.tangentJacobian2
    return rows
  else
    return #[candidate.normalJacobian]

/--
Turn scalar normal/tangent contact forces into generalized forces.

This is the common full-physics boundary: contact models may compute the scalar
forces by penalty, complementarity, sampled branch hypotheses, or learned
surrogates, while multibody dynamics only needs `J^T f`.
-/
def generalizedForce (normalForce tangentForce : Float)
    (candidate : ContactCandidate) : Array Float :=
  FloatArray.add
    (FloatArray.scale normalForce candidate.normalJacobian)
    (FloatArray.scale tangentForce candidate.tangentJacobian)

def generalizedForce3D (normalForce tangentForce tangentForce2 : Float)
    (candidate : ContactCandidate) : Array Float :=
  FloatArray.add
    (candidate.generalizedForce normalForce tangentForce)
    (FloatArray.scale tangentForce2 candidate.tangentJacobian2)

end ContactCandidate

structure ContactForceScalars where
  candidateId : Nat
  normalForce : Float := 0.0
  tangentForce : Float := 0.0
  tangentForce2 : Float := 0.0
  mode : ContactMode := .separated
  label : String := ""
  deriving Repr, Inhabited

namespace ContactForceScalars

def fromCandidate
    (candidate : ContactCandidate)
    (normalForce tangentForce : Float) : ContactForceScalars :=
  {
    candidateId := candidate.id
    normalForce := normalForce
    tangentForce := tangentForce
    tangentForce2 := 0.0
    mode := candidate.mode
    label := candidate.label
  }

def fromCandidate3D
    (candidate : ContactCandidate)
    (normalForce tangentForce tangentForce2 : Float) : ContactForceScalars :=
  {
    candidateId := candidate.id
    normalForce := normalForce
    tangentForce := tangentForce
    tangentForce2 := tangentForce2
    mode := candidate.mode
    label := candidate.label
  }

def generalizedForce (candidate : ContactCandidate) (force : ContactForceScalars) :
    Array Float :=
  candidate.generalizedForce3D force.normalForce force.tangentForce force.tangentForce2

end ContactForceScalars

/-!
## Hydroelastic contact patches

Point contact support is not enough for Drake's hydroelastic examples.  A
hydroelastic provider exposes an already-computed contact surface patch: area,
centroid, average pressure, compliance metadata, and generalized-velocity
Jacobian rows.  The dynamics layer can consume this primitive with the same
`J^T f` boundary used by point contacts, while more sophisticated geometry
providers can keep mesh, polygon, or triangle details internally.
-/

inductive HydroelasticCompliance where
  | rigid
  | compliant
  deriving Repr, BEq, Inhabited

inductive HydroelasticSurfaceRepresentation where
  | triangle
  | polygon
  deriving Repr, BEq, Inhabited

inductive HydroelasticPairKind where
  | rigidRigid
  | rigidCompliant
  | compliantCompliant
  deriving Repr, BEq, Inhabited

structure HydroelasticContactPatch where
  id : Nat
  bodyA : String := ""
  bodyB : String := ""
  complianceA : HydroelasticCompliance := .compliant
  complianceB : HydroelasticCompliance := .rigid
  representation : HydroelasticSurfaceRepresentation := .polygon
  area : Float := 0.0
  centroid : Array Float := #[0.0, 0.0, 0.0]
  normal : Array Float := #[0.0, 0.0, 1.0]
  averagePressure : Float := 0.0
  normalVelocity : Float := 0.0
  tangentVelocity : Float := 0.0
  tangentVelocity2 : Float := 0.0
  normalJacobian : Array Float := #[]
  tangentJacobian : Array Float := #[]
  tangentJacobian2 : Array Float := #[]
  label : String := ""
  deriving Repr, Inhabited

namespace HydroelasticContactPatch

def pairKind (patch : HydroelasticContactPatch) : HydroelasticPairKind :=
  match patch.complianceA, patch.complianceB with
  | .rigid, .rigid => .rigidRigid
  | .compliant, .compliant => .compliantCompliant
  | _, _ => .rigidCompliant

def normalForce (patch : HydroelasticContactPatch) : Float :=
  patch.area * patch.averagePressure

def validateJacobianWidth? (width : Nat) (patch : HydroelasticContactPatch) :
    Except String Unit := do
  if patch.normalJacobian.size != width then
    .error s!"hydroelastic patch {patch.id}: normal Jacobian width {patch.normalJacobian.size} != expected {width}"
  else if patch.tangentJacobian.size != width then
    .error s!"hydroelastic patch {patch.id}: tangent Jacobian width {patch.tangentJacobian.size} != expected {width}"
  else if !patch.tangentJacobian2.isEmpty && patch.tangentJacobian2.size != width then
    .error s!"hydroelastic patch {patch.id}: second tangent Jacobian width {patch.tangentJacobian2.size} != expected {width}"
  else
    .ok ()

def validateGeometry? (patch : HydroelasticContactPatch) :
    Except String Unit := do
  if !(Float.isFinite patch.area) || patch.area < 0.0 then
    .error s!"hydroelastic patch {patch.id}: area must be nonnegative and finite, got {patch.area}"
  if !(Float.isFinite patch.averagePressure) || patch.averagePressure < 0.0 then
    .error s!"hydroelastic patch {patch.id}: average pressure must be nonnegative and finite, got {patch.averagePressure}"
  if patch.centroid.size != 3 then
    .error s!"hydroelastic patch {patch.id}: centroid size {patch.centroid.size} != 3"
  if patch.normal.size != 3 then
    .error s!"hydroelastic patch {patch.id}: normal size {patch.normal.size} != 3"

def generalizedForce3D
    (patch : HydroelasticContactPatch)
    (normalForce : Float := patch.normalForce)
    (tangentForce : Float := 0.0)
    (tangentForce2 : Float := 0.0) : Array Float :=
  FloatArray.add
    (FloatArray.add
      (FloatArray.scale normalForce patch.normalJacobian)
      (FloatArray.scale tangentForce patch.tangentJacobian))
    (FloatArray.scale tangentForce2 patch.tangentJacobian2)

def equivalentContactCandidate (patch : HydroelasticContactPatch) :
    ContactCandidate :=
  {
    id := patch.id
    bodyA := patch.bodyA
    bodyB := patch.bodyB
    point_W := patch.centroid
    normal_W := patch.normal
    signedDistance := if patch.area > 0.0 then 0.0 else 1.0
    normalVelocity := patch.normalVelocity
    tangentVelocity := patch.tangentVelocity
    tangentVelocity2 := patch.tangentVelocity2
    normalJacobian := patch.normalJacobian
    tangentJacobian := patch.tangentJacobian
    tangentJacobian2 := patch.tangentJacobian2
    mode := if patch.area > 0.0 then .sticking else .separated
    label := patch.label
  }

def contactForceScalars
    (patch : HydroelasticContactPatch)
    (normalForce : Float := patch.normalForce)
    (tangentForce : Float := 0.0)
    (tangentForce2 : Float := 0.0) : ContactForceScalars :=
  ContactForceScalars.fromCandidate3D patch.equivalentContactCandidate
    normalForce tangentForce tangentForce2

end HydroelasticContactPatch

structure HydroelasticPatchSupport where
  policy : SupportPolicy
  patches : Array HydroelasticContactPatch := #[]
  selectedLocalIndices : Array Nat := #[]
  sourcePatchCount? : Option Nat := none
  label : String := ""
  deriving Repr, Inhabited

namespace HydroelasticPatchSupport

def totalPatches (support : HydroelasticPatchSupport) : Nat :=
  match support.sourcePatchCount? with
  | some n => n
  | none => support.patches.size

def selectedPatches? (support : HydroelasticPatchSupport) :
    Except String (Array HydroelasticContactPatch) := do
  let mut out : Array HydroelasticContactPatch := #[]
  for i in support.selectedLocalIndices do
    if i < support.patches.size then
      out := out.push support.patches[i]!
    else
      .error s!"hydroelastic patch support {support.label}: selected local index {i} out of bounds for {support.patches.size} patches"
  pure out

def selectedIds? (support : HydroelasticPatchSupport) :
    Except String (Array Nat) := do
  let patches ← support.selectedPatches?
  pure (patches.map (fun patch => patch.id))

def toRuntimeSupport? (support : HydroelasticPatchSupport) :
    Except String RuntimeSupport := do
  let ids ← support.selectedIds?
  pure {
    policy := support.policy
    selectedIds := ids
    totalCandidates? := some support.totalPatches
    label := support.label
  }

def validateJacobianWidth? (width : Nat) (support : HydroelasticPatchSupport) :
    Except String Unit := do
  for patch in support.patches do
    patch.validateJacobianWidth? width

def validateGeometry? (support : HydroelasticPatchSupport) :
    Except String Unit := do
  for patch in support.patches do
    patch.validateGeometry?

def selectByArea
    (minArea : Float)
    (patches : Array HydroelasticContactPatch)
    (label : String := "") : HydroelasticPatchSupport := Id.run do
  let mut selected : Array Nat := #[]
  for i in [:patches.size] do
    if patches[i]!.area > minArea then
      selected := selected.push i
  return {
    policy := .threshold minArea
    patches := patches
    selectedLocalIndices := selected
    sourcePatchCount? := some patches.size
    label := label
  }

end HydroelasticPatchSupport

def sumGeneralizedForces (forces : Array (Array Float)) : Array Float :=
  forces.foldl FloatArray.add #[]

/--
The contact support selected on one forward pass.

`selectedLocalIndices` point into `candidates`; `toRuntimeSupport?` maps them to
stable candidate IDs for the generic event trace.
-/
structure ContactSupport where
  policy : SupportPolicy
  candidates : Array ContactCandidate := #[]
  selectedLocalIndices : Array Nat := #[]
  sourceCandidateCount? : Option Nat := none
  label : String := ""
  deriving Repr, Inhabited

namespace ContactSupport

private def natRangeArray (n : Nat) : Array Nat := Id.run do
  let mut out : Array Nat := #[]
  for i in [:n] do
    out := out.push i
  return out

private def trimNatArray (n : Nat) (xs : Array Nat) : Array Nat := Id.run do
  let mut out : Array Nat := #[]
  for i in [:Nat.min n xs.size] do
    out := out.push xs[i]!
  return out

private def localIndexForId? (id : Nat) (candidates : Array ContactCandidate) : Option Nat := Id.run do
  for i in [:candidates.size] do
    if candidates[i]!.id == id then
      return some i
  return none

private def insertClosestIndex
    (k candidateIndex : Nat)
    (candidates : Array ContactCandidate)
    (selected : Array Nat) : Array Nat := Id.run do
  if k == 0 then
    return #[]
  let d := candidates[candidateIndex]!.signedDistance
  let mut out : Array Nat := #[]
  let mut inserted := false
  for selectedIndex in selected do
    if !inserted && d < candidates[selectedIndex]!.signedDistance then
      out := out.push candidateIndex
      inserted := true
    out := out.push selectedIndex
  if !inserted then
    out := out.push candidateIndex
  return trimNatArray k out

def validateSelectedIndices? (support : ContactSupport) : Except String Unit := do
  for i in support.selectedLocalIndices do
    if i < support.candidates.size then
      pure ()
    else
        .error s!"contact support {support.label}: selected local index {i} out of bounds for {support.candidates.size} candidates"

def selectedCandidates? (support : ContactSupport) :
    Except String (Array ContactCandidate) := do
  let mut out : Array ContactCandidate := #[]
  for i in support.selectedLocalIndices do
    if i < support.candidates.size then
      out := out.push support.candidates[i]!
    else
        .error s!"contact support {support.label}: selected local index {i} out of bounds for {support.candidates.size} candidates"
  pure out

def selectedIds? (support : ContactSupport) : Except String (Array Nat) := do
  let candidates ← support.selectedCandidates?
  pure (candidates.map (fun candidate => candidate.id))

def totalCandidates (support : ContactSupport) : Nat :=
  match support.sourceCandidateCount? with
  | some n => n
  | none => support.candidates.size

def validateSourceCandidateCount? (support : ContactSupport) : Except String Unit := do
  match support.sourceCandidateCount? with
  | none => pure ()
  | some n =>
      if n < support.candidates.size then
        .error s!"contact support {support.label}: source candidate count {n} is smaller than retained candidate count {support.candidates.size}"
      else
        pure ()

def minimumSignedDistance? (support : ContactSupport) : Option Float :=
  support.candidates.foldl
    (fun acc candidate =>
      match acc with
      | none => some candidate.signedDistance
      | some d => some (if candidate.signedDistance < d then candidate.signedDistance else d))
    none

def toRuntimeSupport? (support : ContactSupport) : Except String RuntimeSupport := do
  let ids ← support.selectedIds?
  pure {
    policy := support.policy
    selectedIds := ids
    totalCandidates? := some support.totalCandidates
    label := support.label
  }

def validateJacobianWidth? (width : Nat) (support : ContactSupport) :
    Except String Unit := do
  for candidate in support.candidates do
    candidate.validateJacobianWidth? width

def constraintJacobianRows? (support : ContactSupport) (includeTangent : Bool) :
    Except String (Array (Array Float)) := do
  let candidates ← support.selectedCandidates?
  let mut rows : Array (Array Float) := #[]
  for candidate in candidates do
    for row in candidate.constraintJacobianRows includeTangent do
      rows := rows.push row
  pure rows

def classifyCandidates
    (distanceTol tangentVelocityTol : Float)
    (support : ContactSupport) : ContactSupport :=
  { support with
    candidates :=
      support.candidates.map
        (fun candidate => candidate.withClassifiedMode distanceTol tangentVelocityTol) }

def selectByDistance
    (distanceTol : Float)
    (candidates : Array ContactCandidate)
    (label : String := "") : ContactSupport := Id.run do
  let mut selected : Array Nat := #[]
  for i in [:candidates.size] do
    if candidates[i]!.withinDistance distanceTol then
      selected := selected.push i
  return {
    policy := .threshold distanceTol
    candidates := candidates
    selectedLocalIndices := selected
    sourceCandidateCount? := some candidates.size
    label := label
  }

def selectFull
    (candidates : Array ContactCandidate)
    (label : String := "") : ContactSupport :=
  {
    policy := .fullSupport
    candidates := candidates
    selectedLocalIndices := natRangeArray candidates.size
    sourceCandidateCount? := some candidates.size
    label := label
  }

def selectTopK
    (k : Nat)
    (localIndices : Array Nat)
    (candidates : Array ContactCandidate)
    (label : String := "") : ContactSupport :=
  {
    policy := .topK k
    candidates := candidates
    selectedLocalIndices := localIndices
    sourceCandidateCount? := some candidates.size
    label := label
  }

def selectClosestK
    (k : Nat)
    (candidates : Array ContactCandidate)
    (label : String := "") : ContactSupport := Id.run do
  let mut selected : Array Nat := #[]
  for i in [:candidates.size] do
    selected := insertClosestIndex k i candidates selected
  return {
    policy := .topK k
    candidates := candidates
    selectedLocalIndices := selected
    sourceCandidateCount? := some candidates.size
    label := label
  }

def selectWithPolicy
    (policy : SupportPolicy)
    (candidates : Array ContactCandidate)
    (label : String := "") : ContactSupport :=
  match policy with
  | .fullSupport =>
      selectFull candidates label
  | .threshold distanceTol =>
      selectByDistance distanceTol candidates label
  | .topK k =>
      selectClosestK k candidates label
  | .learnedTail explicitCount =>
      { selectClosestK explicitCount candidates label with
        policy := policy }
  | .sampled sampleId =>
      let selected :=
        match localIndexForId? sampleId candidates with
        | some i => #[i]
        | none => if sampleId < candidates.size then #[sampleId] else #[]
      {
        policy := policy
        candidates := candidates
        selectedLocalIndices := selected
        sourceCandidateCount? := some candidates.size
        label := label
      }
  | .deterministicPick selectedId =>
      let selected :=
        match localIndexForId? selectedId candidates with
        | some i => #[i]
        | none => if selectedId < candidates.size then #[selectedId] else #[]
      {
        policy := policy
        candidates := candidates
        selectedLocalIndices := selected
        sourceCandidateCount? := some candidates.size
        label := label
      }

end ContactSupport

/-!
## Contact candidate provider result

`ContactCandidateSet` is the primitive output of a dynamic contact provider.
The provider may internally use a broadphase, a packed structure-of-arrays, or
URDF/SceneGraph-specific collision data; the full-physics layer still receives
stable candidate IDs and the small `ContactCandidate` views needed for support
selection and `J^T f`.
-/

structure ContactCandidateSet where
  candidates : Array ContactCandidate := #[]
  sourceCandidateCount? : Option Nat := none
  label : String := ""
  deriving Repr, Inhabited

namespace ContactCandidateSet

def ofArray (candidates : Array ContactCandidate) (label : String := "") :
    ContactCandidateSet :=
  {
    candidates := candidates
    sourceCandidateCount? := some candidates.size
    label := label
  }

def totalCandidates (set : ContactCandidateSet) : Nat :=
  match set.sourceCandidateCount? with
  | some n => n
  | none => set.candidates.size

def minimumSignedDistance? (set : ContactCandidateSet) : Option Float :=
  set.candidates.foldl
    (fun acc candidate =>
      match acc with
      | none => some candidate.signedDistance
      | some d => some (if candidate.signedDistance < d then candidate.signedDistance else d))
    none

def validateSourceCandidateCount? (set : ContactCandidateSet) :
    Except String Unit := do
  match set.sourceCandidateCount? with
  | none => pure ()
  | some n =>
      if n < set.candidates.size then
        .error s!"contact candidate set {set.label}: source candidate count {n} is smaller than retained candidate count {set.candidates.size}"
      else
        pure ()

def validateUniqueIds? (set : ContactCandidateSet) : Except String Unit := do
  for i in [:set.candidates.size] do
    for j in [:(set.candidates.size - i - 1)] do
      let k := i + j + 1
      if set.candidates[i]!.id == set.candidates[k]!.id then
        .error s!"contact candidate set {set.label}: duplicate candidate id {set.candidates[i]!.id}"

def validateJacobianWidth? (width : Nat) (set : ContactCandidateSet) :
    Except String Unit := do
  for candidate in set.candidates do
    candidate.validateJacobianWidth? width

def validate? (set : ContactCandidateSet) (jacobianWidth? : Option Nat := none) :
    Except String Unit := do
  set.validateSourceCandidateCount?
  set.validateUniqueIds?
  match jacobianWidth? with
  | none => pure ()
  | some width => set.validateJacobianWidth? width

private def supportLabel (set : ContactCandidateSet) (label : String) : String :=
  if label.isEmpty then set.label else label

def selectWithPolicy
    (policy : SupportPolicy)
    (set : ContactCandidateSet)
    (label : String := "") : ContactSupport :=
  let support :=
    ContactSupport.selectWithPolicy policy set.candidates (set.supportLabel label)
  { support with sourceCandidateCount? := some set.totalCandidates }

def selectByDistance
    (distanceTol : Float)
    (set : ContactCandidateSet)
    (label : String := "") : ContactSupport :=
  set.selectWithPolicy (.threshold distanceTol) label

def selectFull (set : ContactCandidateSet) (label : String := "") : ContactSupport :=
  set.selectWithPolicy .fullSupport label

def selectClosestK
    (k : Nat)
    (set : ContactCandidateSet)
    (label : String := "") : ContactSupport :=
  set.selectWithPolicy (.topK k) label

end ContactCandidateSet

/-!
## Packed contact candidate batches

Many-contact examples should not have to allocate a `ContactCandidate` record
for every possible geometry pair before broadphase/support selection.  This
batch is a simple structure-of-arrays provider format: it keeps cheap scalar
columns packed, materializes `ContactCandidate` views only for retained
candidates, and preserves the original source count for trace diagnostics.
-/

structure PackedContactCandidateBatch where
  ids : Array Nat := #[]
  bodyA : Array String := #[]
  bodyB : Array String := #[]
  point_W : Array (Array Float) := #[]
  normal_W : Array (Array Float) := #[]
  signedDistance : Array Float := #[]
  normalVelocity : Array Float := #[]
  tangentVelocity : Array Float := #[]
  tangentVelocity2 : Array Float := #[]
  normalJacobian : Array (Array Float) := #[]
  tangentJacobian : Array (Array Float) := #[]
  tangentJacobian2 : Array (Array Float) := #[]
  labels : Array String := #[]
  sourceCandidateCount? : Option Nat := none
  label : String := ""
  deriving Repr, Inhabited

namespace PackedContactCandidateBatch

def size (batch : PackedContactCandidateBatch) : Nat :=
  batch.ids.size

def totalCandidates (batch : PackedContactCandidateBatch) : Nat :=
  match batch.sourceCandidateCount? with
  | some n => n
  | none => batch.size

/-- The columns every candidate reads have one entry per id. -/
structure RequiredColumns (batch : PackedContactCandidateBatch) : Prop where
  signedDistance : batch.signedDistance.size = batch.size
  normalJacobian : batch.normalJacobian.size = batch.size
  tangentJacobian : batch.tangentJacobian.size = batch.size

private def validateRequiredColumn?
    (batch : PackedContactCandidateBatch)
    (field : String)
    (actual : Nat) : Except String (PLift (actual = batch.size)) :=
  if h : actual = batch.size then
    .ok ⟨h⟩
  else
    .error s!"packed contact batch {batch.label}: {field} size {actual} != ids size {batch.size}"

private def validateOptionalColumn?
    (batch : PackedContactCandidateBatch)
    (field : String)
    (actual : Nat) : Except String Unit :=
  if actual == 0 || actual == batch.size then
    .ok ()
  else
    .error s!"packed contact batch {batch.label}: optional {field} size {actual} must be 0 or ids size {batch.size}"

private def validateFiniteColumn?
    (batch : PackedContactCandidateBatch)
    (field : String)
    (values : Array Float) : Except String Unit := do
  for i in [:values.size] do
    if !(values[i]!).isFinite then
      .error s!"packed contact batch {batch.label}: {field}[{i}] must be finite, got {values[i]!}"

def validateSourceCandidateCount? (batch : PackedContactCandidateBatch) :
    Except String Unit := do
  match batch.sourceCandidateCount? with
  | none => pure ()
  | some n =>
      if n < batch.size then
        .error s!"packed contact batch {batch.label}: source candidate count {n} is smaller than retained column count {batch.size}"
      else
        pure ()

def validate? (batch : PackedContactCandidateBatch) :
    Except String (PLift batch.RequiredColumns) := do
  let ⟨hDistance⟩ ← batch.validateRequiredColumn? "signedDistance" batch.signedDistance.size
  let ⟨hNormal⟩ ← batch.validateRequiredColumn? "normalJacobian" batch.normalJacobian.size
  let ⟨hTangent⟩ ← batch.validateRequiredColumn? "tangentJacobian" batch.tangentJacobian.size
  batch.validateOptionalColumn? "bodyA" batch.bodyA.size
  batch.validateOptionalColumn? "bodyB" batch.bodyB.size
  batch.validateOptionalColumn? "point_W" batch.point_W.size
  batch.validateOptionalColumn? "normal_W" batch.normal_W.size
  batch.validateOptionalColumn? "normalVelocity" batch.normalVelocity.size
  batch.validateOptionalColumn? "tangentVelocity" batch.tangentVelocity.size
  batch.validateOptionalColumn? "tangentVelocity2" batch.tangentVelocity2.size
  batch.validateOptionalColumn? "tangentJacobian2" batch.tangentJacobian2.size
  batch.validateOptionalColumn? "labels" batch.labels.size
  batch.validateSourceCandidateCount?
  batch.validateFiniteColumn? "signedDistance" batch.signedDistance
  batch.validateFiniteColumn? "normalVelocity" batch.normalVelocity
  batch.validateFiniteColumn? "tangentVelocity" batch.tangentVelocity
  batch.validateFiniteColumn? "tangentVelocity2" batch.tangentVelocity2
  pure ⟨⟨hDistance, hNormal, hTangent⟩⟩

/-- An absent optional column (empty) reads as the default; `validate?` rejects
    a present column without one entry per id. -/
private def optionalColumnValue [Inhabited α] (values : Array α) (i : Nat) : α :=
  values[i]?.getD default

private def candidateAt
    (batch : PackedContactCandidateBatch)
    (columns : batch.RequiredColumns)
    (i : Fin batch.size) : ContactCandidate :=
  {
    id := batch.ids[i.val]'i.isLt
    bodyA := optionalColumnValue batch.bodyA i
    bodyB := optionalColumnValue batch.bodyB i
    point_W := optionalColumnValue batch.point_W i
    normal_W := optionalColumnValue batch.normal_W i
    signedDistance := batch.signedDistance[i.val]'(columns.signedDistance.symm ▸ i.isLt)
    normalVelocity := optionalColumnValue batch.normalVelocity i
    tangentVelocity := optionalColumnValue batch.tangentVelocity i
    tangentVelocity2 := optionalColumnValue batch.tangentVelocity2 i
    normalJacobian := batch.normalJacobian[i.val]'(columns.normalJacobian.symm ▸ i.isLt)
    tangentJacobian := batch.tangentJacobian[i.val]'(columns.tangentJacobian.symm ▸ i.isLt)
    tangentJacobian2 := optionalColumnValue batch.tangentJacobian2 i
    label := optionalColumnValue batch.labels i
  }

def candidateAt? (batch : PackedContactCandidateBatch) (i : Nat) :
    Except String ContactCandidate := do
  let ⟨columns⟩ ← batch.validate?
  if h : i < batch.size then
    pure (batch.candidateAt columns ⟨i, h⟩)
  else
    .error s!"packed contact batch {batch.label}: candidate index {i} out of bounds for {batch.size}"

private def materializeIndices?
    (batch : PackedContactCandidateBatch)
    (indices : Array Nat)
    (label : String) : Except String ContactCandidateSet := do
  let ⟨columns⟩ ← batch.validate?
  let mut candidates : Array ContactCandidate := #[]
  for i in indices do
    if h : i < batch.size then
      candidates := candidates.push (batch.candidateAt columns ⟨i, h⟩)
    else
      .error s!"packed contact batch {batch.label}: retained index {i} out of bounds for {batch.size}"
  let set : ContactCandidateSet := {
    candidates := candidates
    sourceCandidateCount? := some batch.totalCandidates
    label := if label.isEmpty then batch.label else label
  }
  set.validate?
  pure set

private def allIndices (n : Nat) : Array Nat := Id.run do
  let mut out : Array Nat := #[]
  for i in [:n] do
    out := out.push i
  return out

def toCandidateSet? (batch : PackedContactCandidateBatch) (label : String := "") :
    Except String ContactCandidateSet :=
  batch.materializeIndices? (allIndices batch.size) label

def minimumSignedDistance? (batch : PackedContactCandidateBatch) :
    Except String (Option Float) := do
  let _ ← batch.validate?
  pure <|
    batch.signedDistance.foldl
      (fun acc d =>
        match acc with
        | none => some d
        | some best => some (if d < best then d else best))
      none

def retainedByDistance?
    (batch : PackedContactCandidateBatch)
    (distanceTol : Float)
    (label : String := "") :
    Except String ContactCandidateSet := do
  let _ ← batch.validate?
  let mut indices : Array Nat := #[]
  for i in [:batch.size] do
    if batch.signedDistance[i]! <= distanceTol then
      indices := indices.push i
  batch.materializeIndices? indices label

private def trimNatArray (n : Nat) (xs : Array Nat) : Array Nat := Id.run do
  let mut out : Array Nat := #[]
  for i in [:Nat.min n xs.size] do
    out := out.push xs[i]!
  return out

private def insertClosestIndex
    (batch : PackedContactCandidateBatch)
    (k candidateIndex : Nat)
    (selected : Array Nat) : Array Nat := Id.run do
  if k == 0 then
    return #[]
  let d := batch.signedDistance[candidateIndex]!
  let mut out : Array Nat := #[]
  let mut inserted := false
  for selectedIndex in selected do
    if !inserted && d < batch.signedDistance[selectedIndex]! then
      out := out.push candidateIndex
      inserted := true
    out := out.push selectedIndex
  if !inserted then
    out := out.push candidateIndex
  return trimNatArray k out

def retainedClosestK?
    (batch : PackedContactCandidateBatch)
    (k : Nat)
    (label : String := "") :
    Except String ContactCandidateSet := do
  let _ ← batch.validate?
  let mut indices : Array Nat := #[]
  for i in [:batch.size] do
    indices := insertClosestIndex batch k i indices
  batch.materializeIndices? indices label

end PackedContactCandidateBatch

/-!
## Contact candidate providers

Full physics should not require `ContactCandidate` to be the provider's native
storage format.  A broadphase, URDF importer, or SceneGraph query can keep its
own model-specific data structures and expose a `ContactCandidateSet` only at
the support-selection boundary.  This provider abstraction makes that dynamic
boundary explicit while keeping the solver interface primitive-only.
-/

structure ContactCandidateProvider (State : Type) where
  label : String := ""
  candidatesAt? : State → Except String ContactCandidateSet

namespace ContactCandidateProvider

private def resolvedLabel (providerLabel label : String) : String :=
  if label.isEmpty then providerLabel else label

def validCandidatesAt? {State : Type}
    (provider : ContactCandidateProvider State)
    (state : State)
    (jacobianWidth? : Option Nat := none) :
    Except String ContactCandidateSet := do
  let set ← provider.candidatesAt? state
  set.validate? jacobianWidth?
  pure set

def supportAt? {State : Type}
    (provider : ContactCandidateProvider State)
    (state : State)
    (policy : SupportPolicy)
    (distanceTol : Float := 0.0)
    (tangentVelocityTol : Float := 1.0e-9)
    (jacobianWidth? : Option Nat := none)
    (label : String := "") :
    Except String ContactSupport := do
  let set ← provider.validCandidatesAt? state jacobianWidth?
  pure <|
    (set.selectWithPolicy policy (resolvedLabel provider.label label))
      |>.classifyCandidates distanceTol tangentVelocityTol

def selectedCandidatesAt? {State : Type}
    (provider : ContactCandidateProvider State)
    (state : State)
    (policy : SupportPolicy)
    (distanceTol : Float := 0.0)
    (tangentVelocityTol : Float := 1.0e-9)
    (jacobianWidth? : Option Nat := none)
    (label : String := "") :
    Except String (Array ContactCandidate) := do
  let support ← provider.supportAt? state policy distanceTol tangentVelocityTol
    jacobianWidth? label
  support.selectedCandidates?

def runtimeSupportAt? {State : Type}
    (provider : ContactCandidateProvider State)
    (state : State)
    (policy : SupportPolicy)
    (distanceTol : Float := 0.0)
    (tangentVelocityTol : Float := 1.0e-9)
    (jacobianWidth? : Option Nat := none)
    (label : String := "") :
    Except String RuntimeSupport := do
  let support ← provider.supportAt? state policy distanceTol tangentVelocityTol
    jacobianWidth? label
  support.toRuntimeSupport?

def minimumSignedDistanceAt? {State : Type}
    (provider : ContactCandidateProvider State)
    (state : State)
    (jacobianWidth? : Option Nat := none) :
    Except String (Option Float) := do
  let set ← provider.validCandidatesAt? state jacobianWidth?
  pure set.minimumSignedDistance?

end ContactCandidateProvider

namespace HydroelasticPatchSupport

/--
Expose retained hydroelastic patches through the generic contact-support
primitive.  The patch provider remains free to keep polygon/mesh details
internally; the full-physics layer consumes only the selected ids and `J` rows.
-/
def equivalentContactSupport (support : HydroelasticPatchSupport) : ContactSupport :=
  {
    policy := support.policy
    candidates := support.patches.map (fun patch => patch.equivalentContactCandidate)
    selectedLocalIndices := support.selectedLocalIndices
    sourceCandidateCount? := support.sourcePatchCount?
    label := s!"hydroelastic-equivalent-contact-support:{support.label}"
  }

def selectedContactForces? (support : HydroelasticPatchSupport) :
    Except String (Array ContactForceScalars) := do
  support.validateGeometry?
  let selected ← support.selectedPatches?
  pure (selected.map (fun patch => patch.contactForceScalars))

end HydroelasticPatchSupport

/-!
## Compliant contact force primitive

This is the smallest reusable force law needed by the examples: normal penalty
plus velocity damping, with tangential damping clipped to the Coulomb cone.  A
future SAP, TAMSI, or complementarity solver can replace this force provider
while still returning `ContactForceScalars` at the same `J^T f` boundary.
-/

structure CompliantContactModel where
  normalStiffness : Float := 0.0
  normalDamping : Float := 0.0
  tangentDamping : Float := 0.0
  tangentDamping2 : Float := 0.0
  friction : CoulombFriction := CoulombFriction.frictionless
  label : String := ""
  deriving Repr, Inhabited

namespace CompliantContactModel

private def validateNonnegativeFinite (value : Float) (field label : String) :
    Except String Unit :=
  if !value.isFinite || value < 0.0 then
    .error s!"compliant contact model {label}: {field} must be nonnegative and finite, got {value}"
  else
    .ok ()

def validate? (model : CompliantContactModel) : Except String Unit := do
  validateNonnegativeFinite model.normalStiffness "normalStiffness" model.label
  validateNonnegativeFinite model.normalDamping "normalDamping" model.label
  validateNonnegativeFinite model.tangentDamping "tangentDamping" model.label
  validateNonnegativeFinite model.tangentDamping2 "tangentDamping2" model.label
  model.friction.validate? s!"compliant contact model {model.label}: friction"

private def clamp (lo hi x : Float) : Float :=
  if x < lo then lo else if x > hi then hi else x

def normalForce (model : CompliantContactModel) (candidate : ContactCandidate) : Float :=
  let penetration := max 0.0 (-candidate.signedDistance)
  let closingSpeed := max 0.0 (-candidate.normalVelocity)
  max 0.0 (model.normalStiffness * penetration + model.normalDamping * closingSpeed)

def tangentForce
    (model : CompliantContactModel)
    (candidate : ContactCandidate)
    (normalForce : Float) : Float :=
  let limit := model.friction.dynamicFriction * normalForce
  clamp (-limit) limit (-(model.tangentDamping) * candidate.tangentVelocity)

def tangentForce2
    (model : CompliantContactModel)
    (candidate : ContactCandidate)
    (normalForce : Float) : Float :=
  let limit := model.friction.dynamicFriction * normalForce
  clamp (-limit) limit (-(model.tangentDamping2) * candidate.tangentVelocity2)

def forceForCandidate? (model : CompliantContactModel) (candidate : ContactCandidate) :
    Except String ContactForceScalars := do
  model.validate?
  let normal := model.normalForce candidate
  pure (ContactForceScalars.fromCandidate3D candidate normal
    (model.tangentForce candidate normal)
    (model.tangentForce2 candidate normal))

def forcesForSupport? (model : CompliantContactModel) (support : ContactSupport) :
    Except String (Array ContactForceScalars) := do
  let selected ← support.selectedCandidates?
  let mut out : Array ContactForceScalars := #[]
  for candidate in selected do
    out := out.push (← model.forceForCandidate? candidate)
  pure out

end CompliantContactModel

end Tyr.EventSkeleton
