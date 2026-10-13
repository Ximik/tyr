import Tyr.DiffEq.Types
import Tyr.DiffEq.Path
import Tyr.DiffEq.Brownian

namespace torch
namespace DiffEq

/-! ## Term System

Terms capture vector fields, controls, and their bilinear interaction.
-/

structure AbstractTerm (Y VF Control Args : Type) where
  vf : Time → Y → Args → VF
  contr : Time → Time → Control
  prod : VF → Control → Y
  vf_prod : Time → Y → Args → Control → Y
  is_vf_expensive : Time → Time → Y → Args → Bool

/-- Typeclass view of terms for solver genericity. -/
class TermLike (τ : Type) (Y VF Control Args : Type) where
  vf : τ → Time → Y → Args → VF
  contr : τ → Time → Time → Control
  prod : τ → VF → Control → Y
  vf_prod : τ → Time → Y → Args → Control → Y
  is_vf_expensive : τ → Time → Time → Y → Args → Bool

/-- Additional interface for diffusion terms that can provide a Jacobian-vector product. -/
class DiffusionTermLike (τ : Type) (Y VF Control Args : Type) where
  jacobian_prod : τ → Time → Y → Args → Y

/-- Recursive structural metadata for PyTree-style term layouts. -/
inductive TermTree where
  | leaf (arity? : Option Nat)
  | pair (left right : TermTree)
  | array (size : Nat) (element : TermTree)
  deriving Repr, BEq

/-- Optional structural metadata for terms (arity/layout hints). -/
class TermShape (τ : Type) where
  arity? : τ → Option Nat
  layoutTag? : τ → Option String := fun _ => none
  partitionArities? : τ → Option (Array Nat) := fun _ => none
  tree? : τ → Option TermTree := fun _ => none

namespace TermShape

def combineArity (lhs rhs : Option Nat) : Option Nat :=
  match lhs, rhs with
  | some n1, some n2 => some (n1 + n2)
  | _, _ => none

def pairPartition (lhs rhs : Option Nat) : Option (Array Nat) :=
  match lhs, rhs with
  | some n1, some n2 => some #[n1, n2]
  | _, _ => none

def leafTreeFromArity? (arity? : Option Nat) : TermTree :=
  .leaf arity?

def treeOrLeaf (tree? : Option TermTree) (arity? : Option Nat) : TermTree :=
  tree?.getD (leafTreeFromArity? arity?)

end TermShape

namespace TermTree

def single : TermTree := .leaf (some 1)

def arity? : TermTree → Option Nat
  | .leaf arity? => arity?
  | .pair left right => TermShape.combineArity (arity? left) (arity? right)
  | .array size element =>
      match arity? element with
      | some elementArity => some (size * elementArity)
      | none => none

def partitionArities? : TermTree → Option (Array Nat)
  | .leaf _ => none
  | .pair left right => TermShape.pairPartition (arity? left) (arity? right)
  | .array size element =>
      match arity? element with
      | some elementArity => some (Array.replicate size elementArity)
      | none => none

def depth : TermTree → Nat
  | .leaf _ => 1
  | .pair left right => Nat.succ (Nat.max (depth left) (depth right))
  | .array _ element => Nat.succ (depth element)

def leafArities? : TermTree → Option (Array Nat)
  | .leaf arity? =>
      match arity? with
      | some n => some #[n]
      | none => none
  | .pair left right =>
      match leafArities? left, leafArities? right with
      | some leftArities, some rightArities => some (leftArities ++ rightArities)
      | _, _ => none
  | .array size element =>
      match leafArities? element with
      | some elementArities =>
          Id.run do
            let mut acc := #[]
            for _ in [:size] do
              acc := acc ++ elementArities
            some acc
      | none => none

def layoutTag : TermTree → String
  | .leaf _ => "single"
  | .pair _ _ => "pair"
  | .array _ _ => "array"

end TermTree

/-- Default `TermShape` for a single (non-composite) term. -/
@[reducible] def defaultSingleTermShape (τ : Type) : TermShape τ where
  arity? _ := some 1
  layoutTag? _ := some "single"
  tree? _ := some TermTree.single

instance : TermLike (AbstractTerm Y VF Control Args) Y VF Control Args where
  vf term := term.vf
  contr term := term.contr
  prod term := term.prod
  vf_prod term := term.vf_prod
  is_vf_expensive term := term.is_vf_expensive

instance : TermShape (AbstractTerm Y VF Control Args) := defaultSingleTermShape _

namespace AbstractTerm

def ofTermLike {τ Y VF Control Args : Type} [TermLike τ Y VF Control Args] (term : τ) :
    AbstractTerm Y VF Control Args := {
  vf := (inferInstance : TermLike τ Y VF Control Args).vf term
  contr := (inferInstance : TermLike τ Y VF Control Args).contr term
  prod := (inferInstance : TermLike τ Y VF Control Args).prod term
  vf_prod := (inferInstance : TermLike τ Y VF Control Args).vf_prod term
  is_vf_expensive := (inferInstance : TermLike τ Y VF Control Args).is_vf_expensive term
}

end AbstractTerm

/-- ODE term: vector field with time as the control. -/
structure ODETerm (Y Args : Type) where
  vectorField : Time → Y → Args → Y

instance [DiffEqSpace Y] : TermLike (ODETerm Y Args) Y Y Time Args where
  vf term := term.vectorField
  contr _ t0 t1 := t1 - t0
  prod _ vf control := DiffEqSpace.scale control vf
  vf_prod term t y args control :=
    DiffEqSpace.scale control (term.vectorField t y args)
  is_vf_expensive _ _ _ _ _ := false

instance : TermShape (ODETerm Y Args) := defaultSingleTermShape _

namespace ODETerm

def toAbstract [DiffEqSpace Y] (term : ODETerm Y Args) : AbstractTerm Y Y Time Args :=
  AbstractTerm.ofTermLike term

end ODETerm

/-- Control term: general control with a user-provided product. -/
structure ControlTerm (Y VF Control Args : Type) where
  vectorField : Time → Y → Args → VF
  control : Time → Time → Control
  prod : VF → Control → Y
  controlDerivative? : Option (Time → Bool → Control) := none

instance : TermLike (ControlTerm Y VF Control Args) Y VF Control Args where
  vf term := term.vectorField
  contr term := term.control
  prod term := term.prod
  vf_prod term t y args control := term.prod (term.vectorField t y args) control
  is_vf_expensive _ _ _ _ _ := false

instance : TermShape (ControlTerm Y VF Control Args) := defaultSingleTermShape _

namespace ControlTerm

def toAbstract (term : ControlTerm Y VF Control Args) : AbstractTerm Y VF Control Args :=
  AbstractTerm.ofTermLike term

def ofTermLike {τ Y VF Control Args : Type} [TermLike τ Y VF Control Args] (term : τ)
    (controlDerivative? : Option (Time → Bool → Control) := none) :
    ControlTerm Y VF Control Args :=
  {
    vectorField := (inferInstance : TermLike τ Y VF Control Args).vf term
    control := (inferInstance : TermLike τ Y VF Control Args).contr term
    prod := (inferInstance : TermLike τ Y VF Control Args).prod term
    controlDerivative? := controlDerivative?
  }

def ofAbstract (term : AbstractTerm Y VF Control Args)
    (controlDerivative? : Option (Time → Bool → Control) := none) :
    ControlTerm Y VF Control Args :=
  ofTermLike term controlDerivative?

def withControlDerivative (term : ControlTerm Y VF Control Args)
    (controlDerivative : Time → Bool → Control) : ControlTerm Y VF Control Args :=
  { term with controlDerivative? := some controlDerivative }

def clearControlDerivative (term : ControlTerm Y VF Control Args) : ControlTerm Y VF Control Args :=
  { term with controlDerivative? := none }

def withPath (term : ControlTerm Y VF Control Args) (path : AbstractPath Control)
    (left : Bool := true) : ControlTerm Y VF Control Args :=
  {
    term with
      control := fun t0 t1 => path.evaluate t0 (some t1) left
      controlDerivative? := path.derivativeFn?
  }

def toODEWithDerivative (term : ControlTerm Y VF Control Args)
    (controlDerivative : Time → Bool → Control) (left : Bool := true) :
    ODETerm Y Args :=
  {
    vectorField := fun t y args =>
      term.prod (term.vectorField t y args) (controlDerivative t left)
  }

def toODE? (term : ControlTerm Y VF Control Args) (left : Bool := true) :
    Option (ODETerm Y Args) :=
  match term.controlDerivative? with
  | some controlDerivative => some (toODEWithDerivative term controlDerivative left)
  | none => none

def ofPathWithSide (vectorField : Time → Y → Args → VF) (path : AbstractPath Control)
    (prod : VF → Control → Y) (left : Bool := true) : ControlTerm Y VF Control Args :=
  {
    vectorField := vectorField
    control := fun t0 t1 => path.evaluate t0 (some t1) left
    prod := prod
    controlDerivative? := path.derivativeFn?
  }

def ofPath (vectorField : Time → Y → Args → VF) (path : AbstractPath Control)
    (prod : VF → Control → Y) : ControlTerm Y VF Control Args :=
  ofPathWithSide vectorField path prod true

def ofDifferentiablePathWithSide (vectorField : Time → Y → Args → VF) (path : AbstractPath Control)
    (prod : VF → Control → Y) (controlDerivative : Time → Bool → Control)
    (left : Bool := true) :
    ControlTerm Y VF Control Args :=
  withControlDerivative (ofPathWithSide vectorField path prod left) controlDerivative

def ofDifferentiablePath (vectorField : Time → Y → Args → VF) (path : AbstractPath Control)
    (prod : VF → Control → Y) (controlDerivative : Time → Bool → Control) :
    ControlTerm Y VF Control Args :=
  ofDifferentiablePathWithSide vectorField path prod controlDerivative true

def derivativeAware? (term : ControlTerm Y VF Control Args) : Bool :=
  term.controlDerivative?.isSome

end ControlTerm

/-- Validate that a scalar scale parameter is finite and nonnegative. -/
def validateScale (name : String) (value : Scalar) : Option String :=
  if !Float.isFinite value then
    some s!"{name} must be finite, got {value}"
  else if value < 0.0 then
    some s!"{name} must be nonnegative, got {value}"
  else
    none

/-- Diffusion term with a Jacobian-vector product for Milstein-like solvers. -/
structure DiffusionTerm (Y VF Control Args : Type) where
  vectorField : Time → Y → Args → VF
  control : Time → Time → Control
  prod : VF → Control → Y
  jacobianProd : Time → Y → Args → Y

instance : TermLike (DiffusionTerm Y VF Control Args) Y VF Control Args where
  vf term := term.vectorField
  contr term := term.control
  prod term := term.prod
  vf_prod term t y args control := term.prod (term.vectorField t y args) control
  is_vf_expensive _ _ _ _ _ := false

instance : DiffusionTermLike (DiffusionTerm Y VF Control Args) Y VF Control Args where
  jacobian_prod term := term.jacobianProd

instance : TermShape (DiffusionTerm Y VF Control Args) := defaultSingleTermShape _

namespace DiffusionTerm

def toAbstract (term : DiffusionTerm Y VF Control Args) : AbstractTerm Y VF Control Args :=
  AbstractTerm.ofTermLike term

def ofPath (vectorField : Time → Y → Args → VF) (path : AbstractPath Control)
    (prod : VF → Control → Y) (jacobianProd : Time → Y → Args → Y) :
    DiffusionTerm Y VF Control Args :=
  {
    vectorField := vectorField
    control := fun t0 t1 => path.evaluate t0 (some t1) true
    prod := prod
    jacobianProd := jacobianProd
  }

end DiffusionTerm

/-- Underdamped Langevin drift term on pair state `(position, velocity)`. -/
structure UnderdampedLangevinDriftTerm (X Args : Type) where
  gradPotential : Time → X → Args → X
  gamma : Time → X → X → Args → Scalar := fun _ _ _ _ => 1.0
  u : Time → X → X → Args → Scalar := fun _ _ _ _ => 1.0
  argsValid : Time → X → X → Args → Bool := fun _ _ _ _ => true

namespace UnderdampedLangevinDriftTerm

def validate? (term : UnderdampedLangevinDriftTerm X Args)
    (t : Time) (y : X × X) (args : Args) : Option String :=
  let x := y.1
  let v := y.2
  if !(term.argsValid t x v args) then
    some "argsValid predicate rejected (t, x, v, args)."
  else
    match validateScale "gamma" (term.gamma t x v args) with
    | some msg => some msg
    | none => validateScale "u" (term.u t x v args)

private def ensureValid (term : UnderdampedLangevinDriftTerm X Args)
    (t : Time) (y : X × X) (args : Args) : Unit :=
  match validate? term t y args with
  | none => ()
  | some msg => panic! s!"UnderdampedLangevinDriftTerm invalid inputs: {msg}"

private def vfCore [DiffEqSpace X] (term : UnderdampedLangevinDriftTerm X Args)
    (t : Time) (y : X × X) (args : Args) : X × X :=
  let x := y.1
  let v := y.2
  let grad := term.gradPotential t x args
  let gamma := term.gamma t x v args
  let u := term.u t x v args
  let damping := DiffEqSpace.scale (-gamma) v
  let restoring := DiffEqSpace.scale (-u) grad
  (v, DiffEqSpace.add damping restoring)

instance [DiffEqSpace X] :
    TermLike (UnderdampedLangevinDriftTerm X Args) (X × X) (X × X) Time Args where
  vf term t y args :=
    let _ := ensureValid term t y args
    vfCore term t y args
  contr _ t0 t1 := t1 - t0
  prod _ vf control := (DiffEqSpace.scale control vf.1, DiffEqSpace.scale control vf.2)
  vf_prod term t y args control :=
    let _ := ensureValid term t y args
    let vf := vfCore term t y args
    (DiffEqSpace.scale control vf.1, DiffEqSpace.scale control vf.2)
  is_vf_expensive _ _ _ _ _ := false

instance : TermShape (UnderdampedLangevinDriftTerm X Args) := defaultSingleTermShape _

def toAbstract [DiffEqSpace X] (term : UnderdampedLangevinDriftTerm X Args) :
    AbstractTerm (X × X) (X × X) Time Args :=
  AbstractTerm.ofTermLike term

end UnderdampedLangevinDriftTerm

/-- Underdamped Langevin diffusion term on pair state `(position, velocity)`.
Noise is injected into the velocity component only.
-/
structure UnderdampedLangevinDiffusionTerm (X Args : Type) where
  control : Time → Time → X
  controlH? : Option (Time → Time → X) := none
  controlK? : Option (Time → Time → X) := none
  gamma : Time → X → X → Args → Scalar := fun _ _ _ _ => 1.0
  u : Time → X → X → Args → Scalar := fun _ _ _ _ => 1.0
  argsValid : Time → X → X → Args → Bool := fun _ _ _ _ => true

class UnderdampedLevyControl (Control X : Type) where
  projW : Control → X
  projH? : Option (Control → X) := none
  projK? : Option (Control → X) := none

instance : UnderdampedLevyControl X X where
  projW control := control

instance : UnderdampedLevyControl (SpaceTimeLevyArea Time X) X where
  projW control := control.W
  projH? := some fun control => control.H

instance : UnderdampedLevyControl (SpaceTimeTimeLevyArea Time X) X where
  projW control := control.W
  projH? := some fun control => control.H
  projK? := some fun control => control.K

namespace UnderdampedLangevinDiffusionTerm

def validate? (term : UnderdampedLangevinDiffusionTerm X Args)
    (t : Time) (y : X × X) (args : Args) : Option String :=
  let x := y.1
  let v := y.2
  if !(term.argsValid t x v args) then
    some "argsValid predicate rejected (t, x, v, args)."
  else
    let gamma := term.gamma t x v args
    let u := term.u t x v args
    match validateScale "gamma" gamma with
    | some msg => some msg
    | none =>
        match validateScale "u" u with
        | some msg => some msg
        | none =>
            let sigma2 := 2.0 * gamma * u
            if !Float.isFinite sigma2 then
              some s!"2*gamma*u must be finite, got {sigma2}"
            else if sigma2 < 0.0 then
              some s!"2*gamma*u must be nonnegative, got {sigma2}"
            else
              none

private def ensureValid (term : UnderdampedLangevinDiffusionTerm X Args)
    (t : Time) (y : X × X) (args : Args) : Unit :=
  match validate? term t y args with
  | none => ()
  | some msg => panic! s!"UnderdampedLangevinDiffusionTerm invalid inputs: {msg}"

private def noiseScale (term : UnderdampedLangevinDiffusionTerm X Args)
    (t : Time) (y : X × X) (args : Args) : Scalar :=
  let x := y.1
  let v := y.2
  let gamma := term.gamma t x v args
  let u := term.u t x v args
  Float.sqrt (2.0 * gamma * u)

def controlW (term : UnderdampedLangevinDiffusionTerm X Args) (t0 t1 : Time) : X :=
  term.control t0 t1

def controlH [DiffEqSpace X] (term : UnderdampedLangevinDiffusionTerm X Args)
    (t0 t1 : Time) : X :=
  match term.controlH? with
  | some h => h t0 t1
  | none => DiffEqSpace.scale 0.0 (term.control t0 t1)

def controlK [DiffEqSpace X] (term : UnderdampedLangevinDiffusionTerm X Args)
    (t0 t1 : Time) : X :=
  match term.controlK? with
  | some k => k t0 t1
  | none => DiffEqSpace.scale 0.0 (term.control t0 t1)

def withControlH (term : UnderdampedLangevinDiffusionTerm X Args)
    (controlH : Time → Time → X) : UnderdampedLangevinDiffusionTerm X Args :=
  { term with controlH? := some controlH }

def withControlK (term : UnderdampedLangevinDiffusionTerm X Args)
    (controlK : Time → Time → X) : UnderdampedLangevinDiffusionTerm X Args :=
  { term with controlK? := some controlK }

def withControlsHK (term : UnderdampedLangevinDiffusionTerm X Args)
    (controlH : Time → Time → X) (controlK : Time → Time → X) :
    UnderdampedLangevinDiffusionTerm X Args :=
  { term with controlH? := some controlH, controlK? := some controlK }

instance [DiffEqSpace X] :
    TermLike (UnderdampedLangevinDiffusionTerm X Args) (X × X) Scalar
      (SpaceTimeTimeLevyArea Time X) Args where
  vf term t y args :=
    let _ := ensureValid term t y args
    noiseScale term t y args
  contr term t0 t1 := {
    dt := t1 - t0
    W := term.control t0 t1
    H := controlH term t0 t1
    K := controlK term t0 t1
  }
  prod _ vf control :=
    let zero := DiffEqSpace.scale 0.0 control.W
    (zero, DiffEqSpace.scale vf control.W)
  vf_prod term t y args control :=
    let _ := ensureValid term t y args
    let sigma := noiseScale term t y args
    let zero := DiffEqSpace.scale 0.0 control.W
    (zero, DiffEqSpace.scale sigma control.W)
  is_vf_expensive _ _ _ _ _ := false

instance : TermShape (UnderdampedLangevinDiffusionTerm X Args) := defaultSingleTermShape _

def toAbstract [DiffEqSpace X] (term : UnderdampedLangevinDiffusionTerm X Args) :
    AbstractTerm (X × X) Scalar (SpaceTimeTimeLevyArea Time X) Args :=
  AbstractTerm.ofTermLike term

private def ofProjectedPathWithSide {Control : Type}
    [UnderdampedLevyControl Control X] (path : AbstractPath Control)
    (gamma : Time → X → X → Args → Scalar := fun _ _ _ _ => 1.0)
    (u : Time → X → X → Args → Scalar := fun _ _ _ _ => 1.0)
    (argsValid : Time → X → X → Args → Bool := fun _ _ _ _ => true)
    (left : Bool := true) :
    UnderdampedLangevinDiffusionTerm X Args :=
  let proj : UnderdampedLevyControl Control X := inferInstance
  {
    control := fun t0 t1 => proj.projW (path.evaluate t0 (some t1) left)
    controlH? :=
      proj.projH?.map fun projectH =>
        fun t0 t1 => projectH (path.evaluate t0 (some t1) left)
    controlK? :=
      proj.projK?.map fun projectK =>
        fun t0 t1 => projectK (path.evaluate t0 (some t1) left)
    gamma := gamma
    u := u
    argsValid := argsValid
  }

def ofPathWithSide (path : AbstractPath X)
    (gamma : Time → X → X → Args → Scalar := fun _ _ _ _ => 1.0)
    (u : Time → X → X → Args → Scalar := fun _ _ _ _ => 1.0)
    (argsValid : Time → X → X → Args → Bool := fun _ _ _ _ => true)
    (left : Bool := true) :
    UnderdampedLangevinDiffusionTerm X Args :=
  ofProjectedPathWithSide path gamma u argsValid left

def ofPath (path : AbstractPath X)
    (gamma : Time → X → X → Args → Scalar := fun _ _ _ _ => 1.0)
    (u : Time → X → X → Args → Scalar := fun _ _ _ _ => 1.0)
    (argsValid : Time → X → X → Args → Bool := fun _ _ _ _ => true) :
    UnderdampedLangevinDiffusionTerm X Args :=
  ofPathWithSide path gamma u argsValid true

def ofSpaceTimePathWithSide (path : AbstractPath (SpaceTimeLevyArea Time X))
    (gamma : Time → X → X → Args → Scalar := fun _ _ _ _ => 1.0)
    (u : Time → X → X → Args → Scalar := fun _ _ _ _ => 1.0)
    (argsValid : Time → X → X → Args → Bool := fun _ _ _ _ => true)
    (left : Bool := true) :
    UnderdampedLangevinDiffusionTerm X Args :=
  ofProjectedPathWithSide path gamma u argsValid left

def ofSpaceTimePath (path : AbstractPath (SpaceTimeLevyArea Time X))
    (gamma : Time → X → X → Args → Scalar := fun _ _ _ _ => 1.0)
    (u : Time → X → X → Args → Scalar := fun _ _ _ _ => 1.0)
    (argsValid : Time → X → X → Args → Bool := fun _ _ _ _ => true) :
    UnderdampedLangevinDiffusionTerm X Args :=
  ofSpaceTimePathWithSide path gamma u argsValid true

def ofSpaceTimeTimePathWithSide (path : AbstractPath (SpaceTimeTimeLevyArea Time X))
    (gamma : Time → X → X → Args → Scalar := fun _ _ _ _ => 1.0)
    (u : Time → X → X → Args → Scalar := fun _ _ _ _ => 1.0)
    (argsValid : Time → X → X → Args → Bool := fun _ _ _ _ => true)
    (left : Bool := true) :
    UnderdampedLangevinDiffusionTerm X Args :=
  ofProjectedPathWithSide path gamma u argsValid left

def ofSpaceTimeTimePath (path : AbstractPath (SpaceTimeTimeLevyArea Time X))
    (gamma : Time → X → X → Args → Scalar := fun _ _ _ _ => 1.0)
    (u : Time → X → X → Args → Scalar := fun _ _ _ _ => 1.0)
    (argsValid : Time → X → X → Args → Bool := fun _ _ _ _ => true) :
    UnderdampedLangevinDiffusionTerm X Args :=
  ofSpaceTimeTimePathWithSide path gamma u argsValid true

def ofBrownianPathWithSide {Control : Type} [UnderdampedLevyControl Control X]
    (path : AbstractBrownianPath Control)
    (gamma : Time → X → X → Args → Scalar := fun _ _ _ _ => 1.0)
    (u : Time → X → X → Args → Scalar := fun _ _ _ _ => 1.0)
    (argsValid : Time → X → X → Args → Bool := fun _ _ _ _ => true)
    (left : Bool := true) :
    UnderdampedLangevinDiffusionTerm X Args :=
  ofProjectedPathWithSide path.toPath gamma u argsValid left

def ofBrownianPath {Control : Type} [UnderdampedLevyControl Control X]
    (path : AbstractBrownianPath Control)
    (gamma : Time → X → X → Args → Scalar := fun _ _ _ _ => 1.0)
    (u : Time → X → X → Args → Scalar := fun _ _ _ _ => 1.0)
    (argsValid : Time → X → X → Args → Bool := fun _ _ _ _ => true) :
    UnderdampedLangevinDiffusionTerm X Args :=
  ofBrownianPathWithSide path gamma u argsValid true

end UnderdampedLangevinDiffusionTerm

/-- Simple wrapper for forwarding term behavior. -/
structure WrapTerm (Term : Type) where
  term : Term

instance {Term Y VF Control Args : Type} [TermLike Term Y VF Control Args] :
    TermLike (WrapTerm Term) Y VF Control Args where
  vf t := (inferInstance : TermLike Term Y VF Control Args).vf t.term
  contr t := (inferInstance : TermLike Term Y VF Control Args).contr t.term
  prod t := (inferInstance : TermLike Term Y VF Control Args).prod t.term
  vf_prod t := (inferInstance : TermLike Term Y VF Control Args).vf_prod t.term
  is_vf_expensive t := (inferInstance : TermLike Term Y VF Control Args).is_vf_expensive t.term

instance {Term Y VF Control Args : Type} [DiffusionTermLike Term Y VF Control Args] :
    DiffusionTermLike (WrapTerm Term) Y VF Control Args where
  jacobian_prod t := (inferInstance : DiffusionTermLike Term Y VF Control Args).jacobian_prod t.term

instance {Term : Type} [TermShape Term] : TermShape (WrapTerm Term) where
  arity? t := (inferInstance : TermShape Term).arity? t.term
  layoutTag? t := (inferInstance : TermShape Term).layoutTag? t.term
  partitionArities? t := (inferInstance : TermShape Term).partitionArities? t.term
  tree? t := (inferInstance : TermShape Term).tree? t.term

namespace WrapTerm

def toAbstract {Term Y VF Control Args : Type} [TermLike Term Y VF Control Args]
    (term : WrapTerm Term) :
    AbstractTerm Y VF Control Args :=
  AbstractTerm.ofTermLike term

end WrapTerm

/-- Native product terms behave additively and can nest recursively like PyTrees. -/
instance {T1 T2 Y VF1 VF2 C1 C2 Args : Type}
    [TermLike T1 Y VF1 C1 Args] [TermLike T2 Y VF2 C2 Args]
    [DiffEqSpace Y] :
    TermLike (T1 × T2) Y (VF1 × VF2) (C1 × C2) Args where
  vf term t y args :=
    ((inferInstance : TermLike T1 Y VF1 C1 Args).vf term.1 t y args,
     (inferInstance : TermLike T2 Y VF2 C2 Args).vf term.2 t y args)
  contr term t0 t1 :=
    ((inferInstance : TermLike T1 Y VF1 C1 Args).contr term.1 t0 t1,
     (inferInstance : TermLike T2 Y VF2 C2 Args).contr term.2 t0 t1)
  prod term vf control :=
    let y1 := (inferInstance : TermLike T1 Y VF1 C1 Args).prod term.1 vf.1 control.1
    let y2 := (inferInstance : TermLike T2 Y VF2 C2 Args).prod term.2 vf.2 control.2
    DiffEqSpace.add y1 y2
  vf_prod term t y args control :=
    let y1 := (inferInstance : TermLike T1 Y VF1 C1 Args).vf_prod term.1 t y args control.1
    let y2 := (inferInstance : TermLike T2 Y VF2 C2 Args).vf_prod term.2 t y args control.2
    DiffEqSpace.add y1 y2
  is_vf_expensive term t0 t1 y args :=
    (inferInstance : TermLike T1 Y VF1 C1 Args).is_vf_expensive term.1 t0 t1 y args ||
    (inferInstance : TermLike T2 Y VF2 C2 Args).is_vf_expensive term.2 t0 t1 y args

instance {T1 T2 : Type} [TermShape T1] [TermShape T2] : TermShape (T1 × T2) where
  arity? term :=
    TermShape.combineArity
      ((inferInstance : TermShape T1).arity? term.1)
      ((inferInstance : TermShape T2).arity? term.2)
  layoutTag? _ := some "pair"
  partitionArities? term :=
    TermShape.pairPartition
      ((inferInstance : TermShape T1).arity? term.1)
      ((inferInstance : TermShape T2).arity? term.2)
  tree? term :=
    let shape1 := (inferInstance : TermShape T1)
    let shape2 := (inferInstance : TermShape T2)
    let tree1 :=
      TermShape.treeOrLeaf (shape1.tree? term.1) (shape1.arity? term.1)
    let tree2 :=
      TermShape.treeOrLeaf (shape2.tree? term.2) (shape2.arity? term.2)
    some (.pair tree1 tree2)

/-- Additive combination of two terms (pair version). -/
structure MultiTerm (T1 T2 : Type) where
  term1 : T1
  term2 : T2

namespace MultiTerm

def ofProd (term : T1 × T2) : MultiTerm T1 T2 :=
  { term1 := term.1, term2 := term.2 }

def toProd (term : MultiTerm T1 T2) : T1 × T2 :=
  (term.term1, term.term2)

end MultiTerm

abbrev MultiTerm3 (T1 T2 T3 : Type) : Type :=
  MultiTerm (MultiTerm T1 T2) T3

abbrev MultiTerm4 (T1 T2 T3 T4 : Type) : Type :=
  MultiTerm (MultiTerm3 T1 T2 T3) T4

abbrev MultiTerm5 (T1 T2 T3 T4 T5 : Type) : Type :=
  MultiTerm (MultiTerm4 T1 T2 T3 T4) T5

abbrev MultiTerm6 (T1 T2 T3 T4 T5 T6 : Type) : Type :=
  MultiTerm (MultiTerm5 T1 T2 T3 T4 T5) T6

namespace MultiTerm

def ofProd3 (term : (T1 × T2) × T3) : MultiTerm3 T1 T2 T3 :=
  { term1 := ofProd term.1, term2 := term.2 }

def toProd3 (term : MultiTerm3 T1 T2 T3) : (T1 × T2) × T3 :=
  (toProd term.term1, term.term2)

def append (term : MultiTerm T1 T2) (term3 : T3) : MultiTerm3 T1 T2 T3 :=
  { term1 := term, term2 := term3 }

def prepend (term0 : T0) (term : MultiTerm T1 T2) : MultiTerm3 T0 T1 T2 :=
  { term1 := { term1 := term0, term2 := term.term1 }, term2 := term.term2 }

def of3 (term1 : T1) (term2 : T2) (term3 : T3) : MultiTerm3 T1 T2 T3 :=
  append { term1 := term1, term2 := term2 } term3

def of4 (term1 : T1) (term2 : T2) (term3 : T3) (term4 : T4) : MultiTerm4 T1 T2 T3 T4 :=
  append (of3 term1 term2 term3) term4

def of5 (term1 : T1) (term2 : T2) (term3 : T3) (term4 : T4) (term5 : T5) :
    MultiTerm5 T1 T2 T3 T4 T5 :=
  append (of4 term1 term2 term3 term4) term5

def of6 (term1 : T1) (term2 : T2) (term3 : T3) (term4 : T4) (term5 : T5) (term6 : T6) :
    MultiTerm6 T1 T2 T3 T4 T5 T6 :=
  append (of5 term1 term2 term3 term4 term5) term6

def reassocRight (term : MultiTerm (MultiTerm T1 T2) T3) : MultiTerm T1 (MultiTerm T2 T3) :=
  {
    term1 := term.term1.term1
    term2 := {
      term1 := term.term1.term2
      term2 := term.term2
    }
  }

def reassocLeft (term : MultiTerm T1 (MultiTerm T2 T3)) : MultiTerm (MultiTerm T1 T2) T3 :=
  {
    term1 := {
      term1 := term.term1
      term2 := term.term2.term1
    }
    term2 := term.term2.term2
  }

end MultiTerm

instance {T1 T2 Y VF1 VF2 C1 C2 Args : Type}
    [TermLike T1 Y VF1 C1 Args] [TermLike T2 Y VF2 C2 Args]
    [DiffEqSpace Y] :
    TermLike (MultiTerm T1 T2) Y (VF1 × VF2) (C1 × C2) Args where
  vf term := (inferInstance : TermLike (T1 × T2) Y (VF1 × VF2) (C1 × C2) Args).vf (MultiTerm.toProd term)
  contr term := (inferInstance : TermLike (T1 × T2) Y (VF1 × VF2) (C1 × C2) Args).contr (MultiTerm.toProd term)
  prod term := (inferInstance : TermLike (T1 × T2) Y (VF1 × VF2) (C1 × C2) Args).prod (MultiTerm.toProd term)
  vf_prod term := (inferInstance : TermLike (T1 × T2) Y (VF1 × VF2) (C1 × C2) Args).vf_prod (MultiTerm.toProd term)
  is_vf_expensive term := (inferInstance : TermLike (T1 × T2) Y (VF1 × VF2) (C1 × C2) Args).is_vf_expensive (MultiTerm.toProd term)

instance {T1 T2 : Type} [TermShape T1] [TermShape T2] : TermShape (MultiTerm T1 T2) where
  arity? term := (inferInstance : TermShape (T1 × T2)).arity? (MultiTerm.toProd term)
  layoutTag? term := (inferInstance : TermShape (T1 × T2)).layoutTag? (MultiTerm.toProd term)
  partitionArities? term := (inferInstance : TermShape (T1 × T2)).partitionArities? (MultiTerm.toProd term)
  tree? term := (inferInstance : TermShape (T1 × T2)).tree? (MultiTerm.toProd term)

def MultiTerm.toAbstract {T1 T2 Y VF1 VF2 C1 C2 Args : Type}
    [TermLike T1 Y VF1 C1 Args] [TermLike T2 Y VF2 C2 Args] [DiffEqSpace Y]
    (term : MultiTerm T1 T2) :
    AbstractTerm Y (VF1 × VF2) (C1 × C2) Args :=
  AbstractTerm.ofTermLike term

/-- Homogeneous additive combination of 1+ terms (array-like version). -/
structure MultiTermArray (Term : Type) where
  head : Term
  tail : Array Term := #[]

instance {Term Y VF Control Args : Type}
    [TermLike Term Y VF Control Args] [DiffEqSpace Y]
    [Inhabited Y] :
    TermLike (MultiTermArray Term) Y (Array VF) (Array Control) Args where
  vf terms t y args :=
    let termLike := (inferInstance : TermLike Term Y VF Control Args)
    (#[terms.head] ++ terms.tail).map (fun term => termLike.vf term t y args)
  contr terms t0 t1 :=
    let termLike := (inferInstance : TermLike Term Y VF Control Args)
    (#[terms.head] ++ terms.tail).map (fun term => termLike.contr term t0 t1)
  prod terms vf control :=
    let termLike := (inferInstance : TermLike Term Y VF Control Args)
    match vf[0]?, control[0]? with
    | some vf0, some control0 =>
        let y0 := termLike.prod terms.head vf0 control0
        Id.run do
          let mut acc := y0
          for offset in [:terms.tail.size] do
            let i := offset + 1
            match terms.tail[offset]?, vf[i]?, control[i]? with
            | some term, some vf_i, some control_i =>
                acc := DiffEqSpace.add acc (termLike.prod term vf_i control_i)
            | _, _, _ => ()
          acc
    | _, _ => default
  vf_prod terms t y args control :=
    let termLike := (inferInstance : TermLike Term Y VF Control Args)
    match control[0]? with
    | some control0 =>
        let y0 := termLike.vf_prod terms.head t y args control0
        Id.run do
          let mut acc := y0
          for offset in [:terms.tail.size] do
            let i := offset + 1
            match terms.tail[offset]?, control[i]? with
            | some term, some control_i =>
                acc := DiffEqSpace.add acc (termLike.vf_prod term t y args control_i)
            | _, _ => ()
          acc
    | none => default
  is_vf_expensive terms t0 t1 y args :=
    let termLike := (inferInstance : TermLike Term Y VF Control Args)
    Id.run do
      let mut expensive := termLike.is_vf_expensive terms.head t0 t1 y args
      for offset in [:terms.tail.size] do
        match terms.tail[offset]? with
        | some term => expensive := expensive || termLike.is_vf_expensive term t0 t1 y args
        | none => ()
      expensive

instance {Term : Type} : TermShape (MultiTermArray Term) where
  arity? terms := some (terms.tail.size + 1)
  layoutTag? _ := some "array"
  partitionArities? terms := some (Array.replicate (terms.tail.size + 1) 1)
  tree? terms := some (.array (terms.tail.size + 1) TermTree.single)

namespace MultiTermArray

def singleton (term : Term) : MultiTermArray Term :=
  { head := term }

def toArray (terms : MultiTermArray Term) : Array Term :=
  #[terms.head] ++ terms.tail

def size (terms : MultiTermArray Term) : Nat :=
  terms.tail.size + 1

def push (terms : MultiTermArray Term) (term : Term) : MultiTermArray Term :=
  { terms with tail := terms.tail.push term }

def appendArray (terms : MultiTermArray Term) (extraTerms : Array Term) : MultiTermArray Term :=
  { terms with tail := terms.tail ++ extraTerms }

def ofArray? (terms : Array Term) : Option (MultiTermArray Term) :=
  match terms[0]? with
  | none => none
  | some head =>
      some {
        head := head
        tail := terms.extract 1 terms.size
      }

def mapTerms (terms : MultiTermArray Term) (f : Term → Term₂) : MultiTermArray Term₂ :=
  {
    head := f terms.head
    tail := terms.tail.map f
  }

def toAbstract {Term Y VF Control Args : Type}
    [TermLike Term Y VF Control Args] [DiffEqSpace Y]
    [Inhabited Y]
    (terms : MultiTermArray Term) :
    AbstractTerm Y (Array VF) (Array Control) Args :=
  AbstractTerm.ofTermLike terms

end MultiTermArray

end DiffEq
end torch
