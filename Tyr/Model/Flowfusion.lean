import Tyr.Typed.Neural
import Tyr.TensorStruct
import Tyr.Module.Derive

/-!
  Minimal Flowfusion-style abstractions.

  This module provides lightweight wrappers used by BranchingFlows-like code:
  - MaskedState: state with conditioning/loss masks
  - Guide: wrapper for alternative prediction targets

  These are intentionally minimal and avoid committing to a full Flowfusion port.
-/

namespace torch.flowfusion

variable {cmaskSpec lmaskSpec : StaticSpec}

/-- Wrap a state with conditioning and loss masks. Masks are stored as `Static`
    so they are skipped by tensor traversal utilities. -/
structure MaskedState (α : Type) (cmaskSpec lmaskSpec : StaticSpec) where
  state : α
  cmask : Static (Tensor cmaskSpec)
  lmask : Static (Tensor lmaskSpec)
  deriving Repr

instance [TensorStruct α] : TensorStruct (MaskedState α cmaskSpec lmaskSpec) where
  map f x := { x with state := TensorStruct.map f x.state }
  mapM f x := do
    let state' ← TensorStruct.mapM f x.state
    pure { x with state := state' }
  zipWith f x y := { x with state := TensorStruct.zipWith f x.state y.state }
  fold f init x := TensorStruct.fold f init x.state

derive_tensor_struct_for MaskedState

namespace MaskedState

/-- Convenience constructor preserving both mask specifications. -/
def ofMasks (state : α) (cmask : Tensor cmaskSpec) (lmask : Tensor lmaskSpec) : MaskedState α cmaskSpec lmaskSpec :=
  { state, cmask := (cmask : Static (Tensor cmaskSpec)), lmask := (lmask : Static (Tensor lmaskSpec)) }

def getCmask (m : MaskedState α cmaskSpec lmaskSpec) : Tensor cmaskSpec := m.cmask.val
def getLmask (m : MaskedState α cmaskSpec lmaskSpec) : Tensor lmaskSpec := m.lmask.val

end MaskedState

/-- Wrapper for alternative prediction targets (e.g., tangent vectors).
    Masks are optional and stored as `Static`. -/
structure Guide (α : Type) (cmaskSpec lmaskSpec : StaticSpec) where
  value : α
  cmask : Option (Static (Tensor cmaskSpec)) := none
  lmask : Option (Static (Tensor lmaskSpec)) := none
  deriving Repr

instance [TensorStruct α] : TensorStruct (Guide α cmaskSpec lmaskSpec) where
  map f x := { x with value := TensorStruct.map f x.value }
  mapM f x := do
    let value' ← TensorStruct.mapM f x.value
    pure { x with value := value' }
  zipWith f x y := { x with value := TensorStruct.zipWith f x.value y.value }
  fold f init x := TensorStruct.fold f init x.value

derive_tensor_struct_for Guide

namespace Guide

def getCmask (g : Guide α cmaskSpec lmaskSpec) : Option (Tensor cmaskSpec) := g.cmask.map (·.val)
def getLmask (g : Guide α cmaskSpec lmaskSpec) : Option (Tensor lmaskSpec) := g.lmask.map (·.val)

end Guide

/-- Unwrap a masked state to its underlying state. -/
def unmask (m : MaskedState α cmaskSpec lmaskSpec) : α := m.state

/-- Wrap a state with the masks from another masked state. -/
def maskLike (x : α) (m : MaskedState α cmaskSpec lmaskSpec) : MaskedState α cmaskSpec lmaskSpec :=
  MaskedState.ofMasks x (MaskedState.getCmask m) (MaskedState.getLmask m)

end torch.flowfusion
