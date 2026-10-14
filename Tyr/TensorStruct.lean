import Tyr.Typed.Neural

/-!
# Tyr.TensorStruct

`Tyr.TensorStruct` defines Tyr's tensor-tree abstraction: a uniform way to traverse,
transform, and combine nested parameter/state structures that contain tensors.
Conceptually, it is similar to JAX PyTrees, specialized for Tyr's indexed `Tensor σ`.

In practice, this is the foundation that lets training, checkpointing, optimizers,
and model utilities operate generically over arbitrary model parameter types.

## Why This Exists

Real models are not single tensors: they are nested structures of tensors plus metadata
(configuration, counters, cached values, optional branches, arrays/lists of blocks, etc.).
`TensorStruct` gives one common interface for these structures so higher-level code can be
written once and reused across many models.

## Major Components

- `TensorStruct`: typeclass with four core operations over tensor leaves:
  `map`, `mapM`, `zipWith`, and `fold`.
- `Static α`: marks non-tensor metadata that should be skipped during traversal.
- `Frozen σ`: marks tensor fields that should remain in the structure but are typically
  treated as non-trainable.
- `Vector n α`: length-indexed container for compile-time-safe `zipWith`.
- `TensorStruct` utilities: `count`, `grads`, `zeroGrads`, `detach`,
  `requiresGrad`, `makeLeafParams`, `castDType`, `scale`, `add`, `sub`.

## Traversal Semantics

At a high level:

- `map` applies a pure tensor-to-tensor transform at every tensor leaf.
- `mapM` is the monadic form, useful for device moves and IO-backed transforms.
- `zipWith` combines two structures with the same shape-of-structure.
- `fold` reduces all tensor leaves into a summary value.

Wrapper behavior:

- `Static α` is ignored by tensor transforms (passed through unchanged).
- `Frozen σ` still traverses as a tensor leaf, but can be treated specially by your logic.

## Required Import for Deriving and Tensor Leaves

This module defines the `TensorStruct` class and instances for containers
(`Array`, `List`, `Vector`, `Option`), wrappers (`Static`, `Frozen`), and
scalars, including indexed `Tensor σ` leaves. The `deriving TensorStruct`
handler lives in `Tyr.Module.Derive`, so import that module when deriving
instances.
Traversal callbacks preserve each leaf's complete `StaticSpec` index. A map
cannot change a leaf's shape, concrete dtype or pinned device policy. Dynamic
dtype leaves can be explicitly recast with `TensorStruct.castDType`; changing a
concrete dtype contract requires a model conversion with a different result type.

## Examples

### 1) Define TensorStruct-aware model parameters

```lean
import Tyr.Module.Derive

open torch

structure LinearParams (inDim outDim : UInt64) where
  weight : Tensor { shape := #[outDim, inDim], dtype := .Float32 }
  bias : Tensor { shape := #[outDim], dtype := .Float32 }
  name : Static String
  runningMean : Frozen { shape := #[outDim], dtype := .Float32 }
  deriving TensorStruct
```

### 2) Generic transformations over nested parameter trees

```lean
open torch

def detachAll [TensorStruct α] (x : α) : α :=
  TensorStruct.map Tensor.detach x

def clearGradsM [TensorStruct α] (x : α) : IO Unit :=
  TensorStruct.zeroGrads x

def prepareForTraining [TensorStruct α] (x : α) : IO α := do
  let params := TensorStruct.makeLeafParams x
  TensorStruct.zeroGrads params
  pure params

def tensorLeafCount [TensorStruct α] (x : α) : Nat :=
  TensorStruct.count x
```

### 3) Summarize parameter trees with `fold`

```lean
open torch

-- Total number of scalar elements across all tensor leaves.
def numScalars [TensorStruct α] (x : α) : Nat :=
  TensorStruct.fold (fun {σ} _ acc => acc + TensorSpec.numelOfShape σ.shape) 0 x
```

### 4) Combine structures elementwise

```lean
open torch

def averageParams [TensorStruct α] (a b : α) : α :=
  TensorStruct.scale (TensorStruct.add a b) 0.5
```

### 5) Prefer `Vector` for type-safe `zipWith`

```lean
open torch

def blendVec {n : Nat} [TensorStruct α] (a b : Vector n α) : Vector n α :=
  TensorStruct.zipWith (fun x y => x + y) a b
```

Unlike `Array`/`List`, `Vector n α` encodes length in the type, so mismatched zip sizes
are rejected by typing instead of failing at runtime.

## Safety Notes

`TensorStruct` instances for `Array` and `List` perform runtime length checks in `zipWith`.
If lengths differ, they panic. Use `Vector n α` when you want static guarantees that a zip
cannot mismatch.

## Scope

This module provides the core traversal semantics and foundational container instances.
Higher-level model code should usually depend on these abstractions (often via `import Tyr`)
rather than writing tensor-tree plumbing by hand. This is the layer that enables model-agnostic
training loops, optimizer state transforms, and checkpoint load/save pipelines.
-/

namespace torch

/-! ## Static and Frozen Wrappers

These wrapper types control how fields are handled during tensor tree traversal:
- `Static α`: Non-tensor data that should be completely skipped (e.g., config, hyperparameters)
- `Frozen σ`: Tensor that participates in forward pass but not in gradient updates
-/

/-- Non-tensor static data that should be skipped during traversal.
    Use this for configuration, hyperparameters, or any non-tensor metadata. -/
structure Static (α : Type) where
  val : α
  deriving Repr, BEq, Hashable

instance [Inhabited α] : Inhabited (Static α) := ⟨⟨default⟩⟩

/-- Coercion from α to Static α -/
instance : Coe α (Static α) := ⟨Static.mk⟩

/-- A frozen (non-trainable) tensor parameter.
    Participates in forward pass but gradients are not tracked for optimization. -/
structure Frozen (σ : StaticSpec) where
  tensor : Tensor σ

namespace Frozen

def map {σ : StaticSpec} (f : Tensor σ → Tensor σ) (fr : Frozen σ) : Frozen σ :=
  { tensor := f fr.tensor }

def get {σ : StaticSpec} (fr : Frozen σ) : Tensor σ := fr.tensor

end Frozen

/-- Mark an indexed tensor as a frozen parameter. -/
instance {σ : StaticSpec} : Coe (Tensor σ) (Frozen σ) := ⟨Frozen.mk⟩

/-! ## Vector: Length-indexed array

A Vector carries its length in the type, enabling type-safe operations
like zipWith that require matching lengths.
-/

/-- Length-indexed array. Carries length in the type for type-safe operations. -/
structure Vector (n : Nat) (α : Type) where
  data : Array α
  size_eq : data.size = n

namespace Vector

/-- Get the underlying array -/
def toArray {n : Nat} (v : Vector n α) : Array α := v.data

/-- Get element at index (uses Fin for bounds safety) -/
def get {n : Nat} (v : Vector n α) (i : Fin n) : α :=
  v.data[i.val]'(by rw [v.size_eq]; exact i.isLt)

/-- Map a function over all elements -/
def map {n : Nat} (f : α → β) (v : Vector n α) : Vector n β :=
  ⟨v.data.map f, by rw [Array.size_map]; exact v.size_eq⟩

/-- Helper to build array by mapping over Fin indices -/
private def mapMCore {n : Nat} {m : Type → Type} [Monad m]
    (f : α → m β) (v : Vector n α) (acc : Array β) (i : Nat)
    (h_acc : acc.size = i) (h_bound : i ≤ n) : m { arr : Array β // arr.size = n } := do
  if h : i < n then
    let elem := v.data[i]'(by rw [v.size_eq]; exact h)
    let b ← f elem
    let acc' := acc.push b
    have h_acc' : acc'.size = i + 1 := by simp [acc', Array.size_push, h_acc]
    mapMCore f v acc' (i + 1) h_acc' h
  else
    have : i = n := Nat.le_antisymm h_bound (Nat.ge_of_not_lt h)
    pure ⟨acc, by rw [h_acc, this]⟩

/-- Monadic map - maps function over elements, preserving vector size.
    Uses explicit recursion to maintain size proof without relying on Array.size_mapM. -/
def mapM {n : Nat} {m : Type → Type} [Monad m] (f : α → m β) (v : Vector n α) : m (Vector n β) := do
  let ⟨arr, h⟩ ← mapMCore f v #[] 0 rfl (Nat.zero_le n)
  pure ⟨arr, h⟩

/-- Zip two vectors with a function (always safe - types guarantee same length) -/
def zipWith {n : Nat} (f : α → β → γ) (v1 : Vector n α) (v2 : Vector n β) : Vector n γ :=
  ⟨Array.zipWith f v1.data v2.data, by
    rw [Array.size_zipWith, v1.size_eq, v2.size_eq, Nat.min_self]⟩

/-- Fold over elements -/
def foldl {n : Nat} (f : β → α → β) (init : β) (v : Vector n α) : β :=
  v.data.foldl f init

/-- Create vector by replicating a value -/
def replicate (n : Nat) (a : α) : Vector n α :=
  ⟨Array.replicate n a, Array.size_replicate ..⟩

/-- Create an empty vector -/
def empty : Vector 0 α := ⟨#[], rfl⟩

/-- Push an element (increments size in type) -/
def push {n : Nat} (v : Vector n α) (a : α) : Vector (n + 1) α :=
  ⟨v.data.push a, by simp only [Array.size_push, v.size_eq]⟩

end Vector

instance {n : Nat} {α : Type} [Repr α] : Repr (Vector n α) where
  reprPrec v p := reprPrec v.data p

instance {n : Nat} {α : Type} [Inhabited α] : Inhabited (Vector n α) where
  default := Vector.replicate n default

/-! ## TensorStruct Typeclass

A typeclass for structures that contain tensors, enabling generic traversal
and transformation of all tensors in the structure. This is similar to
JAX's PyTree concept or Equinox's filtered transformations.
-/

/-- A marker class for deriving all Tyr structural instances at once.
    Use `deriving Model` to derive TensorStruct, ToTensorStructSchema, and TensorStructFlatten. -/
class Model (α : Type)

/-- A typeclass for structures that contain tensors.
    Allows generic traversal and transformation of all tensors in the structure. -/
class TensorStruct (α : Type) where
  -- Map a function over all tensors
  map : (∀ {σ}, Tensor σ → Tensor σ) → α → α
  -- Monadic map (e.g., for initialization with IO)
  mapM {m : Type → Type} [Monad m] : (∀ {σ}, Tensor σ → m (Tensor σ)) → α → m α
  -- Zip two containers with a function
  zipWith : (∀ {σ}, Tensor σ → Tensor σ → Tensor σ) → α → α → α
  -- Fold over tensors
  fold {β : Type} : (∀ {σ}, Tensor σ → β → β) → β → α → β

instance {σ : StaticSpec} : TensorStruct (Tensor σ) where
  map f tensor := f tensor
  mapM f tensor := f tensor
  zipWith f lhs rhs := f lhs rhs
  fold f init tensor := f tensor init

/-- A heterogeneous shape carrier whose leaves retain an indexed dtype. -/
abbrev ShapePackedTensor (dtype : DType) := Sigma fun shape : Shape => Tensor { shape := shape, dtype := dtype }

instance {dtype : DType} : Inhabited (ShapePackedTensor dtype) := ⟨⟨#[], default⟩⟩

instance {dtype : DType} : TensorStruct (ShapePackedTensor dtype) where
  map f x := ⟨x.1, f x.2⟩
  mapM f x := do pure ⟨x.1, ← f x.2⟩
  zipWith f x y :=
    letI : Inhabited (ShapePackedTensor dtype) := ⟨x⟩
    if h : y.1 = x.1 then ⟨x.1, f x.2 (h ▸ y.2)⟩
    else panic! s!"TensorStruct shape mismatch: {x.1} vs {y.1}"
  fold f initial x := f x.2 initial

/-!
## Container Instances

### Type Safety Note

For containers like Array and List, `zipWith` requires matching sizes at runtime
since these types don't encode length in the type. If sizes mismatch, it panics.

**For type-safe `zipWith` operations, use `Vector n α` instead.**
Vector carries its length in the type, so `zipWith` is statically guaranteed to work.

Example migration:
```lean
-- Instead of: Array (BlockParams n_embd)
-- Use:        Vector numBlocks (BlockParams n_embd)
```

This trades some flexibility for compile-time guarantees.
-/

/-- TensorStruct instance for Array.
    **Warning**: `zipWith` panics if arrays have different sizes.
    For type-safe zipWith, use `Vector n α` instead. -/
instance {α : Type} [TensorStruct α] : TensorStruct (Array α) where
  map f arr := arr.map (TensorStruct.map f)
  mapM f arr := arr.mapM (TensorStruct.mapM f)
  zipWith f arr1 arr2 :=
    if arr1.size == arr2.size then
      Array.zipWith (TensorStruct.zipWith f) arr1 arr2
    else
      panic! "TensorStruct.zipWith: Array size mismatch (use Vector for type safety)"
  fold f init arr := arr.foldl (fun acc x => TensorStruct.fold f acc x) init

/-- TensorStruct instance for List.
    **Warning**: `zipWith` panics if lists have different lengths.
    For type-safe zipWith, use `Vector n α` instead. -/
instance {α : Type} [TensorStruct α] : TensorStruct (List α) where
  map f l := l.map (TensorStruct.map f)
  mapM f l := l.mapM (TensorStruct.mapM f)
  zipWith f l1 l2 :=
    if l1.length == l2.length then
      List.zipWith (TensorStruct.zipWith f) l1 l2
    else
      panic! "TensorStruct.zipWith: List length mismatch (use Vector for type safety)"
  fold f init l := l.foldl (fun acc x => TensorStruct.fold f acc x) init

/-- TensorStruct instance for Vector - `zipWith` is always type-safe!
    Length is encoded in the type, so no runtime check is needed. -/
instance {n : Nat} {α : Type} [TensorStruct α] : TensorStruct (Vector n α) where
  map f v := Vector.map (TensorStruct.map f) v
  mapM f v := Vector.mapM (TensorStruct.mapM f) v
  zipWith f v1 v2 := Vector.zipWith (TensorStruct.zipWith f) v1 v2  -- No runtime check needed!
  fold f init v := v.data.foldl (fun acc x => TensorStruct.fold f acc x) init

instance {α : Type} [TensorStruct α] : TensorStruct (Option α) where
  map f opt := opt.map (TensorStruct.map f)
  mapM f opt := match opt with
    | some x => do let x' ← TensorStruct.mapM f x; return some x'
    | none => return none
  zipWith f opt1 opt2 := match opt1, opt2 with
    | some x, some y => some (TensorStruct.zipWith f x y)
    | _, _ => none
  fold f init opt := match opt with
    | some x => TensorStruct.fold f init x
    | none => init

/-- Static values are completely skipped during TensorStruct traversal -/
instance {α : Type} [Inhabited α] : TensorStruct (Static α) where
  map _ s := s
  mapM _ s := pure s
  zipWith _ s _ := s
  fold _ init _ := init

/-- Frozen tensors are traversed (for forward pass) but can be filtered out for gradients -/
instance {σ : StaticSpec} : TensorStruct (Frozen σ) where
  map f fr := { tensor := TensorStruct.map f fr.tensor }
  mapM f fr := do pure { tensor := ← TensorStruct.mapM f fr.tensor }
  zipWith f fr1 fr2 := { tensor := TensorStruct.zipWith f fr1.tensor fr2.tensor }
  fold f init fr := TensorStruct.fold f init fr.tensor

/-! ## TensorStruct Instances for Basic Types

Basic types contain no tensors, so they are passed through unchanged.
This allows structures with mixed tensor and non-tensor fields to derive TensorStruct.
-/

instance : TensorStruct Bool where
  map _ b := b
  mapM _ b := pure b
  zipWith _ b _ := b
  fold _ init _ := init

instance : TensorStruct Float where
  map _ x := x
  mapM _ x := pure x
  zipWith _ x _ := x
  fold _ init _ := init

instance : TensorStruct UInt8 where
  map _ x := x
  mapM _ x := pure x
  zipWith _ x _ := x
  fold _ init _ := init

instance : TensorStruct UInt64 where
  map _ x := x
  mapM _ x := pure x
  zipWith _ x _ := x
  fold _ init _ := init

instance : TensorStruct Nat where
  map _ x := x
  mapM _ x := pure x
  zipWith _ x _ := x
  fold _ init _ := init

instance : TensorStruct Int where
  map _ x := x
  mapM _ x := pure x
  zipWith _ x _ := x
  fold _ init _ := init

instance : TensorStruct String where
  map _ s := s
  mapM _ s := pure s
  zipWith _ s _ := s
  fold _ init _ := init

/-! ## TensorStruct Utility Methods

Convenient operations for common tensor tree manipulations.
-/

namespace TensorStruct

/-- Count the number of tensor leaves in the structure -/
def count [TensorStruct α] (model : α) : Nat :=
  fold (fun _ n => n + 1) 0 model

/-- Read independent gradient snapshots for all tensors in the structure. -/
def grads [TensorStruct α] (model : α) : IO α :=
  mapM Tensor.gradOf model

/-- Clear all accumulated gradients in the structure in place. -/
def zeroGrads [TensorStruct α] (model : α) : IO Unit := do
  let _ ← mapM (fun t => do Tensor.zeroGrad t; pure t) model
  pure ()

/-- Detach all tensors from the computation graph -/
def detach [TensorStruct α] (model : α) : α :=
  map Tensor.detach model

/-- Set requires_grad on all tensors -/
def requiresGrad [TensorStruct α] (model : α) (b : Bool) : α :=
  map (fun t => Tensor.setRequiresGrad t b) model

/-- Make a tensor a trainable leaf parameter (detach and set requires_grad) -/
private def _makeLeafParam {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  Tensor.setRequiresGrad (Tensor.detach t) true

/-- Make all tensors trainable leaf parameters -/
def makeLeafParams [TensorStruct α] (model : α) : α :=
  map _makeLeafParam model

/-- Explicitly cast tensor-tree leaves while retaining their original static
indices. A dynamic dtype permits a native dtype change; a concrete dtype rejects
a different result. Shape, actual device and pinned placement are preserved.
Failure is reported through IO instead of substituting a fallback tensor. -/
def castDType [TensorStruct α] (dtype : DType) (params : α) : IO α :=
  mapM (fun {σ} tensor => do
    let result : Except String (Tensor σ) := do
      let converted ← tensor.castDType? dtype
      Tensor.ofRaw converted.raw
    match result with
    | .ok converted => pure converted
    | .error message =>
        throw <| IO.userError s!"TensorStruct.castDType to {dtype}: {message}") params

/-- Apply a scalar multiplication to all tensors -/
def scale [TensorStruct α] (model : α) (s : Float) : α :=
  map (fun t => Tensor.Preserving.mulScalar t s) model

/-- Element-wise addition of two structures -/
def add [TensorStruct α] (a b : α) : α :=
  zipWith Add.add a b

/-- Element-wise subtraction of two structures -/
def sub [TensorStruct α] (a b : α) : α :=
  zipWith Sub.sub a b

/-- Check if all tensors in the structure satisfy a predicate -/
def all [TensorStruct α] (p : ∀ {σ}, Tensor σ → Bool) (model : α) : Bool :=
  fold (fun t acc => acc && p t) true model

/-- Check if any tensor in the structure satisfies a predicate -/
def any [TensorStruct α] (p : ∀ {σ}, Tensor σ → Bool) (model : α) : Bool :=
  fold (fun t acc => acc || p t) false model

end TensorStruct

end torch
