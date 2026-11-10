import Tyr.TensorStruct
import Tyr.TensorStruct.Cast

/-!
# Index-determined containers

`Select α β b` holds an `α` when `b` is `false` and a `β` when `b` is `true`.

`IndexedList α start count` holds `count` values; the value at position `i` has
type `α i`.

Two values of the same type have the same structure, so `zipWith` pairs their
contents element for element.
-/

namespace torch

inductive Select (onFalse onTrue : Type) : Bool → Type where
  | ofFalse : onFalse → Select onFalse onTrue false
  | ofTrue : onTrue → Select onFalse onTrue true

namespace Select

variable {α β α' β' : Type} {b : Bool}

/-- Run the action that `b` selects. -/
def build {m : Type → Type} [Monad m] : (b : Bool) → m α → m β → m (Select α β b)
  | false, onFalse, _ => .ofFalse <$> onFalse
  | true, _, onTrue => .ofTrue <$> onTrue

instance [TensorStruct α] [TensorStruct β] : TensorStruct (Select α β b) where
  map f
    | .ofFalse x => .ofFalse (TensorStruct.map f x)
    | .ofTrue x => .ofTrue (TensorStruct.map f x)
  mapM f
    | .ofFalse x => return .ofFalse (← TensorStruct.mapM f x)
    | .ofTrue x => return .ofTrue (← TensorStruct.mapM f x)
  zipWith f
    | .ofFalse x, .ofFalse y => .ofFalse (TensorStruct.zipWith f x y)
    | .ofTrue x, .ofTrue y => .ofTrue (TensorStruct.zipWith f x y)
  fold f init
    | .ofFalse x => TensorStruct.fold f init x
    | .ofTrue x => TensorStruct.fold f init x

instance {P : StaticSpec → Prop} [TensorStructFor P α] [TensorStructFor P β] :
    TensorStructFor P (Select α β b) where
  map f
    | .ofFalse x => .ofFalse (TensorStructFor.map f x)
    | .ofTrue x => .ofTrue (TensorStructFor.map f x)
  mapM f
    | .ofFalse x => return .ofFalse (← TensorStructFor.mapM f x)
    | .ofTrue x => return .ofTrue (← TensorStructFor.mapM f x)
  zipWith f
    | .ofFalse x, .ofFalse y => .ofFalse (TensorStructFor.zipWith f x y)
    | .ofTrue x, .ofTrue y => .ofTrue (TensorStructFor.zipWith f x y)
  fold f init
    | .ofFalse x => TensorStructFor.fold f init x
    | .ofTrue x => TensorStructFor.fold f init x

instance [TensorTreeCast α α'] [TensorTreeCast β β'] :
    TensorTreeCast (Select α β b) (Select α' β' b) where
  castAt path
    | .ofFalse x => return .ofFalse (← TensorTreeCast.castAt path x)
    | .ofTrue x => return .ofTrue (← TensorTreeCast.castAt path x)

end Select

inductive IndexedList (α : Nat → Type) : (start count : Nat) → Type where
  | nil {start : Nat} : IndexedList α start 0
  | cons {start count : Nat} : α start → IndexedList α (start + 1) count → IndexedList α start (count + 1)

namespace IndexedList

variable {α β : Nat → Type}

/-- Run `f start, f (start + 1), …` in order and collect the results. -/
def ofFnM {m : Type → Type} [Monad m] (f : (i : Nat) → m (α i)) :
    (start count : Nat) → m (IndexedList α start count)
  | _, 0 => pure .nil
  | start, count + 1 => do
    let head ← f start
    pure (.cons head (← ofFnM f (start + 1) count))

def mapM {m : Type → Type} [Monad m] (f : (i : Nat) → α i → m (β i)) {start count : Nat} :
    IndexedList α start count → m (IndexedList β start count)
  | .nil => pure .nil
  | .cons head tail => do
    let head ← f _ head
    pure (.cons head (← mapM f tail))

def map (f : (i : Nat) → α i → β i) {start count : Nat} (xs : IndexedList α start count) :
    IndexedList β start count :=
  Id.run (xs.mapM fun i x => pure (f i x))

def zipWith (f : (i : Nat) → α i → α i → α i) {start count : Nat} :
    IndexedList α start count → IndexedList α start count → IndexedList α start count
  | .nil, .nil => .nil
  | .cons x xs, .cons y ys => .cons (f _ x y) (zipWith f xs ys)

/-- Fold in order; `f` receives each element's position. -/
def foldlM {m : Type → Type} [Monad m] {γ : Type} (f : γ → (i : Nat) → α i → m γ) (init : γ)
    {start count : Nat} : IndexedList α start count → m γ
  | .nil => pure init
  | .cons head tail => do foldlM f (← f init _ head) tail

def foldl {γ : Type} (f : γ → (i : Nat) → α i → γ) (init : γ) {start count : Nat}
    (xs : IndexedList α start count) : γ :=
  Id.run (xs.foldlM (fun acc i x => pure (f acc i x)) init)

/-- The element at offset `k`. -/
def get {start count : Nat} : IndexedList α start count → (k : Nat) → k < count → α (start + k)
  | .cons head _, 0, _ => head
  | .cons _ tail, k + 1, h =>
    (by omega : start + 1 + k = start + (k + 1)) ▸ tail.get k (by omega)

instance {start count : Nat} [∀ i, TensorStruct (α i)] : TensorStruct (IndexedList α start count) where
  map f xs := xs.map fun _ x => TensorStruct.map f x
  mapM f xs := xs.mapM fun _ x => TensorStruct.mapM f x
  zipWith f xs ys := xs.zipWith (fun _ x y => TensorStruct.zipWith f x y) ys
  fold f init xs := xs.foldl (fun acc _ x => TensorStruct.fold f acc x) init

instance {P : StaticSpec → Prop} {start count : Nat} [∀ i, TensorStructFor P (α i)] :
    TensorStructFor P (IndexedList α start count) where
  map f xs := xs.map fun _ x => TensorStructFor.map f x
  mapM f xs := xs.mapM fun _ x => TensorStructFor.mapM f x
  zipWith f xs ys := xs.zipWith (fun _ x y => TensorStructFor.zipWith f x y) ys
  fold f init xs := xs.foldl (fun acc _ x => TensorStructFor.fold f acc x) init

instance {start count : Nat} [∀ i, TensorTreeCast (α i) (β i)] :
    TensorTreeCast (IndexedList α start count) (IndexedList β start count) where
  castAt path xs := xs.mapM fun i x => TensorTreeCast.castAt (TensorTreeCast.indexPath path i) x

end IndexedList

end torch
