import Tyr.TensorStruct

/-!
# Type-directed tensor-tree conversion

`TensorStruct.castTo Target source` uses the destination Lean type as a tree of
tensor specifications. Each leaf can request a different dtype and placement;
no destination tensor template is allocated. Shapes and container structure are
preserved. Tensor leaves are converted natively and checked against their complete
destination index; static fields retain their values.
-/

namespace torch

/-- Convert a tensor tree to a possibly different type. Instances preserve its
structural paths and roles, and tensor leaves validate the destination spec.
There is deliberately no catch-all identity instance: tensor-containing records
must traverse their fields even when their source and destination types agree. -/
class TensorTreeCast (α β : Type) where
  castAt : String → α → IO β

namespace TensorTreeCast

/-- Extend a structural path with a named record/product field. -/
def fieldPath (path field : String) : String :=
  if path.isEmpty then field else path ++ "." ++ field

/-- Extend a structural path with a container element index. -/
def indexPath (path : String) (index : Nat) : String :=
  path ++ s!"[{index}]"

private def renderPath (path : String) : String :=
  if path.isEmpty then "<root>" else path

private def checked {α : Type} (result : Except String α) : IO α :=
  match result with
  | .ok value => pure value
  | .error message => throw (IO.userError message)

private def castTensorAt {σ τ : StaticSpec} (path : String)
    (tensor : Tensor σ) : IO (Tensor τ) := do
  try
    checked tensor.validate
    -- A type-directed cast never reshapes data, even for equal element counts.
    checked (TensorSpec.checkShape τ.shape σ.shape)
    let sourceDType := tensor.raw.dtype
    let targetDType := τ.dtype
    -- Unknown native dtype tokens can be retained exactly, but are never used
    -- to request an unsupported conversion or treated as a wildcard.
    let dtypeTag ← if targetDType == sourceDType then pure none else
      match targetDType with
      | .Unknown _ => throw (IO.userError s!"Unsupported target dtype {targetDType}")
      | _ => pure (some targetDType.canonicalName)
    let targetDevice := τ.device
    -- Request dtype and placement together: an intermediate dtype may be
    -- unsupported on the source device (e.g. MPS Float32 -> CPU Float64).
    let raw ← if dtypeTag.isNone && targetDevice == tensor.raw.device then
        pure tensor.raw
      else RawTensor.convertIO σ.shape tensor.raw targetDevice dtypeTag
    -- Check retained metadata as well as requested conversions.
    checked (TensorSpec.checkDType targetDType raw.dtype)
    checked (Device.check targetDevice raw.device)
    checked (Tensor.ofRaw raw)
  catch error =>
    throw <| IO.userError s!"TensorStruct.castTo at {renderPath path}: {error}"

instance {σ τ : StaticSpec} : TensorTreeCast (Tensor σ) (Tensor τ) where
  castAt := castTensorAt

/-- Preserve the hidden dtype while explicitly converting shape-compatible
placement. An existential destination does not request a dtype conversion. -/
instance {shape targetShape : Shape} {device targetDevice : Device} :
    TensorTreeCast (SomeDTypeTensor shape device) (SomeDTypeTensor targetShape targetDevice) where
  castAt path source := do
    let tensor : Tensor ⟨targetShape, source.1.val, targetDevice⟩ ←
      castTensorAt path source.2
    pure ⟨source.1, tensor⟩

instance {shape : Shape} {device : Device} {τ : StaticSpec} :
    TensorTreeCast (SomeDTypeTensor shape device) (Tensor τ) where
  castAt path source := castTensorAt path source.2

instance {σ : StaticSpec} {shape : Shape} {device : Device} :
    TensorTreeCast (Tensor σ) (SomeDTypeTensor shape device) where
  castAt path source := do
    let packed ← checked source.reifyDType
    let tensor : Tensor ⟨shape, packed.1.val, device⟩ ← castTensorAt path packed.2
    pure ⟨packed.1, tensor⟩

instance {σ τ : StaticSpec} : TensorTreeCast (Frozen σ) (Frozen τ) where
  castAt path source := do
    pure { tensor := ← castAt path source.tensor }

instance {α : Type} : TensorTreeCast (Static α) (Static α) where
  castAt _ source := pure source

instance {sourceDType targetDType : DType} {sourceDevice targetDevice : Device} :
    TensorTreeCast (ShapePackedTensor sourceDType sourceDevice)
      (ShapePackedTensor targetDType targetDevice) where
  castAt path source := do
    let tensor : Tensor { shape := source.1, dtype := targetDType, device := targetDevice } ←
      castAt path source.2
    pure ⟨source.1, tensor⟩

instance [TensorTreeCast α α'] [TensorTreeCast β β'] :
    TensorTreeCast (α × β) (α' × β') where
  castAt path source := do
    let first ← castAt (fieldPath path "fst") source.1
    let second ← castAt (fieldPath path "snd") source.2
    pure (first, second)

instance [TensorTreeCast α β] : TensorTreeCast (Array α) (Array β) where
  castAt path source := do
    let mut result := #[]
    for h : index in [:source.size] do
      let converted ← castAt (indexPath path index) source[index]
      result := result.push converted
    pure result

private def castListAt [TensorTreeCast α β] (path : String)
    (source : List α) (index : Nat := 0) : IO (List β) := do
  match source with
  | [] => pure []
  | value :: rest =>
      let converted ← castAt (indexPath path index) value
      pure (converted :: (← castListAt path rest (index + 1)))

instance [TensorTreeCast α β] : TensorTreeCast (List α) (List β) where
  castAt := castListAt

instance [TensorTreeCast α β] : TensorTreeCast (Option α) (Option β) where
  castAt path source := do
    match source with
    | none => pure none
    | some value => pure (some (← castAt path value))

instance {n : Nat} [TensorTreeCast α β] :
    TensorTreeCast (Vector n α) (Vector n β) where
  castAt path source := do
    let (result, _) ← (Vector.mapM (m := StateT Nat IO) (fun value => do
      let index ← get
      set (index + 1)
      liftM (castAt (indexPath path index) value)) source).run 0
    pure result

-- Explicit scalar identities keep metadata while avoiding an instance that
-- could silently skip tensor fields in an arbitrary same-typed record.
instance : TensorTreeCast Unit Unit where castAt _ source := pure source
instance : TensorTreeCast Bool Bool where castAt _ source := pure source
instance : TensorTreeCast Nat Nat where castAt _ source := pure source
instance : TensorTreeCast Int Int where castAt _ source := pure source
instance : TensorTreeCast Float Float where castAt _ source := pure source
instance : TensorTreeCast String String where castAt _ source := pure source
instance : TensorTreeCast Char Char where castAt _ source := pure source
instance : TensorTreeCast UInt8 UInt8 where castAt _ source := pure source
instance : TensorTreeCast UInt16 UInt16 where castAt _ source := pure source
instance : TensorTreeCast UInt32 UInt32 where castAt _ source := pure source
instance : TensorTreeCast UInt64 UInt64 where castAt _ source := pure source
instance : TensorTreeCast USize USize where castAt _ source := pure source
instance : TensorTreeCast Int8 Int8 where castAt _ source := pure source
instance : TensorTreeCast Int16 Int16 where castAt _ source := pure source
instance : TensorTreeCast Int32 Int32 where castAt _ source := pure source
instance : TensorTreeCast Int64 Int64 where castAt _ source := pure source
instance : TensorTreeCast ISize ISize where castAt _ source := pure source
instance {n : Nat} : TensorTreeCast (Fin n) (Fin n) where castAt _ source := pure source
instance : TensorTreeCast DType DType where castAt _ source := pure source
instance : TensorTreeCast Device Device where castAt _ source := pure source
instance : TensorTreeCast StaticSpec StaticSpec where castAt _ source := pure source

end TensorTreeCast

namespace TensorStruct

/-- Convert to the destination tree of tensor types, with no allocated target
template. Each target has an exact dtype and device; differing target
dtypes/devices request real native conversions.
Shape mismatches and unsupported dtype targets report their structural path. -/
def castTo (β : Type) {α : Type} [TensorTreeCast α β] (source : α) : IO β :=
  TensorTreeCast.castAt "" source

end TensorStruct

end torch
