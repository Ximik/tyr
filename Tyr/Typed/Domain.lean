import Tyr.Typed.Promotion

/-! Operator domains are independent of dtype promotion. These witnesses rule
out invalid categories; device-specific kernel availability is still checked
by the native `Except` entry points. Dynamic dtypes use the checked APIs. -/
namespace torch.DType

@[reducible] def reluDomain (dtype : DType) : _root_.Bool :=
  match dtype with
  | .UInt8 | .UInt16 | .UInt32 | .UInt64 | .Int8 | .Int16 | .Int32 | .Int64 => true
  | .QInt8 | .QUInt8 | .QInt32 | .QUInt4x2 | .QUInt2x4 => true
  | _ => dtype.isFloating

@[reducible] def orderedDomain (dtype : DType) : _root_.Bool :=
  match dtype with
  | .Bool => true
  | _ => dtype.reluDomain

@[reducible] def meanDomain (dtype : DType) : _root_.Bool :=
  match dtype with
  | .Complex32 | .Complex64 | .Complex128 => true
  | _ => dtype.isFloating

class Relu (dtype : DType) : Prop where
  valid : reluDomain dtype = true

class Ordered (dtype : DType) : Prop where
  valid : orderedDomain dtype = true

class Mean (dtype : DType) : Prop where
  valid : meanDomain dtype = true

class InDomain (accepted : _root_.Bool) : Prop where
  valid : accepted = true

instance : InDomain true := ⟨rfl⟩

instance {dtype : DType} [h : InDomain (reluDomain dtype)] : Relu dtype := ⟨h.valid⟩
instance {dtype : DType} [h : InDomain (orderedDomain dtype)] : Ordered dtype := ⟨h.valid⟩
instance {dtype : DType} [h : InDomain (meanDomain dtype)] : Mean dtype := ⟨h.valid⟩

instance {dtype : DType} [h : RealFloating dtype] : Relu dtype := by
  constructor
  cases dtype <;> first | rfl | cases h.valid

instance {dtype : DType} [h : RealFloating dtype] : Ordered dtype := by
  constructor
  cases dtype <;> first | rfl | cases h.valid

instance {dtype : DType} [h : RealFloating dtype] : Mean dtype := by
  constructor
  cases dtype <;> first | rfl | cases h.valid

def checkRelu (dtype : DType) : Except String Unit :=
  if dtype.reluDomain then .ok ()
  else .error s!"relu requires a real numeric dtype, got {dtype}"

def checkOrdered (dtype : DType) : Except String Unit :=
  if dtype.orderedDomain then .ok ()
  else .error s!"ordered reduction requires a real or Bool dtype, got {dtype}"

def checkMean (dtype : DType) : Except String Unit :=
  if dtype.meanDomain then .ok ()
  else .error s!"mean requires a floating or complex dtype, got {dtype}"

end torch.DType
