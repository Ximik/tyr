import Tyr.Basic.Types

/-! Evidence that the constant factory paths can represent both zero and one
at an exact dtype and acquired placement. This is deliberately separate from
dtype recognition, promotion, and support for arbitrary arithmetic kernels. -/

namespace torch.DType

/-- Conservative built-in support for the pinned runtime. Accelerator-dependent
formats, including BFloat16 on MPS, require runtime acquisition. -/
@[reducible] def staticConstantSupport (dtype : DType) (device : Device) : _root_.Bool :=
  match dtype with
  | .UInt8 | .Int8 | .Int16 | .Int32 | .Int64 | .Float16 | .Float32 | .Bool => true
  | .BFloat16 | .Float64 | .Complex32 | .Complex64 | .Complex128
  | .Float8E4M3FN | .Float8E5M2 | .Float8E5M2FNUZ | .Float8E4M3FNUZ
  | .UInt16 | .UInt32 | .UInt64 => device.kind == .CPU
  | _ => false

-- Runtime support is an acquired fact. Keeping its body opaque prevents pure
-- Lean code from manufacturing support for an unsupported dtype/device pair.
private opaque RuntimeConstantSupport (_dtype : DType) (_device : Device) : Prop := True

/-- Erased evidence for zero, one, and identity construction at this exact
dtype and device. It certifies the actual Float32 conversion paths used by
the factories; it does not certify arbitrary operations on this dtype. -/
class ConstantSupported (dtype : DType) (device : Device) : Prop where
  valid : staticConstantSupport dtype device = true ∨ RuntimeConstantSupport dtype device

private class StaticSupport (value : _root_.Bool) : Prop where
  valid : value = true

private instance : StaticSupport true := ⟨rfl⟩

instance {dtype : DType} {device : Device}
    [support : StaticSupport (staticConstantSupport dtype device)] :
    ConstantSupported dtype device := ⟨.inl support.valid⟩

-- `Device.CPU` deliberately remains an ordinary definition. Explicit instances
-- avoid asking typeclass reduction to unfold its representation or accessor.
instance : ConstantSupported .BFloat16 .CPU := ⟨.inl rfl⟩
instance : ConstantSupported .Float64 .CPU := ⟨.inl rfl⟩
instance : ConstantSupported .Complex32 .CPU := ⟨.inl rfl⟩
instance : ConstantSupported .Complex64 .CPU := ⟨.inl rfl⟩
instance : ConstantSupported .Complex128 .CPU := ⟨.inl rfl⟩
instance : ConstantSupported .Float8E4M3FN .CPU := ⟨.inl rfl⟩
instance : ConstantSupported .Float8E5M2 .CPU := ⟨.inl rfl⟩
instance : ConstantSupported .Float8E5M2FNUZ .CPU := ⟨.inl rfl⟩
instance : ConstantSupported .Float8E4M3FNUZ .CPU := ⟨.inl rfl⟩
instance : ConstantSupported .UInt16 .CPU := ⟨.inl rfl⟩
instance : ConstantSupported .UInt32 .CPU := ⟨.inl rfl⟩
instance : ConstantSupported .UInt64 .CPU := ⟨.inl rfl⟩

end torch.DType
