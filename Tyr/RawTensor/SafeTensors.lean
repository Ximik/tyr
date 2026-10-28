import Tyr.RawTensor

/-! Native SafeTensors byte transfer. Headers are parsed and written in Lean
(`Tyr.SafeTensors.File`); native code only moves tensor data. -/

namespace torch.RawTensor

/-- Read `length` bytes at byte `offset` of `path` into a new CPU tensor of
`dtype` and `shape`. Fails unless `length` is that tensor's byte size. -/
@[extern "lean_torch_safetensors_read"]
opaque safeTensorsRead (path : @& String) (offset length : @& Nat) (dtype : DType)
    (shape : @& Shape) : IO RawTensor

/-- Write a SafeTensors file: the header's byte size, the header, then each
tensor's contiguous CPU bytes in order. Fails unless each tensor's byte size is
its entry in `lengths`. -/
@[extern "lean_torch_safetensors_write"]
opaque safeTensorsWrite (path header : @& String) (tensors : @& Array RawTensor)
    (lengths : @& Array UInt64) : IO Unit

end torch.RawTensor
