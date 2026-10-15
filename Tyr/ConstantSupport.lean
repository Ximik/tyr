import Tyr.ConstantSupport.Types

namespace torch.DType

@[extern "lean_torch_acquire_constant_support"]
private opaque acquireConstantSupportImpl (dtype : @& DType) (device : @& Device) :
    IO (PLift (ConstantSupported dtype device)) :=
  throw (IO.userError "DType.acquireConstantSupport requires the native runtime")

/-- Validate zero, one, and identity conversion on this acquired placement.
Keep the returned witness while using pure constant factories:
```
let support ← dtype.acquireConstantSupport device
let _ := support.down
```
Known supported combinations use the static support table. Other combinations
allocate small temporary tensors and verify exact zero/one values.
Unknown, quantized-without-parameters, and unsupported shell formats fail here.
The witness says nothing about other numerical kernels or later resource
exhaustion. Its proof erases, so factories carry no runtime support flag. -/
def acquireConstantSupport (dtype : DType) (device : Device) :
    IO (PLift (ConstantSupported dtype device)) :=
  if supported : staticConstantSupport dtype device = true then
    pure ⟨⟨.inl supported⟩⟩
  else
    acquireConstantSupportImpl dtype device

end torch.DType
