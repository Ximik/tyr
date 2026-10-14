import Tyr.RawTensor

/-! Native SafeTensors boundaries. Public loaders validate stored metadata before
returning indexed tensors. An empty native shape requests the stored layout. -/
namespace torch.safetensors
opaque SafeTensorsHandle : Type
@[extern "lean_torch_safetensors_open"]
opaque openHandle (path : @& String) : IO SafeTensorsHandle
end torch.safetensors

namespace torch.RawTensor
@[extern "lean_torch_safetensors_load_from_handle"]
opaque safeTensorsLoadFromHandle (handle : @& safetensors.SafeTensorsHandle)
    (name : @& String) (shape : Shape) : IO RawTensor
@[extern "lean_torch_safetensors_load"]
opaque safeTensorsLoad (path name : @& String) (shape : Shape) : IO RawTensor
@[extern "lean_torch_safetensors_load_sharded"]
opaque safeTensorsLoadSharded (dir name : @& String) (shape : Shape) : IO RawTensor

/-- Preserve stored dimensions until the public contract has been checked. -/
def safeTensorsLoadOnDevice (path name : String) (device : Device) : IO RawTensor := do
  let raw ← safeTensorsLoad path name #[]
  pure (toDevice raw.runtimeShape raw device)

def safeTensorsLoadShardedOnDevice (dir name : String) (device : Device) : IO RawTensor := do
  let raw ← safeTensorsLoadSharded dir name #[]
  pure (toDevice raw.runtimeShape raw device)

def safeTensorsLoadFromHandleOnDevice (handle : safetensors.SafeTensorsHandle)
    (name : String) (device : Device) : IO RawTensor := do
  let raw ← safeTensorsLoadFromHandle handle name #[]
  pure (toDevice raw.runtimeShape raw device)
end torch.RawTensor
