import Tyr.Typed.Spec

/-!
# Runtime placement

`StaticSpec.device` is an actual placement, possibly selected at runtime.
-/

namespace torch

namespace DeviceKind

def render : DeviceKind → String
  | .CPU => "CPU"
  | .MPS => "MPS"
  | .CUDA idx => s!"CUDA:{idx}"

/-- Validate actual placement for operations that require one device.
The check is independent of the typed indices and also guards raw boundaries. -/
def sameResult? (left right : DeviceKind) : Except String DeviceKind :=
  if left == right then
    .ok left
  else
    .error s!"Expected the same device, got {left.render} and {right.render}"

end DeviceKind

namespace Device

def render (device : Device) : String := device.kind.render

/-- Validate acquired placement identity at a native boundary. -/
def sameResult? (left right : Device) : Except String Device := do
  let _ ← DeviceKind.sameResult? left.kind right.kind
  pure left

/-- Equality evidence for a same-device operation. Runtime-selected devices can
be related by checking equality before entering a shared-device operation. -/
class SameDeviceCompatible (left right : Device) : Prop where
  valid : left = right

instance sameDevice (device : Device) : SameDeviceCompatible device device := ⟨rfl⟩

/-- Check placement at a native admission or conversion boundary. -/
def check (expected actual : Device) : Except String Unit :=
  if actual == expected then .ok ()
  else .error s!"Expected device {expected.render}, got {actual.render}"

end Device

end torch
