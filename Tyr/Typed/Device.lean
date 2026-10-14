import Tyr.Typed.Spec

/-!
# Runtime placement contracts

`StaticSpec.device` carries either an exact placement or an unrestricted policy.
`TensorContract` adds runtime placement checks to shape/dtype metadata for
boundaries such as file loading and external native handles.
-/

namespace torch

namespace Device

def render : Device → String
  | .CPU => "CPU"
  | .MPS => "MPS"
  | .CUDA idx => s!"CUDA:{idx}"

end Device

namespace DevicePolicy

def check (policy : DevicePolicy) (actual : Device) : Except String Unit :=
  match policy with
  | .any => .ok ()
  | .exact expected =>
      if actual == expected then
        .ok ()
      else
        .error s!"Expected device {expected.render}, got {actual.render}"

end DevicePolicy

inductive TensorRole where
  | activation
  | parameter
  | index
  | mask
  | logits
  | loss
  | cache
  | quantScale
deriving Repr, BEq, DecidableEq

structure TensorContract where
  spec : TensorSpec
  role : TensorRole := .activation
  devicePolicy : DevicePolicy := .any
deriving Repr, BEq

namespace TensorContract

def check (contract : TensorContract) (actualSpec : TensorSpec) (actualDevice : Device) :
    Except String Unit := do
  TensorSpec.checkCompatible contract.spec actualSpec
  contract.devicePolicy.check actualDevice

end TensorContract

end torch
