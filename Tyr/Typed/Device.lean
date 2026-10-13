import Tyr.Typed.Spec

/-!
# Runtime placement contracts

LeanMLX keeps device as an explicit runtime target instead of making it part of
the tensor type.  Tyr's typed facade follows the same direction: dtype belongs
in `TensorSpec`, while device is checked at boundaries that care about placement.
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
