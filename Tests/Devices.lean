import Tyr.Torch

namespace Tests

/-- Acquired accelerators for runtime admission tests. Pure metadata tests use
DeviceKind and do not depend on accelerator availability. -/
def acceleratorDevices : IO (Array torch.Device) := do
  let mut result := #[]
  if ← torch.cuda_is_available then
    result := result.push (← torch.Device.cuda 0)
  if ← torch.mps_is_available then
    result := result.push (← torch.Device.mps)
  pure result

end Tests
