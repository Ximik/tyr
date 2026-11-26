import Tyr.Core.FFI.Basic

/-! Native runtime services: device acquisition, CUDA streams and events,
file helpers, and the global RNG seed. Re-exported by `Tyr.Core.Torch`. -/

namespace torch

namespace Device

/-- Acquire an available CUDA device, checking its ordinal before producing a
placement that pure tensor operations can accept. -/
@[extern "lean_torch_device_cuda"]
opaque cuda (index : Nat := 0) : IO Device :=
  throw (IO.userError "Device.cuda requires the native runtime")

/-- Acquire the MPS device after checking backend availability. -/
@[extern "lean_torch_device_mps"]
opaque mps : IO Device :=
  throw (IO.userError "Device.mps requires the native runtime")

end Device

/-- Check if CUDA is available -/
@[extern "lean_torch_cuda_is_available"]
opaque cuda_is_available : IO Bool

/-- Get the current CUDA stream as an opaque `UInt64` handle (`cudaStream_t`). -/
@[extern "lean_torch_cuda_current_stream"]
opaque cuda_current_stream : IO UInt64

/-- Synchronize CUDA device; useful for deterministic validation around custom kernels. -/
@[extern "lean_torch_cuda_synchronize"]
opaque cuda_synchronize : IO Unit

/-- Opaque CUDA event handle for device-side benchmark timing. -/
abbrev CudaEvent := UInt64

@[extern "lean_torch_cuda_event_create"]
opaque cuda_event_create : IO CudaEvent

@[extern "lean_torch_cuda_event_record"]
opaque cuda_event_record (event : CudaEvent) (stream : UInt64 := 0) : IO Unit

@[extern "lean_torch_cuda_event_synchronize"]
opaque cuda_event_synchronize (event : CudaEvent) : IO Unit

@[extern "lean_torch_cuda_event_elapsed_ms"]
opaque cuda_event_elapsed_ms (start stop : CudaEvent) : IO Float

@[extern "lean_torch_cuda_event_destroy"]
opaque cuda_event_destroy (event : CudaEvent) : IO Unit

/-- Check if MPS (Metal Performance Shaders) is available -/
@[extern "lean_torch_mps_is_available"]
opaque mps_is_available : IO Bool

namespace data

@[extern "lean_torch_bin_file_token_count"]
opaque binFileTokenCount (path : @& String) : IO Nat

/-- High-quality mono resampling. Zero rates, unrepresentable output sizes, and
native failures return IO errors. Equal-rate and empty inputs retain the input array. -/
@[extern "lean_torch_resample_soxr_hq"]
opaque resampleSoxrHQ (samples : @& Array Float) (origSr targetSr : @& Nat) : IO (Array Float)

end data

@[extern "lean_torch_get_live_tensors"]
opaque get_live_tensors : IO UInt64

/-- Seed the default CPU generator. CUDA generators are currently unchanged. -/
@[extern "lean_torch_manual_seed"]
opaque manualSeed (seed : UInt64) : IO Unit

end torch
