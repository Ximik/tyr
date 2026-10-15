import Tyr.Differentiable

/-! Runtime services and shape utilities for Tyr. Tensor operations are exposed
through `Tyr.Typed` and `Tyr.Typed.Neural`, using `torch.Tensor` indices. -/

namespace torch

namespace Device

/-- Acquire an available CUDA device, checking its ordinal before producing a
placement that pure tensor operations can accept. -/
@[extern "lean_torch_device_cuda"]
opaque cuda (index : UInt64 := 0) : IO Device :=
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


/-- Begin CUDA graph capture into the process-wide graph slot. Subsequent
    CUDA work (including autograd backward) is recorded until
    `cudaGraphCaptureEnd`. Capture requires static shapes and stable buffer
    addresses; copy fresh inputs into the captured buffers before each
    `cudaGraphReplay`. Throws an IO error if capture is not possible
    (e.g. an op synchronizes or allocates illegally inside the region). -/
@[extern "lean_torch_cuda_graph_capture_begin"]
opaque cudaGraphCaptureBegin : IO Unit

/-- End the active CUDA graph capture. -/
@[extern "lean_torch_cuda_graph_capture_end"]
opaque cudaGraphCaptureEnd : IO Unit

/-- Replay the captured CUDA graph. -/
@[extern "lean_torch_cuda_graph_replay"]
opaque cudaGraphReplay : IO Unit

/-- Free the captured CUDA graph. -/
@[extern "lean_torch_cuda_graph_reset"]
opaque cudaGraphReset : IO Unit

/-- Check if MPS (Metal Performance Shaders) is available -/
@[extern "lean_torch_mps_is_available"]
opaque mps_is_available : IO Bool

/--
Probe MPS with a short retry budget.

On some macOS/libtorch combinations the first process-local probe can race MPS
backend startup and transiently return `false`. Retrying here keeps device
selection stable for downstream callers.
-/
def mps_is_available_stable (retries : Nat := 4) (delayMs : Nat := 250) : IO Bool := do
  let rec loop (remaining : Nat) : IO Bool := do
    let available ← mps_is_available
    if available then
      return true
    match remaining with
    | 0 => return false
    | remaining + 1 =>
        IO.sleep delayMs.toUInt32
        loop remaining
  loop retries

/-- Get the best available device: MPS > CUDA > CPU -/
def getBestDevice : IO Device := do
  if ← mps_is_available_stable then return ← Device.mps
  if ← cuda_is_available then return ← Device.cuda 0
  return Device.CPU

/-- Compute output shape for slicing along a dimension -/
def slicedShape (s : Shape) (dim : Nat) (start : UInt64) (stop : UInt64) (step : UInt64 := 1) : Shape :=
  if dim < s.size then
    let s' := s[:dim].toArray
    let s'' := s[dim+1:].toArray
    let d := (stop - start) / step
    s' ++ #[d] ++ s''
  else s

def convOutputSize (input_size kernel_size : UInt64) (stride : UInt64 := 1) (padding : UInt64 := 0) (dilation : UInt64 := 1) : UInt64 :=
  (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1

def convTransposeOutputSize (input_size kernel_size : UInt64)
    (stride : UInt64 := 1) (padding : UInt64 := 0) (output_padding : UInt64 := 0)
    (dilation : UInt64 := 1) : UInt64 :=
  (input_size - 1) * stride - (2 * padding) + dilation * (kernel_size - 1) + output_padding + 1

/-- Compute output shape for 1D convolution (safe version) -/
def conv1dShape (input_shape weight_shape : Shape) (stride padding dilation : UInt64) : Shape :=
  let batch := input_shape.getD 0 0
  let out_channels := weight_shape.getD 0 0
  let input_length := input_shape.getD 2 0
  let kernel_size := weight_shape.getD 2 0
  let output_length := convOutputSize input_length kernel_size stride padding dilation
  #[batch, out_channels, output_length]

/-- Compute output shape for 1D transposed convolution (safe version). -/
def convTranspose1dShape (input_shape weight_shape : Shape)
    (stride padding output_padding dilation : UInt64) : Shape :=
  let batch := input_shape.getD 0 0
  let out_channels := weight_shape.getD 1 0
  let input_length := input_shape.getD 2 0
  let kernel_size := weight_shape.getD 2 0
  let output_length := convTransposeOutputSize input_length kernel_size stride padding output_padding dilation
  #[batch, out_channels, output_length]

/-- Compute output shape for 2D convolution (safe version) -/
def conv2dShape (input_shape weight_shape : Shape) (stride padding dilation : Shape) : Shape :=
  let batch := input_shape.getD 0 0
  let out_channels := weight_shape.getD 0 0
  let input_height := input_shape.getD 2 0
  let input_width := input_shape.getD 3 0
  let kernel_height := weight_shape.getD 2 0
  let kernel_width := weight_shape.getD 3 0
  let output_height := convOutputSize input_height kernel_height (stride.getD 0 1) (padding.getD 0 0) (dilation.getD 0 1)
  let output_width := convOutputSize input_width kernel_width (stride.getD 1 1) (padding.getD 1 0) (dilation.getD 1 1)
  #[batch, out_channels, output_height, output_width]

/-- Compute output shape for 3D convolution (safe version) -/
def conv3dShape (input_shape weight_shape : Shape) (stride padding dilation : Shape) : Shape :=
  let batch := input_shape.getD 0 0
  let out_channels := weight_shape.getD 0 0
  let input_depth := input_shape.getD 2 0
  let input_height := input_shape.getD 3 0
  let input_width := input_shape.getD 4 0
  let kernel_depth := weight_shape.getD 2 0
  let kernel_height := weight_shape.getD 3 0
  let kernel_width := weight_shape.getD 4 0
  let output_depth := convOutputSize input_depth kernel_depth (stride.getD 0 1) (padding.getD 0 0) (dilation.getD 0 1)
  let output_height := convOutputSize input_height kernel_height (stride.getD 1 1) (padding.getD 1 0) (dilation.getD 1 1)
  let output_width := convOutputSize input_width kernel_width (stride.getD 2 1) (padding.getD 2 0) (dilation.getD 2 1)
  #[batch, out_channels, output_depth, output_height, output_width]

def poolOutputSize (input_size kernel_size : UInt64) (stride : UInt64 := 0) (padding : UInt64 := 0) : UInt64 :=
  let effective_stride := if stride = 0 then kernel_size else stride
  (input_size + 2 * padding - kernel_size) / effective_stride + 1

/-- Compute output shape for 2D pooling (safe version) -/
def pool2dShape (input_shape kernel_size : Shape) (stride padding : Shape) : Shape :=
  let batch := input_shape.getD 0 0
  let channels := input_shape.getD 1 0
  let input_height := input_shape.getD 2 0
  let input_width := input_shape.getD 3 0
  let effective_stride := if stride.size = 0 then kernel_size else stride
  let output_height := poolOutputSize input_height (kernel_size.getD 0 0) (effective_stride.getD 0 0) (padding.getD 0 0)
  let output_width := poolOutputSize input_width (kernel_size.getD 1 0) (effective_stride.getD 1 0) (padding.getD 1 0)
  #[batch, channels, output_height, output_width]



namespace data

@[extern "lean_torch_bin_file_token_count"]
opaque binFileTokenCount (path : @& String) : IO UInt64

/-- High-quality mono resampling. Zero rates, unrepresentable output sizes, and
native failures return IO errors. Equal-rate and empty inputs retain the input array. -/
@[extern "lean_torch_resample_soxr_hq"]
opaque resampleSoxrHQ (samples : @& Array Float) (origSr targetSr : UInt64) : IO (Array Float)

@[extern "lean_torch_wav_begin"]
opaque wavBegin (path : @& String) (sampleRate : UInt64 := 24000) : IO Unit

@[extern "lean_torch_wav_finalize"]
opaque wavFinalize (path : @& String) : IO Unit

@[extern "lean_torch_file_exists"]
opaque fileExists (path : @& String) : IO Bool

end data

@[extern "lean_torch_get_live_tensors"]
opaque get_live_tensors : IO UInt64

/-- Seed the default CPU generator. CUDA generators are currently unchanged. -/
@[extern "lean_torch_manual_seed"]
opaque manualSeed (seed : UInt64) : IO Unit





end torch
