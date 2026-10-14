import Tyr.Legacy.Tensor
import Tyr.Differentiable

/-!
# Tyr.Torch

`Tyr.Torch` is the core tensor runtime layer for Tyr.
It exposes shape-aware Lean declarations backed by native libtorch bindings and
provides the low-level operations used by higher-level model, optimizer, and training code.

## Major Components

- Tensor creation and basic arithmetic (`zeros`, `ones`, `rand`, scalar/tensor ops).
- Device/runtime interop (CPU/CUDA/MPS selection, stream access, synchronization).
- Shape-indexed tensor transforms (`reshape`, `permute`, slicing, concatenation, expansion).
- Core numerics (matrix multiplication, reductions, pooling/convolution shape helpers).
- `torch.autograd`: gradient queries, graph control (`no_grad`), and differentiation utilities.
- Neural operators and losses use the indexed `torch.Tensor` API.
- Remaining legacy `torch.data` adapters and non-tensor data utilities.
- SafeTensors loading now lives in `Tyr.SafeTensors`, with indexed results.
- Indexed numerics and rotary helpers now live in `torch.Tensor`.

## Scope

This module is intentionally close to the runtime/FFI boundary, but it is also a
primary direct import surface for model and training code.
In this repository, many `Examples` and `Tyr.Model` modules import `Tyr.Torch`
directly to work in the `torch` namespace and use tensor/autograd/device primitives
without pulling the full top-level aggregation.
Use `import Tyr` for convenience; use `import Tyr.Torch` when you want explicit,
lower-level control and a narrower dependency surface.
-/

namespace torch

-- | Tensor Creation API
-- Creation functions with a device parameter default to CPU.
/-- Length of an increasing, end-exclusive range. Compute in `Nat` so the
    ceiling division cannot overflow `UInt64`. A zero step is rejected by
    the runtime. -/
def arangeLength (start stop step : UInt64) : UInt64 :=
  if step == 0 || stop <= start then 0
  else ((stop.toNat - start.toNat + step.toNat - 1) / step.toNat).toUInt64

@[extern "lean_torch_arange"] opaque arange (start : UInt64) (stop : UInt64) (step : UInt64 := 1) : T #[arangeLength start stop step]
@[extern "lean_torch_eye"] opaque eye (n : UInt64) (requires_grad : Bool := false) : T #[n, n]
-- full: Returns a tensor filled with a single value
@[extern "lean_torch_full"] opaque full (s : Shape) (value : Float) (requires_grad : Bool := false) (device : Device := Device.CPU) : T s
@[extern "lean_torch_linspace"] opaque linspace (start : Float) (stop : Float) (steps : UInt64) (requires_grad : Bool := false) : T #[steps]
@[extern "lean_torch_logspace"] opaque logspace (start : Float) (stop : Float) (steps : UInt64) (base : Float := 10.0) (requires_grad : Bool := false) : T #[steps]
-- rand: Returns a tensor filled with values drawn from a uniform distribution on [0, 1)
@[extern "lean_torch_rand"] opaque rand (s : Shape) (requires_grad : Bool := false) (device : Device := Device.CPU) : IO (T s)
-- ones: Returns a tensor filled with all ones
@[extern "lean_torch_ones"] opaque ones (s : Shape) (requires_grad : Bool := false) (device : Device := Device.CPU) : T s
@[extern "lean_torch_randint"] opaque randint (low : Int64) (high : Int64) (s : Shape) (requires_grad : Bool := false) (device : Device := Device.CPU) : IO (T s)
-- zeros: Returns a tensor filled with all zeros
@[extern "lean_torch_zeros"] opaque zeros (s : Shape) (requires_grad : Bool := false) (device : Device := Device.CPU) : T s
-- randn: Returns a tensor filled with values drawn from a unit normal distribution
@[extern "lean_torch_randn"] opaque randn (s : Shape) (requires_grad : Bool := false) (device : Device := Device.CPU) : IO (T s)

-- zeros_like: Returns a tensor filled with zeros, same shape/device as input
@[extern "lean_torch_zeros_like"] opaque zeros_like {s : Shape} (t : @& T s) : T s
-- ones_like: Returns a tensor filled with ones, same shape/device as input
@[extern "lean_torch_ones_like"] opaque ones_like {s : Shape} (t : @& T s) : T s
/-- Copy `src` into the already-allocated `dst`, matching PyTorch copy_. -/
@[extern "lean_torch_copy_"] opaque copy_ {s : Shape} (dst src : @& T s) : IO Unit

@[extern "lean_torch_requires_grad"] opaque T.requires_grad {s : Shape} (t : @& T s) : Bool

instance (shape : Shape) : Inhabited (T shape) := ⟨zeros shape⟩

@[extern "lean_torch_tensor_add"] opaque add {s : Shape} (t t' : @& T s) : T s
@[extern "lean_torch_tensor_sub"] opaque sub {s : Shape} (t t' : @& T s) : T s
@[extern "lean_torch_tensor_mul"] opaque mul {s : Shape} (t t' : @& T s) : T s

-- Scalar-tensor operations
@[extern "lean_torch_mul_scalar"] opaque mul_scalar {s : Shape} (t : @& T s) (scalar : Float) : T s
@[extern "lean_torch_div_scalar"] opaque div_scalar {s : Shape} (t : @& T s) (scalar : Float) : T s
@[extern "lean_torch_add_scalar"] opaque add_scalar {s : Shape} (t : @& T s) (scalar : Float) : T s
@[extern "lean_torch_sub_scalar"] opaque sub_scalar {s : Shape} (t : @& T s) (scalar : Float) : T s

instance {shape : Shape} : Add (T shape) where
  add := add
instance {shape : Shape} : Sub (T shape) where
  sub := sub
instance {shape : Shape} : Mul (T shape) where
  mul := mul
instance {shape : Shape} : HMul (T shape) Float (T shape) where
  hMul := mul_scalar
instance {shape : Shape} : HDiv (T shape) Float (T shape) where
  hDiv := div_scalar
instance {shape : Shape} : HAdd (T shape) Float (T shape) where
  hAdd := add_scalar
instance {shape : Shape} : HSub (T shape) Float (T shape) where
  hSub := sub_scalar


def uniform (s : Shape) (min : Float := 0.0) (max : Float := 1.0) : IO (T s) := do
  let max := full s max;
  let min := full s min;
  let x ← rand s;
  return (min + (max - min) * x)

@[extern "lean_torch_get"] opaque T.getOp {s : Shape} (self : @& T s) (idx : Int) : T (s[1:].toArray)
@[extern "lean_torch_to"] opaque T.to {s : Shape} (self : @& T s) (device : Device) : T s

@[extern "lean_torch_tensor_relu"] opaque relu {s : Shape} (t : @& T s) : T s
@[extern "lean_torch_tensor_relu6"] opaque relu6 {s : Shape} (t : @& T s) : T s
@[extern "lean_torch_rsqrt"] opaque rsqrt {s : Shape} (input : @& T s) : T s
@[extern "lean_torch_to_float"] opaque toFloat' {s : Shape} (input : @& T s) : T s

/-- Broadcasting tensor-tensor multiplication -/
@[extern "lean_torch_tensor_mul_diff"] opaque mul' {s1 s2 : Shape} (t1 : @& T s1) (t2 : @& T s2) : T s2

instance {s1 s2 : Shape} : HMul (T s1) (T s2) (T s2) where
  hMul := mul'

/-- Stack an array of 1D tensors into a 2D tensor -/
@[extern "lean_torch_stack_1d"]
opaque stack1d {n k : UInt64} (tensors : Array (T #[n])) (dim : Int64 := 0) : T (if dim == 0 then #[k, n] else #[n, k])

/-- Unbind a tensor along a dimension into an array of tensors -/
@[extern "lean_torch_unbind"]
opaque unbind {s : Shape} (t : @& T s) (dim : Nat := 0) : Array (T (unbindShape s dim))

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

/-- In-place whole-tensor copy (`dst.copy_(src)`, converting dtype if needed).
    Mutates `dst`; only use on uniquely-owned tensors. Used by the CUDA-graph
    static-buffer discipline. -/
@[extern "lean_torch_copy_inplace"]
opaque copyInplace {s : Shape} (dst : @& T s) (src : @& T s) : T s

/-- No-op that forces its tensor argument at the FFI boundary. Lean `let`
    bindings are lazy, so staging computations for CUDA graph capture/replay
    (static buffers, in-place copies) would otherwise be evaluated *inside*
    the captured region; `touch` forces them eagerly beforehand. -/
@[extern "lean_torch_touch"]
opaque touch {s : Shape} (t : @& T s) : IO Unit

/-- Elementwise multiply by a 0-dim tensor (broadcast scalar). Lets captured
    CUDA graphs take scalar inputs (e.g. the learning rate) through a static
    buffer rather than a baked-in Float. -/
@[extern "lean_torch_mul_tensor_scalar"]
opaque mulTensorScalar {s : Shape} (input : @& T s) (scalar : @& T #[]) : T s

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
  if ← mps_is_available_stable then return Device.MPS
  if ← cuda_is_available then return Device.CUDA 0
  return Device.CPU
@[extern "lean_torch_linear"] opaque linear {m n b : UInt64} (x : @& T #[b, m]) (M : @& T #[n,m]) : T #[b, n]
@[extern "lean_torch_affine"] opaque affine {m n b : UInt64} (x : @& T #[b, m]) (M : @& T #[n,m]) (bias : @& T #[n]) : T #[b, n]

/-- Linear projection for 3D input: [batch, seq, in] @ [out, in]^T -> [batch, seq, out] -/
@[extern "lean_torch_linear3d"]
opaque linear3d {batch seq in_dim out_dim : UInt64}
    (x : @& T #[batch, seq, in_dim]) (weight : @& T #[out_dim, in_dim]) : T #[batch, seq, out_dim]

/-- Affine (linear + bias) for 3D input -/
@[extern "lean_torch_affine3d"]
opaque affine3d {batch seq in_dim out_dim : UInt64}
    (x : @& T #[batch, seq, in_dim]) (weight : @& T #[out_dim, in_dim]) (bias : @& T #[out_dim])
    : T #[batch, seq, out_dim]

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

@[extern "lean_torch_slice"] opaque T.slice {s : Shape} (self : @& T s) (dim : Nat := 0) (start : Int64 := 0) (stop : Int64 := -1) (step : Int64 := 1) : T s
--
--
namespace autograd

@[extern "lean_torch_set_grad_enabled"] opaque set_grad_enabled (enabled : Bool) : IO Unit
@[extern "lean_torch_is_grad_enabled"] opaque is_grad_enabled : IO Bool
/-- Run an IO action with gradient computation disabled.
    This prevents building computation graphs for validation/inference. -/
def no_grad {α : Type} (action : IO α) : IO α := do
  let wasEnabled ← is_grad_enabled
  let _ ← set_grad_enabled false
  try
    action
  finally
    let _ ← set_grad_enabled wasEnabled

end autograd
--
--
--
universe u


--
--
--
--
--
-- -- def rjvp {a b : Type} [differentiable a] [differentiable b] (f : a → b) (x : a) : b × (differentiable.cotangent_space b → differentiable.cotangent_space a) :=
-- --     let y := f x;
-- --     let fT := λ (dy : differentiable.cotangent_space b) => torch.backward y dy
-- --     (y, fT)
--
--

--

/-- Compute output shape for permute operation (safe version) -/
def permuteShape (s : Shape) (permutation : Array UInt64) : Shape :=
  permutation.map fun p => s.getD p.toNat 0

@[extern "lean_torch_permute"] opaque permute {s : Shape} (t : @& T s) (permutation : Array UInt64) : T (permuteShape s permutation)
/-- Legacy raw reshape. An empty target erases the shape annotation without
    reshaping the handle. Use `reshapeExact` for a real scalar reshape, and
    `Tensor.reshape` for a statically checked element count. -/
@[extern "lean_torch_reshape"] opaque reshape {s : Shape} (t : @& T s) (s' : @& Shape) : T s'
@[extern "lean_torch_permute"] opaque T.permute {s : Shape} (self : @& T s) (permutation : Array UInt64) : T (permuteShape s permutation)
@[extern "lean_torch_reshape"] opaque T.reshape {s : Shape} (self : @& T s) (s' : @& Shape) : T s'

/-- LibTorch reshape with exact target semantics, including `#[]` for a
    scalar. The raw API checks element counts at runtime; the typed wrapper
    additionally requires a static proof. -/
@[extern "lean_torch_reshape_exact"] opaque reshapeExact {s : Shape}
    (t : @& T s) (s' : @& Shape) : T s'

-- comparison
@[extern "lean_torch_allclose"] opaque allclose (a b : @& T #[]) (rtol : Float := 1e-05) (atol : Float := 1e-08): Bool


-- Extended autograd operations


-- Data loading utilities
namespace data

/-- Get the number of uint16 tokens in a binary file -/
@[extern "lean_torch_bin_file_token_count"]
opaque binFileTokenCount (path : @& String) : IO UInt64

/-- Load a binary file of uint16 tokens into a 1D int64 tensor with known size -/
@[extern "lean_torch_load_u16_bin"]
opaque loadU16Bin (n : UInt64) (path : @& String) : IO (T #[n])

/-- Load raw float32 binary data into a tensor of known shape.
    The file must contain exactly `numel(shape)` float32 values in row-major order. -/
@[extern "lean_torch_load_f32_bin"]
opaque loadF32Bin (s : Shape) (path : @& String) : IO (T s)

/-- Create a 1D int64 tensor from an array of Int64 -/
@[extern "lean_torch_from_int64_array"]
opaque fromInt64Array (arr : @& Array Int64) : T #[]

/-- Create a 1D float32 tensor from an array of Float. -/
@[extern "lean_torch_from_float_array"]
opaque fromFloatArray (arr : @& Array Float) : T #[]

/-- Apple-only image preprocessing to patchified tensor expected by Qwen3.5-VL.
    Returns shape-erased tensor with runtime shape `[nPatches, patchDim]`.
    Requires macOS system media frameworks at build/runtime. -/
@[extern "lean_torch_media_load_image_patchified"]
opaque loadImagePatchified
    (path : @& String)
    (inChannels : UInt64)
    (patchSize : UInt64)
    (temporalPatchSize : UInt64)
    : IO (T #[])

/-- Apple-only video preprocessing to patchified tensor expected by Qwen3.5-VL.
    Returns shape-erased tensor with runtime shape `[nPatches, patchDim]`.
    `maxFrames` bounds sampled frames and `frameStride` keeps every Nth decoded frame.
    Requires macOS system media frameworks at build/runtime. -/
@[extern "lean_torch_media_load_video_patchified"]
opaque loadVideoPatchified
    (path : @& String)
    (inChannels : UInt64)
    (patchSize : UInt64)
    (temporalPatchSize : UInt64)
    (maxFrames : UInt64)
    (frameStride : UInt64)
    : IO (T #[])

/-- Apple-only image preprocessing for Gemma 4 multimodal inputs.
    Returns a shape-erased tensor with runtime shape
    `[patchRows, patchCols, 3 * patchSize * patchSize]`, where `patchRows` and
    `patchCols` are chosen to fit the requested visual token budget after
    `poolingKernelSize x poolingKernelSize` pooling.
    Requires macOS system media frameworks at build/runtime. -/
@[extern "lean_torch_media_load_image_patch_grid_gemma4"]
opaque loadGemma4ImagePatchGrid
    (path : @& String)
    (patchSize : UInt64)
    (poolingKernelSize : UInt64)
    (maxSoftTokens : UInt64)
    (rescaleFactor : Float := 1.0 / 255.0)
    : IO (T #[])

/-- Resample mono waveform samples with libsoxr high-quality mode.
    Mirrors librosa default `res_type=\"soxr_hq\"` behavior for 1D audio. -/
@[extern "lean_torch_resample_soxr_hq"]
opaque resampleSoxrHQ (samples : @& Array Float) (origSr targetSr : UInt64) : IO (Array Float)

/-- Slice a 1D tensor: data[start:end] (shape-erased) -/
@[extern "lean_torch_slice_1d"]
opaque slice1d' {n : UInt64} (data : @& T #[n]) (start : Int64) (stop : Int64) : T #[]

/-- Slice with known output size -/
def slice1d {n m : UInt64} (data : T #[n]) (start : Int64) (stop : Int64) : T #[m] :=
  reshape (slice1d' data start stop) #[m]

/-- Compute the output shape after slicing along a dimension -/
def sliceShape (s : Shape) (dim : UInt64) (len : UInt64) : Shape :=
  s.set! dim.toNat len

/-- Slice a tensor along a specified dimension: data.slice(dim, start, start+len) -/
@[extern "lean_torch_slice_along_dim"]
opaque slice {s : Shape} (data : @& T s) (dim : UInt64 := 0) (start len : UInt64)
    : T (sliceShape s dim len)

/-- Return `data` with a slice along `dim` replaced by `src` starting at `start`.
    The length on `dim` is inferred from `src`. -/
@[extern "lean_torch_slice_scatter_along_dim"]
opaque sliceScatter {s src : Shape}
    (data : @& T s)
    (dim : UInt64 := 0)
    (start : UInt64)
    (src : @& T src)
    : T s

/-- Like `sliceScatter`, but writes `src` into `data` in place and returns the
    same tensor, avoiding a full clone (e.g. for KV-cache append).
    WARNING: mutates its argument — only use on uniquely-owned tensors; the
    mutation is observable through any other reference to the same tensor. -/
@[extern "lean_torch_slice_scatter_along_dim_inplace"]
opaque sliceScatterInplace {s src : Shape}
    (data : @& T s)
    (dim : UInt64 := 0)
    (start : UInt64)
    (src : @& T src)
    : T s

/-- Slice a 2D tensor along dimension 0: data[start:start+len, :] -/
@[extern "lean_torch_slice_2d"]
opaque slice2d {n d : UInt64} (data : @& T #[n, d]) (start len : UInt64) : T #[len, d]

/-- Convert tensor to Long (int64) dtype -/
@[extern "lean_torch_to_long"]
opaque toLong {s : Shape} (t : @& T s) : T s

/-- Index select: gather elements along a dimension using indices
    For 1D: data[indices] where indices is 1D tensor of positions -/
@[extern "lean_torch_index_select"]
opaque indexSelect {n k : UInt64} (data : @& T #[n]) (dim : Int64) (indices : @& T #[k]) : T #[k]

/-- Save tensor to a file -/
@[extern "lean_torch_save_tensor"]
opaque saveTensor {s : Shape} (t : @& T s) (path : @& String) : IO Unit

/-- Save image tensor as PPM file.
    Expects tensor of shape [batch, 3, height, width] with values in [-1, 1].
    Saves the first image in the batch. -/
@[extern "lean_torch_save_ppm"]
opaque savePPMExplicit (s : Shape) (t : @& T s) (path : @& String) : IO Unit

/-- Save mono waveform tensor as 16-bit PCM WAV.
    Tensor is flattened in row-major order and clamped to [-1, 1]. -/
@[extern "lean_torch_save_wav"]
opaque saveWav {s : Shape} (t : @& T s) (path : @& String) (sampleRate : UInt64 := 24000) : IO Unit

/-- Start a streaming WAV writer at `path` with 16-bit PCM mono format. -/
@[extern "lean_torch_wav_begin"]
opaque wavBegin (path : @& String) (sampleRate : UInt64 := 24000) : IO Unit

/-- Append mono waveform samples to an existing WAV file started by `wavBegin`. -/
@[extern "lean_torch_wav_append"]
opaque wavAppend {s : Shape} (t : @& T s) (path : @& String) : IO Unit

/-- Finalize a streaming WAV file (header size fix-up/validation). -/
@[extern "lean_torch_wav_finalize"]
opaque wavFinalize (path : @& String) : IO Unit

/-- Load the first tensor and reshape it to the requested shape when the
    element count agrees. Use `loadTensorExact` to validate stored dimensions. -/
@[extern "lean_torch_load_tensor"]
opaque loadTensor (s : Shape) (path : @& String) : IO (T s)

/-- Load the first tensor only if its stored shape exactly matches `s`.
    Reports an IO error on mismatch; never repairs dimensions by reshaping. -/
@[extern "lean_torch_load_tensor_exact"]
opaque loadTensorExact (s : Shape) (path : @& String) : IO (T s)

/-- Check if a file exists -/
@[extern "lean_torch_file_exists"]
opaque fileExists (path : @& String) : IO Bool

/-- Find positions of a specific token (e.g., BOS) in a 1D tensor.
    Returns a 1D tensor of indices where tokens == tokenId. -/
@[extern "lean_torch_find_bos_positions"]
opaque findBosPositions {n : UInt64} (tokens : @& T #[n]) (tokenId : Int64) : IO (T #[])

/-- Convert a 1D int64 tensor to a Lean Array UInt64.
    Useful for extracting positions for Lean-side processing. -/
@[extern "lean_torch_tensor_to_uint64_array"]
opaque tensorToUInt64Array {n : UInt64} (t : @& T #[n]) : IO (Array UInt64)

/-- Convert a dynamically-shaped tensor to a Lean Array UInt64.
    Works with shape-erased tensors (T #[]). -/
@[extern "lean_torch_tensor_to_uint64_array_dynamic"]
opaque tensorToUInt64Array' (t : @& T #[]) : IO (Array UInt64)

/-- Convert a dynamically-shaped tensor to a Lean Array Float. -/
@[extern "lean_torch_tensor_to_float_array_dynamic"]
opaque tensorToFloatArray' (t : @& T #[]) : IO (Array Float)

end data


-- SafeTensors loading utilities (alias for data.loadTensor with name-based loading)



@[extern "lean_torch_get_live_tensors"]
opaque get_live_tensors : IO UInt64

/-- Set global RNG seed for deterministic runs (CPU + CUDA generators). -/
@[extern "lean_torch_manual_seed"]
opaque manualSeed (seed : UInt64) : IO Unit




-- ============================================================================
-- Comparison and conditional operations (for diffusion)
-- ============================================================================

/-- Less than comparison: a < b element-wise, returns boolean tensor -/
@[extern "lean_torch_lt"]
opaque lt {s : Shape} (a : @& T s) (b : @& T s) : T s

/-- Less than scalar comparison: input < scalar -/
@[extern "lean_torch_lt_scalar"]
opaque lt_scalar {s : Shape} (input : @& T s) (scalar : Float) : T s

/-- Greater than comparison: a > b element-wise -/
@[extern "lean_torch_gt"]
opaque gt {s : Shape} (a : @& T s) (b : @& T s) : T s

/-- Greater than or equal comparison: a >= b element-wise -/
@[extern "lean_torch_ge"]
opaque ge {s : Shape} (a : @& T s) (b : @& T s) : T s

/-- Equality comparison: a == b element-wise, returns boolean tensor -/
@[extern "lean_torch_eq"]
opaque eq {s : Shape} (a : @& T s) (b : @& T s) : T s

/-- Equality with scalar comparison: input == scalar -/
@[extern "lean_torch_eq_scalar"]
opaque eq_scalar {s : Shape} (input : @& T s) (scalar : Int64) : T s

/-- Conditional selection: where condition is true, select x, else y -/
@[extern "lean_torch_where"]
opaque where_ {s : Shape} (condition : @& T s) (x : @& T s) (y : @& T s) : T s

/-- Create int64 tensor filled with given value -/
@[extern "lean_torch_full_int"]
opaque full_int (s : Shape) (value : Int64) : T s

/-- Maximum along dimension, returns (values, indices).
    WARNING: This signature is incorrect - the output shape should have dim removed.
    Use max_dim_3d for 3D tensors instead. -/
@[extern "lean_torch_max_dim"]
opaque max_dim {s : Shape} (input : @& T s) (dim : UInt64) : T s × T s

/-- Maximum along last dimension for 3D tensors [batch, seq, vocab] -> [batch, seq].
    Returns (max_values, argmax_indices). -/
@[extern "lean_torch_max_dim_3d"]
opaque max_dim_3d {d0 d1 d2 : UInt64} (input : @& T #[d0, d1, d2]) (dim : UInt64)
    : T #[d0, d1] × T #[d0, d1]

/-- Boolean any: returns true if any element is true -/
@[extern "lean_torch_any"]
opaque any {s : Shape} (input : @& T s) : Bool

/-- Logical NOT for boolean tensors -/
@[extern "lean_torch_logical_not"]
opaque logical_not {s : Shape} (input : @& T s) : T s

/-- Logical AND for boolean tensors -/
@[extern "lean_torch_logical_and"]
opaque logical_and {s : Shape} (a : @& T s) (b : @& T s) : T s

/-- Logical OR for boolean tensors (De Morgan). -/
def logicalOr {s : Shape} (a b : T s) : T s :=
  logical_not (logical_and (logical_not a) (logical_not b))

/-- Boolean mask of all false values on a device. -/
def falseMask {n : UInt64} (device : Device) : T #[n] :=
  let zeros : T #[n] := (full_int #[n] 0).to device
  eq_scalar zeros 1

/-- Zero tensor on a specific device. -/
def zerosOn {s : Shape} (device : Device) : T s :=
  torch.zeros s false device

/-- Ones tensor on a specific device. -/
def onesOn {s : Shape} (device : Device) : T s :=
  torch.ones s false device

/-- Convert to bfloat16 dtype. -/
@[extern "lean_torch_to_bfloat16"]
opaque toBFloat16' {s : Shape} (input : @& T s) : T s

/-- Cast `t` to the same dtype as `reference`. -/
def castLike {sRef s : Shape} (reference : T sRef) (t : T s) : T s :=
  match reference.dtype with
  | .BFloat16 => toBFloat16' t
  | _ => t

/-- Alias for `castLike` with argument names matching the "restore input dtype" pattern. -/
def restoreInputDType {s : Shape} (input : T s) (output : T s) : T s :=
  castLike input output

/-- Index select on first dimension -/
@[extern "lean_torch_index_select_1d"]
opaque index_select_1d {n m : UInt64} (input : @& T #[n]) (indices : @& T #[m]) : T #[m]

/-- Gather elements along an axis using indices.
    output[i][j][k] = input[index[i][j][k]][j][k]  (if dim=0)
    The output has the same shape as indices. -/
@[extern "lean_torch_gather"]
opaque gather {input_shape index_shape : Shape}
    (input : @& T input_shape)
    (dim : Int64)
    (indices : @& T index_shape)
    : T index_shape

/-- Scatter src values into input at positions specified by indices.
    output = input.clone(); output[indices] = src (along dim)
    Returns a new tensor; does not modify input in place. -/
@[extern "lean_torch_scatter"]
opaque scatter {s : Shape}
    (input : @& T s)
    (dim : Int64)
    (indices : @& T s)
    (src : @& T s)
    : T s

/-- Scatter with add reduction: accumulates values at indices -/
@[extern "lean_torch_scatter_add"]
opaque scatter_add {s : Shape}
    (input : @& T s)
    (dim : Int64)
    (indices : @& T s)
    (src : @& T s)
    : T s

/-- Scatter with different shapes: scatter k values into seq positions.
    Useful for top-k style operations where indices/src have fewer elements.
    input: [batch, seq], indices: [batch, k], src: [batch, k] -> [batch, seq] -/
@[extern "lean_torch_scatter_2d"]
opaque scatter_2d {batch seq k : UInt64}
    (input : @& T #[batch, seq])
    (dim : Int64)
    (indices : @& T #[batch, k])
    (src : @& T #[batch, k])
    : T #[batch, seq]

/-- Einstein summation: general tensor contraction using index notation.
    Example: "ij,jk->ik" for matrix multiplication
    Returns shape-erased tensor (use reshape if needed). -/
@[extern "lean_torch_einsum"]
opaque einsum (equation : String) (tensors : @& Array (T #[])) : T #[]

/-- Einsum with 2 input tensors (common case) -/
@[extern "lean_torch_einsum2"]
opaque einsum2 {s1 s2 : Shape}
    (equation : String)
    (a : @& T s1)
    (b : @& T s2)
    : T #[]

/-- Interpolate (resize) tensor to target size.
    Supports: "nearest", "linear", "bilinear", "bicubic", "trilinear", "area"
    input should be 3D (1D signal), 4D (2D image), or 5D (3D volume) -/
@[extern "lean_torch_interpolate"]
opaque interpolate {s : Shape}
    (input : @& T s)
    (size : Array UInt64)
    (mode : String := "nearest")
    (align_corners : Bool := false)
    : T #[]

/-- Interpolate with scale factor instead of target size -/
@[extern "lean_torch_interpolate_scale"]
opaque interpolate_scale {s : Shape}
    (input : @& T s)
    (scale_factor : Array Float)
    (mode : String := "nearest")
    (align_corners : Bool := false)
    : T #[]

/-- Clamp values to a range -/
@[extern "lean_torch_clamp"]
opaque clamp {s : Shape} (input : @& T s) (min_val max_val : Int64) : T s

/-- Clamp values to a floating-point range -/
@[extern "lean_torch_clamp_float"]
opaque clampFloat {s : Shape} (input : @& T s) (min_val max_val : Float) : T s

/-- Top-k values and indices along dimension -/
@[extern "lean_torch_topk"]
opaque topk {s : Shape} (input : @& T s) (k : UInt64) (dim : UInt64) : T s × T s

/-- Top-k values and indices for 2D tensors with proper output shape.
    input: [d1, d2], dim=1 -> output: [d1, k] for both values and indices -/
@[extern "lean_torch_topk_2d"]
opaque topk_2d {d1 d2 : UInt64}
    (input : @& T #[d1, d2]) (k : UInt64) (dim : UInt64)
    : T #[d1, k] × T #[d1, k]

end torch
