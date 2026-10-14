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
- `torch.nn`: neural network functional operators and loss primitives.
- `torch.data` and `torch.signal`: data and signal-processing helpers.
- `torch.safetensors`: tensor checkpoint loading/saving, including sharded loading.
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

namespace nn
-- torch::nn::functional::adaptive_avg_pool1d
@[extern "lean_torch_adaptive_avg_pool2d"] opaque adaptive_avg_pool2d {input_shape : Shape} (input : @& T input_shape) (output_size : Array UInt64) : T (#[input_shape.getD 0 0, input_shape.getD 1 0] ++ output_size)
-- torch::nn::functional::adaptive_avg_pool3d
-- torch::nn::functional::adaptive_max_pool1d
-- torch::nn::functional::adaptive_max_pool2d
-- torch::nn::functional::adaptive_max_pool2d_with_indices
-- torch::nn::functional::adaptive_max_pool3d
-- torch::nn::functional::adaptive_max_pool3d_with_indices
-- torch::nn::functional::affine_grid
-- torch::nn::functional::alpha_dropout
-- torch::nn::functional::avg_pool1d
@[extern "lean_torch_avg_pool2d"] opaque avg_pool2d {input_shape : Shape} (input : @& T input_shape) (kernel_size : Array UInt64) (stride : Array UInt64 := #[]) (padding : Array UInt64 := #[0, 0]) : T (pool2dShape input_shape kernel_size stride padding)
-- torch::nn::functional::avg_pool3d
@[extern "lean_torch_batch_norm"] opaque batch_norm {s : Shape} (input : @& T s) (weight : Option (T s)) (bias : Option (T s)) (running_mean : Option (T s)) (running_var : Option (T s)) (training : Bool := true) (momentum : Float := 0.1) (eps : Float := 1e-5) : T s
-- torch::nn::functional::bilinear
@[extern "lean_torch_binary_cross_entropy"] opaque binary_cross_entropy {s : Shape} (input : @& T s) (target : @& T s) (weight : Option (T s) := none) (reduction : String := "mean") : T #[]
-- torch::nn::functional::binary_cross_entropy_with_logits
-- torch::nn::functional::celu
@[extern "lean_torch_conv1d"] opaque conv1d {input_shape weight_shape : Shape} (input : @& T input_shape) (weight : @& T weight_shape) (stride : UInt64 := 1) (padding : UInt64 := 0) (dilation : UInt64 := 1) : T (conv1dShape input_shape weight_shape stride padding dilation)
@[extern "lean_torch_conv1d_group_bias"]
opaque conv1d_group_bias {input_shape weight_shape bias_shape : Shape}
    (input : @& T input_shape)
    (weight : @& T weight_shape)
    (bias : @& T bias_shape)
    (stride : UInt64 := 1)
    (padding : UInt64 := 0)
    (dilation : UInt64 := 1)
    (groups : UInt64 := 1)
    : T (conv1dShape input_shape weight_shape stride padding dilation)
-- torch::nn::functional::conv2d
@[extern "lean_torch_conv2d"] opaque conv2d {input_shape weight_shape : Shape} (input : @& T input_shape) (weight : @& T weight_shape) (stride : Array UInt64 := #[1, 1]) (padding : Array UInt64 := #[0, 0]) (dilation : Array UInt64 := #[1, 1]) : T (conv2dShape input_shape weight_shape stride padding dilation)
-- torch::nn::functional::conv2d with bias
@[extern "lean_torch_conv2d_bias"] opaque conv2d_bias {input_shape weight_shape bias_shape : Shape} (input : @& T input_shape) (weight : @& T weight_shape) (bias : @& T bias_shape) (stride : Array UInt64 := #[1, 1]) (padding : Array UInt64 := #[0, 0]) (dilation : Array UInt64 := #[1, 1]) : T (conv2dShape input_shape weight_shape stride padding dilation)
@[extern "lean_torch_conv3d"] opaque conv3d {input_shape weight_shape : Shape} (input : @& T input_shape) (weight : @& T weight_shape) (stride : Array UInt64 := #[1, 1, 1]) (padding : Array UInt64 := #[0, 0, 0]) (dilation : Array UInt64 := #[1, 1, 1]) : T (conv3dShape input_shape weight_shape stride padding dilation)

/-- Transposed 2D convolution (deconvolution) for upsampling.
    input: [N, C_in, H, W], weight: [C_in, C_out, kH, kW] -/
@[extern "lean_torch_conv_transpose2d"]
opaque conv_transpose2d {input_shape weight_shape : Shape}
    (input : @& T input_shape)
    (weight : @& T weight_shape)
    (stride : Array UInt64 := #[1, 1])
    (padding : Array UInt64 := #[0, 0])
    (output_padding : Array UInt64 := #[0, 0])
    (dilation : Array UInt64 := #[1, 1])
    : T #[]  -- Shape depends on all parameters, computed at runtime

/-- Transposed 1D convolution (deconvolution) with bias.
    input: [N, C_in, T], weight: [C_in, C_out, K], bias: [C_out] -/
@[extern "lean_torch_conv_transpose1d_bias"]
opaque conv_transpose1d_bias {input_shape weight_shape bias_shape : Shape}
    (input : @& T input_shape)
    (weight : @& T weight_shape)
    (bias : @& T bias_shape)
    (stride : UInt64 := 1)
    (padding : UInt64 := 0)
    (output_padding : UInt64 := 0)
    (dilation : UInt64 := 1)
    : T (convTranspose1dShape input_shape weight_shape stride padding output_padding dilation)
-- torch::nn::functional::cosine_embedding_loss
-- torch::nn::functional::cosine_similarity
-- torch::nn::functional::cross_entropy
@[extern "lean_torch_tensor_cross_entropy"] opaque cross_entropy' {s : Shape} (t t' : @& T s) : T s

/-- Shape-aware cross entropy: logits [N, C] + targets [N] -> scalar loss -/
@[extern "lean_torch_cross_entropy_2d"]
opaque cross_entropy {n c : UInt64} (logits : @& T #[n, c]) (targets : @& T #[n]) : T #[]

/-- Cross entropy with no reduction: logits [N, C] + targets [N] -> per-element loss [N] -/
@[extern "lean_torch_cross_entropy_none"]
opaque cross_entropy_none {n c : UInt64} (logits : @& T #[n, c]) (targets : @& T #[n]) : T #[n]
-- torch::nn::functional::ctc_loss
@[extern "lean_torch_dropout"] opaque dropout {s : Shape} (input : @& T s) (p : Float := 0.5) (training : Bool := true) : IO (T s)

/-- 2D spatial dropout: zeros entire channels.
    input: [N, C, H, W] -/
@[extern "lean_torch_dropout2d"]
opaque dropout2d {n c h w : UInt64}
    (input : @& T #[n, c, h, w]) (p : Float := 0.5) (training : Bool := true)
    : IO (T #[n, c, h, w])

/-- 3D spatial dropout: zeros entire channels.
    input: [N, C, D, H, W] -/
@[extern "lean_torch_dropout3d"]
opaque dropout3d {n c d h w : UInt64}
    (input : @& T #[n, c, d, h, w]) (p : Float := 0.5) (training : Bool := true)
    : IO (T #[n, c, d, h, w])
-- torch::nn::functional::elu
@[extern "lean_torch_tensor_elu"] opaque elu {s : Shape} (t : @& T s) : T s
/-- Embedding lookup: input [batch, seq] + weight [vocab, embed] -> [batch, seq, embed] -/
@[extern "lean_torch_embedding"]
opaque embedding {batch seq vocab embed : UInt64}
    (input : @& T #[batch, seq]) (weight : @& T #[vocab, embed])
    (padding_idx : Option Int := none) (max_norm : Option Float := none)
    (norm_type : Float := 2.0) (scale_grad_by_freq : Bool := false)
    (sparse : Bool := false) : T #[batch, seq, embed]

/-- Embedding lookup for 1D input: input [seq] + weight [vocab, embed] -> [seq, embed] -/
@[extern "lean_torch_embedding_1d"]
opaque embedding1d {seq vocab embed : UInt64}
    (input : @& T #[seq]) (weight : @& T #[vocab, embed])
    (padding_idx : Option Int := none) (max_norm : Option Float := none)
    (norm_type : Float := 2.0) (scale_grad_by_freq : Bool := false)
    (sparse : Bool := false) : T #[seq, embed]
-- torch::nn::functional::embedding_bag
-- torch::nn::functional::feature_alpha_dropout
-- torch::nn::functional::fold
-- torch::nn::functional::fractional_max_pool2d
-- torch::nn::functional::fractional_max_pool2d_with_indices
-- torch::nn::functional::fractional_max_pool3d
-- torch::nn::functional::fractional_max_pool3d_with_indices
-- torch::nn::functional::gelu
@[extern "lean_torch_tensor_gelu"] opaque gelu {s : Shape} (t : @& T s) : T s
-- torch::nn::functional::glu
-- torch::nn::functional::grid_sample
@[extern "lean_torch_group_norm"] opaque group_norm {s : Shape} (input : @& T s) (num_groups : UInt64) (weight : Option (T s) := none) (bias : Option (T s) := none) (eps : Float := 1e-5) : T s
-- torch::nn::functional::gumbel_softmax
-- torch::nn::functional::hardshrink
-- torch::nn::functional::hardtanh
-- torch::nn::functional::hinge_embedding_loss
-- torch::nn::functional::huber_loss
@[extern "lean_torch_instance_norm"] opaque instance_norm {s : Shape} (input : @& T s) (running_mean : Option (T s) := none) (running_var : Option (T s) := none) (weight : Option (T s) := none) (bias : Option (T s) := none) (use_input_stats : Bool := true) (momentum : Float := 0.1) (eps : Float := 1e-5) : T s
-- torch::nn::functional::interpolate
-- torch::nn::functional::kl_div
@[extern "lean_torch_l1_loss"] opaque l1_loss {s : Shape} (input : @& T s) (target : @& T s) (reduction : String := "mean") : T #[]
@[extern "lean_torch_layer_norm"] opaque layer_norm' {s : Shape} (input : @& T s) (normalized_shape : Array UInt64) (weight : Option (T s) := none) (bias : Option (T s) := none) (eps : Float := 1e-5) : T s

/-- Shape-aware layer norm for 3D tensors: normalizes over last dimension -/
@[extern "lean_torch_layer_norm_3d"]
opaque layer_norm {batch seq n : UInt64}
    (input : @& T #[batch, seq, n])
    (weight : @& T #[n])
    (bias : @& T #[n])
    (eps : Float := 1e-5) : T #[batch, seq, n]

/-- Fused layer_norm + GELU - saves memory bandwidth in transformer blocks -/
@[extern "lean_torch_layer_norm_gelu"]
opaque layer_norm_gelu {batch seq n : UInt64}
    (input : @& T #[batch, seq, n])
    (weight : @& T #[n])
    (bias : @& T #[n])
    (eps : Float := 1e-5) : T #[batch, seq, n]

/-- Fused layer_norm + ReLU -/
@[extern "lean_torch_layer_norm_relu"]
opaque layer_norm_relu {batch seq n : UInt64}
    (input : @& T #[batch, seq, n])
    (weight : @& T #[n])
    (bias : @& T #[n])
    (eps : Float := 1e-5) : T #[batch, seq, n]

/-- Fused layer_norm + SiLU (for SwiGLU MLP) -/
@[extern "lean_torch_layer_norm_silu"]
opaque layer_norm_silu {batch seq n : UInt64}
    (input : @& T #[batch, seq, n])
    (weight : @& T #[n])
    (bias : @& T #[n])
    (eps : Float := 1e-5) : T #[batch, seq, n]

@[extern "lean_torch_leaky_relu"] opaque leaky_relu {s : Shape} (input : @& T s) (negative_slope : Float := 0.01) : T s
-- torch::nn::functional::linear
-- torch::nn::functional::local_response_norm
@[extern "lean_torch_log_softmax"] opaque log_softmax (input : @& T #[]) (dim : Int32 := -1) : T #[]
-- torch::nn::functional::logsigmoid
-- torch::nn::functional::lp_pool1d
-- torch::nn::functional::lp_pool2d
-- torch::nn::functional::margin_ranking_loss
-- torch::nn::functional::max_pool1d
-- torch::nn::functional::max_pool1d_with_indices
@[extern "lean_torch_max_pool2d"] opaque max_pool2d {input_shape : Shape} (input : @& T input_shape) (kernel_size : Array UInt64) (stride : Array UInt64 := #[]) (padding : Array UInt64 := #[0, 0]) : T (pool2dShape input_shape kernel_size stride padding)
-- torch::nn::functional::max_pool2d_with_indices
-- torch::nn::functional::max_pool3d
-- torch::nn::functional::max_pool3d_with_indices
-- torch::nn::functional::max_unpool1d
-- torch::nn::functional::max_unpool2d
-- torch::nn::functional::max_unpool3d
-- torch::nn::functional::mish
@[extern "lean_torch_mse_loss"] opaque mse_loss {s : Shape} (input : @& T s) (target : @& T s) (reduction : String := "mean") : T #[]
-- torch::nn::functional::multi_head_attention_forward
-- torch::nn::functional::multi_margin_loss
-- torch::nn::functional::multilabel_margin_loss
-- torch::nn::functional::multilabel_soft_margin_loss

/-- Negative log likelihood loss: log_probs [N, C] + targets [N] -> scalar loss.
    Note: Input should be log-probabilities (use log_softmax first). -/
@[extern "lean_torch_nll_loss"]
opaque nll_loss {n c : UInt64}
    (log_probs : @& T #[n, c])
    (targets : @& T #[n])
    (reduction : String := "mean")
    : T #[]

/-- NLL loss with no reduction: returns per-element loss [N] -/
@[extern "lean_torch_nll_loss_none"]
opaque nll_loss_none {n c : UInt64}
    (log_probs : @& T #[n, c])
    (targets : @& T #[n])
    : T #[n]

-- Note: focal_loss and triplet_margin_loss are defined after end nn (see below)

-- torch::nn::functional::normalize
-- torch::nn::functional::one_hot
-- torch::nn::functional::pad
-- torch::nn::functional::pairwise_distance
-- torch::nn::functional::pdist
-- torch::nn::functional::pixel_shuffle
-- torch::nn::functional::pixel_unshuffle
-- torch::nn::functional::poisson_nll_loss
-- torch::nn::functional::prelu
-- torch::nn::functional::rrelu
@[extern "lean_torch_tensor_sigmoid"] opaque sigmoid {s : Shape} (t : @& T s) : T s
@[extern "lean_torch_tensor_silu"] opaque silu {s : Shape} (t : @& T s) : T s

@[extern "lean_torch_tensor_cos"] opaque cos {s : Shape} (t : @& T s) : T s
@[extern "lean_torch_tensor_sin"] opaque sin {s : Shape} (t : @& T s) : T s
@[extern "lean_torch_tensor_atan"] opaque atan {s : Shape} (t : @& T s) : T s
@[extern "lean_torch_tensor_floor"] opaque floor {s : Shape} (t : @& T s) : T s
@[extern "lean_torch_tensor_exp"] opaque exp {s : Shape} (t : @& T s) : T s
@[extern "lean_torch_tensor_log"] opaque log {s : Shape} (t : @& T s) : T s
@[extern "lean_torch_tensor_log10"] opaque log10 {s : Shape} (t : @& T s) : T s
@[extern "lean_torch_atan2"] opaque atan2 {s : Shape} (y : @& T s) (x : @& T s) : T s
@[extern "lean_torch_smooth_l1_loss"] opaque smooth_l1_loss {s : Shape} (input : @& T s) (target : @& T s) (reduction : String := "mean") (beta : Float := 1.0) : T #[]
-- torch::nn::functional::soft_margin_loss
-- torch::nn::functional::softmax
@[extern "lean_torch_tensor_softmax"] opaque softmax {s : Shape} (t : @& T s) (dim : Int32 := -1) : T s
-- torch::nn::functional::softmin
-- torch::nn::functional::softplus
@[extern "lean_torch_tensor_softplus"] opaque softplus {s : Shape} (t : @& T s) : T s
-- torch::nn::functional::softshrink
-- torch::nn::functional::softsign
-- torch::nn::functional::tanhshrink
@[extern "lean_torch_tensor_tanh"] opaque tanh {s : Shape} (t : @& T s) : T s

-- Matrix multiplication (generic shape-aware version)
@[extern "lean_torch_matmul"] private opaque matmul_impl {s1 s2 : Shape} (a : @& T s1) (b : @& T s2) : T #[]

/-- Generic matrix multiplication with shape inference following PyTorch broadcasting rules.
    - 1D @ 1D: dot product -> scalar
    - 2D @ 2D: [m,k] @ [k,n] -> [m,n]
    - 1D @ 2D: [k] @ [k,n] -> [n]
    - 2D @ 1D: [m,k] @ [k] -> [m]
    - ND @ ND: broadcast batch dims, matmul last 2 dims -/
def matmul {s1 s2 : Shape} (a : T s1) (b : T s2) : T (matmulShape s1 s2) :=
  reshape (matmul_impl a b) (matmulShape s1 s2)

-- Specialized matmul for common cases (more efficient, no reshape needed)
@[extern "lean_torch_bmm"] opaque bmm {b m n k : UInt64} (input : @& T #[b, m, k]) (mat2 : @& T #[b, k, n]) : T #[b, m, n]
@[extern "lean_torch_mm"] opaque mm {m n k : UInt64} (input : @& T #[m, k]) (mat2 : @& T #[k, n]) : T #[m, n]

/-- 4D batched matmul for multi-head attention: [b,h,m,k] @ [b,h,k,n] -> [b,h,m,n] -/
@[extern "lean_torch_bmm4d"]
opaque bmm4d_impl {b h m k n : UInt64}
    (input : @& T #[b, h, m, k]) (mat2 : @& T #[b, h, k, n]) : T #[b, h, m, n]

@[extern "lean_torch_transpose"] opaque transpose {s : Shape} (input : @& T s) (dim0 : UInt64) (dim1 : UInt64) : T (transposeShape s dim0.toNat dim1.toNat)

/-- Shape-aware matmul for 3D @ 2D: [batch, seq, k] @ [k, n] -> [batch, seq, n] -/
@[extern "lean_torch_matmul3d_2d"]
opaque matmul3d {batch seq k n : UInt64}
    (a : @& T #[batch, seq, k]) (b : @& T #[k, n]) : T #[batch, seq, n]

/-- Shape-aware matmul for 2D tensors: [m, k] @ [k, n] -> [m, n] -/
def matmul2d {m k n : UInt64}
    (a : T #[m, k]) (b : T #[k, n]) : T #[m, n] :=
  mm a b

/-- 4D batched matmul for attention: [b, h, m, k] @ [b, h, k, n] -> [b, h, m, n]
    Uses typed FFI to avoid shape erasure -/
def bmm4d {b h m k n : UInt64}
    (a : T #[b, h, m, k]) (x : T #[b, h, k, n]) : T #[b, h, m, n] :=
  bmm4d_impl a x

/-- Transpose last two dimensions of a 2D tensor -/
@[extern "lean_torch_transpose_2d"]
opaque transpose2d {m n : UInt64} (input : @& T #[m, n]) : T #[n, m]

/-- Transpose for 3D tensors: swap dims 1 and 2 -/
@[extern "lean_torch_transpose3d_12"]
opaque transpose3d_12 {a b c : UInt64} (input : @& T #[a, b, c]) : T #[a, c, b]

/-- Reshape 3D to 4D: [batch, seq, n_head * head_dim] -> [batch, seq, n_head, head_dim] -/
def reshape_to_heads {batch seq n_head head_dim : UInt64}
    (x : T #[batch, seq, n_head * head_dim]) : T #[batch, seq, n_head, head_dim] :=
  reshape x #[batch, seq, n_head, head_dim]

/-- Reshape 4D to 3D: [batch, seq, n_head, head_dim] -> [batch, seq, n_head * head_dim] -/
def reshape_from_heads {batch seq n_head head_dim : UInt64}
    (x : T #[batch, seq, n_head, head_dim]) : T #[batch, seq, n_head * head_dim] :=
  reshape x #[batch, seq, n_head * head_dim]

/-- Transpose [batch, seq, n_head, head_dim] -> [batch, n_head, seq, head_dim] -/
@[extern "lean_torch_transpose_for_attention"]
opaque transpose_for_attention {batch seq n_head head_dim : UInt64}
    (x : @& T #[batch, seq, n_head, head_dim]) : T #[batch, n_head, seq, head_dim]

/-- Transpose [batch, n_head, seq, head_dim] -> [batch, seq, n_head, head_dim] -/
@[extern "lean_torch_transpose_from_attention"]
opaque transpose_from_attention {batch n_head seq head_dim : UInt64}
    (x : @& T #[batch, n_head, seq, head_dim]) : T #[batch, seq, n_head, head_dim]

@[extern "lean_torch_softmax_dim"] opaque softmax_dim {s : Shape} (input : @& T s) (dim : Int64) : T s
@[extern "lean_torch_sqrt"] opaque sqrt {s : Shape} (input : @& T s) : T s
@[extern "lean_torch_div"] opaque div {s : Shape} (input : @& T s) (other : @& T s) : T s
@[extern "lean_torch_pow"] opaque pow {s : Shape} (input : @& T s) (exponent : Float) : T s
@[extern "lean_torch_unsqueeze"] opaque unsqueeze {s : Shape} (input : @& T s) (dim : Nat) : T (unsqueezeShape s dim)
@[extern "lean_torch_squeeze"] opaque squeeze {s : Shape} (input : @& T s) (dim : Nat) : T (squeezeShape s dim)
@[extern "lean_torch_masked_fill"] opaque masked_fill {s : Shape} (input : @& T s) (mask : @& T s) (value : Float) : T s

/-- Select flattened values where mask is true (PyTorch `masked_select`). -/
@[extern "lean_torch_masked_select"]
opaque masked_select {s : Shape} (input : @& T s) (mask : @& T s) : T #[]

/-- Scatter source values into `input` at mask=true positions (PyTorch `masked_scatter`). -/
@[extern "lean_torch_masked_scatter"]
opaque masked_scatter {s src : Shape} (input : @& T s) (mask : @& T s) (source : @& T src) : T s
@[extern "lean_torch_expand"] opaque expand' {s : Shape} (input : @& T s) (size : Array UInt64) : T s
@[extern "lean_torch_repeat"] opaque tensor_repeat {s : Shape} (input : @& T s) (repeats : Array UInt64) : T s

/-- Concatenate tensors along a dimension (untyped FFI, requires same-typed array) -/
@[extern "lean_torch_cat"] opaque cat_impl {s : Shape} (tensors : @& Array (T s)) (dim : Int) : T #[]

/-- Direct 2-tensor concatenation (no intermediate reshapes) -/
@[extern "lean_torch_cat2"] private opaque cat2_impl {s1 s2 : Shape}
    (t1 : @& T s1) (t2 : @& T s2) (dim : Int) : T #[]

/-- Erase static shape information while preserving runtime tensor shape. -/
@[extern "lean_torch_erase_shape"]
opaque eraseShape {s : Shape} (t : @& T s) : T #[]

/-- Concatenate shape-erased tensors along a dimension (runtime-checked). -/
@[extern "lean_torch_cat_dyn"]
opaque cat_dyn (tensors : @& Array (T #[])) (dim : Int) : T #[]

/-- Compute the output shape when concatenating two shapes along dimension `dim`.
    The shapes must match in all dimensions except `dim`, where they are summed. -/
def catShape (s1 s2 : Shape) (dim : Nat) : Shape :=
  s1.set! dim (s1.getD dim 0 + s2.getD dim 0)

/-- Type-safe concatenation along any dimension.
    Given tensors of shapes s1 and s2 that match in all dimensions except `dim`,
    produces a tensor where dimension `dim` is the sum of the input dimensions.

    Example: cat #[2,3] #[2,5] 1 produces shape #[2, 8] -/
def cat {s1 s2 : Shape} (t1 : T s1) (t2 : T s2) (dim : Nat) : T (catShape s1 s2 dim) :=
  reshape (cat2_impl t1 t2 dim) (catShape s1 s2 dim)

/-- Expand tensor to target shape (typed version) -/
def expand {s : Shape} (input : T s) (targetShape : Shape) : T targetShape :=
  let result := expand' input targetShape
  reshape result targetShape

-- torch::nn::functional::threshold
-- torch::nn::functional::triplet_margin_loss
-- torch::nn::functional::triplet_margin_with_distance_loss
-- torch::nn::functional::unfold

-- Attention mechanism (with causal masking support)
@[extern "lean_torch_scaled_dot_product_attention"]
opaque scaled_dot_product_attention' {s : Shape}
  (query : @& T s) (key : @& T s) (value : @& T s)
  (attn_mask : Option (T s) := none)
  (dropout_p : Float := 0.0)
  (is_causal : Bool := false) : T s

/-- Shape-aware scaled dot-product attention for GPT
    Q, K, V: [batch, n_head, seq, head_dim] -> output: [batch, n_head, seq, head_dim] -/
@[extern "lean_torch_sdpa_4d"]
opaque scaled_dot_product_attention {batch n_head seq head_dim : UInt64}
    (query : @& T #[batch, n_head, seq, head_dim])
    (key : @& T #[batch, n_head, seq, head_dim])
    (value : @& T #[batch, n_head, seq, head_dim])
    (dropout_p : Float := 0.0)
    (is_causal : Bool := true) : T #[batch, n_head, seq, head_dim]

/-- Fused scaled dot-product attention with an explicit additive float bias.
    Q, K, V: [batch, n_head, seq, head_dim], bias: [batch, n_head, seq, seq]
    -> output: [batch, n_head, seq, head_dim]. Replaces a manual
    matmul/bias/masked_fill/softmax pipeline with one fused kernel. -/
@[extern "lean_torch_sdpa_4d_bias"]
opaque scaled_dot_product_attention_bias {batch n_head seq head_dim : UInt64}
    (query : @& T #[batch, n_head, seq, head_dim])
    (key : @& T #[batch, n_head, seq, head_dim])
    (value : @& T #[batch, n_head, seq, head_dim])
    (bias : @& T #[batch, n_head, seq, seq])
    (dropout_p : Float := 0.0)
    (is_causal : Bool := false) : T #[batch, n_head, seq, head_dim]

-- Lower triangular (for manual causal masking)
@[extern "lean_torch_tril"] opaque tril {s : Shape} (t : @& T s) (diagonal : Int := 0) : T s

-- Reductions (internal, uses Option for dim)
@[extern "lean_torch_sum"] private opaque sum_impl {s : Shape} (t : @& T s) (dim : Option (Array UInt64)) (keepdim : Bool) : T #[]
@[extern "lean_torch_mean"] private opaque mean_impl {s : Shape} (t : @& T s) (dim : Option (Array UInt64)) (keepdim : Bool) : T #[]

-- Element-wise absolute value
@[extern "lean_torch_abs"] opaque abs {s : Shape} (t : @& T s) : T s

-- Reductions to scalar (all elements)
def sumAll {s : Shape} (t : T s) : T #[] := sum_impl t none false
def meanAll {s : Shape} (t : T s) : T #[] := mean_impl t none false

-- Max/Min reductions
-- The shape argument was never read by the native side; the alias is
-- shape-erased, so `T #[]` at the boundary accepts any input.
@[extern "lean_torch_max_all"] private opaque max_all_impl (t : @& T #[]) : T #[]
@[extern "lean_torch_min_all"] private opaque min_all_impl (t : @& T #[]) : T #[]

def maxAll {s : Shape} (t : T s) : T #[] := max_all_impl t
def minAll {s : Shape} (t : T s) : T #[] := min_all_impl t

-- Reductions along a dimension (shape-aware versions)
-- Note: These use reshape internally since C++ doesn't track shapes
def sumDim {s : Shape} (t : T s) (dim : Nat) (keepdim : Bool := false) : T (reduceShape s dim keepdim) :=
  let result := sum_impl t (some #[dim.toUInt64]) keepdim
  reshape result (reduceShape s dim keepdim)

def meanDim {s : Shape} (t : T s) (dim : Nat) (keepdim : Bool := false) : T (reduceShape s dim keepdim) :=
  let result := mean_impl t (some #[dim.toUInt64]) keepdim
  reshape result (reduceShape s dim keepdim)

/-- Cumulative sum along a dimension, preserving shape. -/
@[extern "lean_torch_cumsum"]
opaque cumsum {s : Shape} (input : @& T s) (dim : Int64) : T s

-- Sampling operations
@[extern "lean_torch_topk_values"] opaque topk_values {s : Shape} (t : @& T s) (k : UInt64) (dim : UInt64) : T (replaceAtDim s dim.toNat k)
@[extern "lean_torch_multinomial"] opaque multinomial {s : Shape} (probs : @& T s) (num_samples : UInt64) (replacement : Bool := false) : IO (T (replaceAtDim s (s.size - 1) num_samples))

/-- Top-k filtering: set all logits outside the top k to -infinity.
    Used for top-k sampling in text generation. -/
@[extern "lean_torch_topk_filter"]
opaque topKFilter {s : Shape} (logits : @& T s) (k : UInt64) : T s

/-- Top-p (nucleus) filtering: set logits outside cumulative probability p to -infinity.
    Used for nucleus sampling in text generation. -/
@[extern "lean_torch_topp_filter"]
opaque topPFilter {s : Shape} (logits : @& T s) (p : Float) : T s

/-- Squeeze tensor along a specific dimension (supports negative indices). -/
@[extern "lean_torch_squeeze_dim"]
opaque squeezeDim {s : Shape} (input : @& T s) (dim : Int64) : T #[]

-- Argmax (returns indices along dimension, removing that dimension)
@[extern "lean_torch_argmax"] opaque argmax {s : Shape} (t : @& T s) (dim : UInt64) : T (reduceShape s dim.toNat false)

-- Scalar extraction
@[extern "lean_torch_item"] opaque item {s : Shape} (t : @& T s) : Float
@[extern "lean_torch_item_int"] opaque itemInt {s : Shape} (t : @& T s) : Int64

-- Gradient clipping
@[extern "lean_torch_clip_grad_norm_"] opaque clip_grad_norm_ {s : Shape} (param : @& T s) (max_norm : Float) : IO Float

/-- Clip gradient values element-wise to [-clip_value, clip_value].
    Modifies gradients in-place. -/
@[extern "lean_torch_clip_grad_value_"]
opaque clip_grad_value_ {s : Shape} (param : @& T s) (clip_value : Float) : IO Unit

end nn

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

namespace signal

/-- Hann window for STFT and spectral analysis. -/
@[extern "lean_torch_hann_window"]
opaque hannWindow (n : UInt64) : T #[n]

/-- 1D STFT returning real/imag last-dimension packing.
    Output shape is dynamic and typically `[n_fft/2+1, frames, 2]`. -/
@[extern "lean_torch_stft_1d"]
opaque stft1d {n : UInt64}
    (input : @& T #[n])
    (n_fft : UInt64)
    (hop_length : UInt64)
    (win_length : UInt64)
    (window : @& T #[win_length])
    (center : Bool := true)
    (normalized : Bool := false)
    : T #[]

/-- 1D RFFT returning real/imag packed output with dynamic shape. -/
@[extern "lean_torch_rfft_1d"]
opaque rfft1d {n : UInt64} (input : @& T #[n]) : T #[]

/-- 1D inverse STFT from packed real/imag input.
    Input should follow `stft1d` packing with last dimension 2. -/
@[extern "lean_torch_istft_1d"]
opaque istft1d
    (input : @& T #[])
    (n_fft : UInt64)
    (hop_length : UInt64)
    (win_length : UInt64)
    (window : @& T #[win_length])
    (center : Bool := true)
    (normalized : Bool := false)
    (length : UInt64 := 0)
    : T #[]

end signal

-- SafeTensors loading utilities (alias for data.loadTensor with name-based loading)
namespace safetensors

/-- Opaque handle to a parsed SafeTensors file with open file handle. -/
opaque SafeTensorsHandle : Type

/-- Open a SafeTensors file, parsing its header once. -/
@[extern "lean_torch_safetensors_open"]
opaque openHandle (path : @& String) : IO SafeTensorsHandle

/-- Load a tensor from an open SafeTensors handle by name and shape. -/
@[extern "lean_torch_safetensors_load_from_handle"]
opaque loadFromHandle (handle : @& SafeTensorsHandle) (name : @& String) (shape : Shape) : IO (T shape)

/-- Load a tensor from SafeTensors handle and move it to `device`. -/
def loadFromHandleOnDevice (handle : @& SafeTensorsHandle) (name : String) (s : Shape) (device : Device := Device.CPU) : IO (T s) := do
  let t ← loadFromHandle handle name s
  pure (t.to device)

/-- Load a tensor from a SafeTensors file by name.
    path: path to the .safetensors file
    name: tensor name in the file
    s: expected tensor shape -/
@[extern "lean_torch_safetensors_load"]
opaque loadTensor (path : @& String) (name : @& String) (s : Shape) : IO (T s)

/-- Load a tensor from SafeTensors and move it to `device`. -/
def loadTensorOnDevice (path : String) (name : String) (s : Shape) (device : Device := Device.CPU) : IO (T s) := do
  let t ← loadTensor path name s
  pure (t.to device)

/-- Save one tensor to a SafeTensors file under `name`.
    Current runtime implementation writes a single-tensor file. -/
@[extern "lean_torch_safetensors_save"]
opaque saveTensor {s : Shape} (path : @& String) (name : @& String) (t : @& T s) : IO Unit

/-- Low-level multi-tensor save primitive.
    `names[i]` and `tensors[i]` form one named tensor entry.
    `metaKeys`/`metaVals` are optional `__metadata__` key/value pairs. -/
@[extern "lean_torch_safetensors_save_many"]
opaque saveTensorsRaw
    (path : @& String)
    (names : @& Array String)
    (tensors : @& Array (T #[]))
    (metaKeys : @& Array String)
    (metaVals : @& Array String)
    : IO Unit

/-- Save multiple tensors to one SafeTensors file.
    Names must be unique; metadata is optional. -/
def saveTensors
    (path : String)
    (entries : Array (String × T #[]))
    (metadata : Array (String × String) := #[])
    : IO Unit := do
  let names := entries.map Prod.fst
  let tensors := entries.map Prod.snd
  let metaKeys := metadata.map Prod.fst
  let metaVals := metadata.map Prod.snd
  saveTensorsRaw path names tensors metaKeys metaVals

/-- Load a tensor from sharded SafeTensors files.
    dir: directory containing sharded .safetensors files
    name: tensor name to search for
    s: expected tensor shape -/
@[extern "lean_torch_safetensors_load_sharded"]
opaque loadTensorSharded (dir : @& String) (name : @& String) (s : Shape) : IO (T s)

/-- Load a tensor from sharded SafeTensors and move it to `device`. -/
def loadTensorShardedOnDevice (dir : String) (name : String) (s : Shape) (device : Device := Device.CPU) : IO (T s) := do
  let t ← loadTensorSharded dir name s
  pure (t.to device)

end safetensors



@[extern "lean_torch_get_live_tensors"]
opaque get_live_tensors : IO UInt64

/-- Set global RNG seed for deterministic runs (CPU + CUDA generators). -/
@[extern "lean_torch_manual_seed"]
opaque manualSeed (seed : UInt64) : IO Unit



namespace nn

/-- Scaled dot-product attention with Group-Query Attention (GQA) support.
    Q: [batch, n_head, seq, head_dim]
    K, V: [batch, n_kv_head, seq, head_dim]
    When enable_gqa=true, each KV head serves a group of query heads. Fused
    backends consume grouped K/V directly; portable fallback avoids materialized repeats. -/
@[extern "lean_torch_sdpa_gqa"]
opaque scaledDotProductAttentionGQA
    {batch n_head n_kv_head seq head_dim : UInt64}
    (query : @& T #[batch, n_head, seq, head_dim])
    (key : @& T #[batch, n_kv_head, seq, head_dim])
    (value : @& T #[batch, n_kv_head, seq, head_dim])
    (dropout_p : Float := 0.0)
    (is_causal : Bool := true)
    (enable_gqa : Bool := false)
    : T #[batch, n_head, seq, head_dim]

/-- Scaled dot-product attention with GQA where query and KV sequence lengths may differ.
    Q: [batch, n_head, q_seq, head_dim]
    K, V: [batch, n_kv_head, kv_seq, head_dim]
    Causality uses upper-left alignment: query i may attend to keys j ≤ i.
    For KV-cache decoding with q_seq=1, pass is_causal=false to attend to all cached keys. -/
@[extern "lean_torch_sdpa_gqa_qkv"]
opaque scaledDotProductAttentionGQAQKV
    {batch n_head n_kv_head q_seq kv_seq head_dim : UInt64}
    (query : @& T #[batch, n_head, q_seq, head_dim])
    (key : @& T #[batch, n_kv_head, kv_seq, head_dim])
    (value : @& T #[batch, n_kv_head, kv_seq, head_dim])
    (dropout_p : Float := 0.0)
    (is_causal : Bool := true)
    (enable_gqa : Bool := false)
    : T #[batch, n_head, q_seq, head_dim]

/-- Scaled dot-product attention with GQA and sliding window support.
    Q: [batch, n_head, seq, head_dim]
    K, V: [batch, n_kv_head, seq, head_dim]
    window_size: number of positions each query can attend to (sliding window).
    When enable_gqa=true, each KV head serves a group of query heads. Fused
    backends consume grouped K/V directly; portable fallback avoids materialized repeats.
    This binding is always causal (including when is_causal=false): each position
    attends only to the previous window_size positions, including itself. A zero
    window masks every key. Query chunks and reachable KV slices bound temporary
    fallback work; autograd may retain intermediates from all chunks for backward. -/
@[extern "lean_torch_sdpa_gqa_window"]
opaque scaledDotProductAttentionGQAWindow
    {batch n_head n_kv_head seq head_dim : UInt64}
    (query : @& T #[batch, n_head, seq, head_dim])
    (key : @& T #[batch, n_kv_head, seq, head_dim])
    (value : @& T #[batch, n_kv_head, seq, head_dim])
    (dropout_p : Float := 0.0)
    (is_causal : Bool := true)
    (enable_gqa : Bool := false)
    (window_size : UInt64)
    : T #[batch, n_head, seq, head_dim]

/-- Scaled dot-product attention with GQA and attention mask support.
    Q: [batch, n_head, seq, head_dim]
    K, V: [batch, n_kv_head, seq, head_dim]
    attn_mask: [batch, seq] - padding mask (1 for valid, 0 for padding)
    When enable_gqa=true, each KV head serves a group of query heads. Fused
    backends consume grouped K/V directly; portable fallback avoids materialized repeats. -/
@[extern "lean_torch_sdpa_gqa_mask"]
opaque scaledDotProductAttentionGQAMask
    {batch n_head n_kv_head seq head_dim : UInt64}
    (query : @& T #[batch, n_head, seq, head_dim])
    (key : @& T #[batch, n_kv_head, seq, head_dim])
    (value : @& T #[batch, n_kv_head, seq, head_dim])
    (attn_mask : @& T #[batch, seq])
    (dropout_p : Float := 0.0)
    (is_causal : Bool := true)
    (enable_gqa : Bool := false)
    : T #[batch, n_head, seq, head_dim]

/-- Scaled dot-product attention with GQA and an explicit query/key mask.
    Q: [batch, n_head, q_seq, head_dim]
    K, V: [batch, n_kv_head, kv_seq, head_dim]
    attn_mask: [batch, q_seq, kv_seq] with 1 for allowed attention edges and 0 for masked edges.
    When enable_gqa=true, each KV head serves a group of query heads. Fused
    backends consume grouped K/V directly; portable fallback avoids materialized repeats. -/
@[extern "lean_torch_sdpa_gqa_mask_qkv"]
opaque scaledDotProductAttentionGQAMaskQKV
    {batch n_head n_kv_head q_seq kv_seq head_dim : UInt64}
    (query : @& T #[batch, n_head, q_seq, head_dim])
    (key : @& T #[batch, n_kv_head, kv_seq, head_dim])
    (value : @& T #[batch, n_kv_head, kv_seq, head_dim])
    (attn_mask : @& T #[batch, q_seq, kv_seq])
    (dropout_p : Float := 0.0)
    (enable_gqa : Bool := false)
    : T #[batch, n_head, q_seq, head_dim]

/-- Experimental Tyr FlashAttention entry point.

    This calls the C++ `tyr::flash_attn` operator, which currently dispatches
    to native ThunderKittens-backed H100 kernels for the validated fixed shapes
    and falls back to PyTorch SDPA for everything else. -/
@[extern "lean_torch_tyr_flash_attn_4d"]
opaque tyrFlashAttn4d
    {batch n_head n_kv_head q_seq kv_seq head_dim : UInt64}
    (query : @& T #[batch, n_head, q_seq, head_dim])
    (key : @& T #[batch, n_kv_head, kv_seq, head_dim])
    (value : @& T #[batch, n_kv_head, kv_seq, head_dim])
    (attn_mask : Option (T #[batch, kv_seq]) := none)
    (dropout_p : Float := 0.0)
    (is_causal : Bool := false)
    (scale : Option Float := none)
    (enable_gqa : Bool := false)
    : T #[batch, n_head, q_seq, head_dim]

/-- RMSNorm without learnable parameters: x / sqrt(mean(x^2) + eps)
    Normalizes over the last dimension. -/
def rmsNorm {s : Shape} (x : T s) (eps : Float := 1e-6) : T s :=
  let xf := toFloat' x
  let var := meanDim (xf * xf) (s.size - 1) true
  let inv := rsqrt (var + eps)
  xf * inv

/-- RMSNorm with learnable weight: (x / sqrt(mean(x^2) + eps)) * weight
    Normalizes over the last dimension, then scales by weight.
    Weight broadcasts over the last dimension. -/
def rmsNormWeighted {s : Shape} {w : Shape}
    (x : T s) (weight : T w) (eps : Float := 1e-6) : T s :=
  let normed := rmsNorm x eps
  normed * weight

/-- ReLU squared activation: relu(x)^2 -/
def reluSquared {s : Shape} (x : T s) : T s :=
  let r := relu x
  r * r

/-- Logit softcap: cap * tanh(x / cap)
    Prevents logits from growing too large. -/
@[extern "lean_torch_softcap"]
opaque softcap {s : Shape} (input : @& T s) (cap : Float := 15.0) : T s

def relu {s : Shape} (x : T s) := torch.relu x
def relu6 {s : Shape} (x : T s) := torch.relu6 x
def rsqrt {s : Shape} (x : T s) := torch.rsqrt x
def toFloat' {s : Shape} (x : T s) := torch.toFloat' x

end nn

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

-- ============================================================================
-- High-level loss functions (defined after nn namespace for access to helpers)
-- ============================================================================

/-- Focal loss for class-imbalanced classification (simplified version).
    Uses cross-entropy loss with focal weighting.
    Focal loss = -alpha * (1 - pt)^gamma * log(pt)
    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)

    Note: This is a convenience wrapper. For precise shape tracking, use
    nn.cross_entropy (or nn.cross_entropy_none) directly and implement focal weighting manually. -/
def focal_loss {batch num_classes : UInt64}
    (logits : T #[batch, num_classes])
    (targets : T #[batch])
    (alpha : Float := 0.25)
    (_gamma : Float := 2.0)
    : T #[] :=
  -- Standard cross-entropy: -log(softmax(logits)[target])
  let ce_loss := nn.cross_entropy logits targets
  -- For focal loss, we'd ideally weight by (1-pt)^gamma
  -- This simplified version applies focal scaling to the mean loss
  -- For full focal loss, use nll_loss with reduction="none" and weight manually
  mul_scalar ce_loss alpha

/-- Triplet margin loss for metric learning (simplified version).
    L = mean(max(0, ||anchor - positive|| - ||anchor - negative|| + margin))
    Reference: Schroff et al., "FaceNet" (2015)

    Note: This computes element-wise differences and uses meanAll.
    For batch-aware distance computation, consider using einsum. -/
def triplet_margin_loss {batch dim : UInt64}
    (anchor : T #[batch, dim])
    (positive : T #[batch, dim])
    (negative : T #[batch, dim])
    (margin : Float := 1.0)
    : T #[] :=
  -- Compute element-wise squared differences
  let diff_pos := anchor - positive  -- [batch, dim]
  let diff_neg := anchor - negative  -- [batch, dim]
  -- Sum of squares gives squared L2 norm
  let sq_pos := diff_pos * diff_pos  -- [batch, dim]
  let sq_neg := diff_neg * diff_neg  -- [batch, dim]
  -- Use meanAll on differences to approximate the triplet objective
  -- Full implementation would need per-sample L2 norms
  let mean_sq_pos := nn.meanAll sq_pos
  let mean_sq_neg := nn.meanAll sq_neg
  -- sqrt of mean squared differences (approximates mean distance)
  let dist_pos := nn.sqrt mean_sq_pos
  let dist_neg := nn.sqrt mean_sq_neg
  -- max(0, dist_pos - dist_neg + margin)
  let raw := nn.item dist_pos - nn.item dist_neg + margin
  if raw > 0.0 then full #[] raw else full #[] 0.0

-- ============================================================================
-- Linear Algebra operations for manifold optimization
-- ============================================================================



end torch
