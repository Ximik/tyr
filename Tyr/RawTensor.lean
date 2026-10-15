import Tyr.Basic

/-! Native libtorch ABI. These handles carry no static guarantees.
Model code should use `torch.Tensor`; only typed adapters call this module. -/
namespace torch.RawTensor

/-- Nontrainable constant factories with recoverable native failures. -/
@[extern "lean_torch_zeros_checked"] opaque zerosChecked (shape : Shape)
    (device : Device := .CPU) : Except String RawTensor
@[extern "lean_torch_ones_checked"] opaque onesChecked (shape : Shape)
    (device : Device := .CPU) : Except String RawTensor

@[extern "lean_torch_ones"] opaque ones (shape : Shape) (requiresGrad : Bool := false) (device : Device := .CPU) (noGrad : requiresGrad = false := by decide) : RawTensor

@[extern "lean_torch_randn"] opaque randn (shape : Shape) (requiresGrad : Bool := false) (device : Device := .CPU) : IO RawTensor

@[extern "lean_torch_full"] opaque full (shape : Shape) (value : Float) (requiresGrad : Bool := false) (device : Device := .CPU) (noGrad : requiresGrad = false := by decide) : RawTensor

@[extern "lean_torch_full_int"] opaque fullInt (shape : Shape) (value : Int64) : RawTensor

@[extern "lean_torch_zeros_like"] opaque zerosLike (shape : Shape) (t : @& RawTensor) : RawTensor

@[extern "lean_torch_ones_like"] opaque onesLike (shape : Shape) (t : @& RawTensor) : RawTensor

@[extern "lean_torch_to_float"] opaque toFloat32 (shape : Shape) (t : @& RawTensor) : RawTensor

@[extern "lean_torch_to_bfloat16"] opaque toBFloat16 (shape : Shape) (t : @& RawTensor) : RawTensor

@[extern "lean_torch_to_long"] opaque toInt64 (shape : Shape) (t : @& RawTensor) : RawTensor

@[extern "lean_torch_to"] opaque toDevice (shape : Shape) (t : @& RawTensor) (device : Device) : RawTensor

/-- Convert dtype and device together, with native failures returned through IO.
`none` retains the source dtype; a supplied tag must name a supported dtype.
The source shape is checked before conversion and never reshaped. -/
@[extern "lean_torch_convert_tensor"] opaque convertIO (shape : Shape)
    (t : @& RawTensor) (device : @& Device) (dtype : @& Option String) : IO RawTensor

@[extern "lean_torch_requires_grad"] opaque requiresGrad (shape : Shape) (t : @& RawTensor) : Bool

@[extern "lean_torch_l2_norm"] opaque l2Norm (n : UInt64) (t : @& RawTensor) : Float

@[extern "lean_torch_item"] opaque item (shape : Shape) (t : @& RawTensor) : Float

@[extern "lean_torch_item_checked"] opaque itemChecked (shape : Shape)
    (t : @& RawTensor) : Except String Float

@[extern "lean_torch_gt"] opaque gt (shape : Shape) (a b : @& RawTensor) : RawTensor

@[extern "lean_torch_where"] opaque whereSelect (shape : Shape) (condition a b : @& RawTensor) : RawTensor

@[extern "lean_torch_tensor_relu"] opaque relu (shape : Shape) (t : @& RawTensor) : RawTensor

@[extern "lean_torch_abs"] opaque abs (shape : Shape) (t : @& RawTensor) : RawTensor

@[extern "lean_torch_tensor_exp"] opaque exp (shape : Shape) (t : @& RawTensor) : RawTensor

@[extern "lean_torch_sqrt"] opaque sqrt (shape : Shape) (t : @& RawTensor) : RawTensor

@[extern "lean_torch_rsqrt"] opaque rsqrt (shape : Shape) (t : @& RawTensor) : RawTensor

@[extern "lean_torch_tensor_sigmoid"] opaque sigmoid (shape : Shape) (t : @& RawTensor) : RawTensor

@[extern "lean_torch_tensor_tanh"] opaque tanh (shape : Shape) (t : @& RawTensor) : RawTensor

@[extern "lean_torch_tensor_gelu"] opaque gelu (shape : Shape) (t : @& RawTensor) : RawTensor

@[extern "lean_torch_tensor_silu"] opaque silu (shape : Shape) (t : @& RawTensor) : RawTensor

@[extern "lean_torch_transpose_2d"] opaque transpose2d (m n : UInt64) (t : @& RawTensor) : RawTensor

@[extern "lean_torch_detach"] opaque detach (shape : Shape) (t : @& RawTensor) : RawTensor

/-- Read an independent, detached snapshot of the accumulated gradient.
    An undefined gradient is represented by a zero tensor. -/
@[extern "lean_torch_grad_of"] opaque gradOf (shape : Shape) (t : @& RawTensor) : IO RawTensor

/-- Clear the tensor's accumulated gradient in place. -/
@[extern "lean_torch_zero_grad"] opaque zeroGrad (shape : Shape) (t : @& RawTensor) : IO Unit

@[extern "lean_torch_tensor_add"] opaque add (shape : Shape) (a b : @& RawTensor) : RawTensor

@[extern "lean_torch_tensor_sub"] opaque sub (shape : Shape) (a b : @& RawTensor) : RawTensor

@[extern "lean_torch_tensor_mul"] opaque mul (shape : Shape) (a b : @& RawTensor) : RawTensor

@[extern "lean_torch_div"] opaque div (shape : Shape) (a b : @& RawTensor) : RawTensor

@[extern "lean_torch_matmul"] opaque matmul (s1 s2 : Shape) (a b : @& RawTensor) : RawTensor

@[extern "lean_torch_mm"] opaque mm (m n k : UInt64) (a b : @& RawTensor) : RawTensor

@[extern "lean_torch_linear"] opaque linear (m n b : UInt64) (a b : @& RawTensor) : RawTensor

@[extern "lean_torch_linear3d"] opaque linear3d (batch seq inDim outDim : UInt64) (a b : @& RawTensor) : RawTensor

@[extern "lean_torch_matmul3d_2d"] opaque matmul3d (batch seq k n : UInt64) (a b : @& RawTensor) : RawTensor

@[extern "lean_torch_bmm4d"] opaque bmm4d (b h m k n : UInt64) (a b : @& RawTensor) : RawTensor

@[extern "lean_torch_add_scalar"] opaque addScalar (shape : Shape) (t : @& RawTensor) (value : Float) : RawTensor

@[extern "lean_torch_sub_scalar"] opaque subScalar (shape : Shape) (t : @& RawTensor) (value : Float) : RawTensor

@[extern "lean_torch_mul_scalar"] opaque mulScalar (shape : Shape) (t : @& RawTensor) (value : Float) : RawTensor

@[extern "lean_torch_div_scalar"] opaque divScalar (shape : Shape) (t : @& RawTensor) (value : Float) : RawTensor

@[extern "lean_torch_pow"] opaque pow (shape : Shape) (t : @& RawTensor) (exponent : Float) : RawTensor

@[extern "lean_torch_tensor_softmax"] opaque softmax (shape : Shape) (t : @& RawTensor) (dim : Int32 := -1) : RawTensor

@[extern "lean_torch_expand"] opaque expand (shape : Shape) (t : @& RawTensor) (target : Array UInt64) : RawTensor

@[extern "lean_torch_reshape_exact"] opaque reshape (shape : Shape) (t : @& RawTensor) (target : @& Shape) : RawTensor

@[extern "lean_torch_permute"] opaque permute (shape : Shape) (t : @& RawTensor) (permutation : Array UInt64) : RawTensor

@[extern "lean_torch_transpose"] opaque transpose (shape : Shape) (t : @& RawTensor)
    (dim0 dim1 : UInt64) : Except String RawTensor

@[extern "lean_torch_unsqueeze"] opaque unsqueeze (shape : Shape) (t : @& RawTensor) (dim : Nat) : RawTensor

@[extern "lean_torch_squeeze"] opaque squeeze (shape : Shape) (t : @& RawTensor) (dim : Nat) : RawTensor

@[extern "lean_torch_cat2"] opaque cat (s1 s2 : Shape) (a b : @& RawTensor) (dim : Int) : RawTensor

@[extern "lean_torch_sum"] opaque sum (shape : Shape) (t : @& RawTensor) (dims : Option (Array UInt64)) (keepdim : Bool) : RawTensor

@[extern "lean_torch_mean"] opaque mean (shape : Shape) (t : @& RawTensor) (dims : Option (Array UInt64)) (keepdim : Bool) : RawTensor

/-- Return a tensor with the requested autograd flag, preserving other aliases. -/
@[extern "lean_torch_set_requires_grad"] opaque setRequiresGrad (shape : Shape) (t : RawTensor) (requiresGrad : Bool) : IO RawTensor

@[extern "lean_torch_affine3d"] opaque affine3d (batch seq inDim outDim : UInt64) (x weight bias : @& RawTensor) : RawTensor
@[extern "lean_torch_affine3d_checked"] opaque affine3dChecked (batch seq inDim outDim : UInt64)
    (x weight bias : @& RawTensor) : Except String RawTensor


def sumAll (shape : Shape) (t : RawTensor) : RawTensor := sum shape t none false
def meanAll (shape : Shape) (t : RawTensor) : RawTensor := mean shape t none false

/-- Match the existing FP32 RMS normalization and weight promotion semantics. -/
def rmsNormWeighted (shape _weightShape : Shape) (x weight : RawTensor) (eps : Float := 1e-6) : RawTensor :=
  let xf := toFloat32 shape x
  let variance := mean shape (mul shape xf xf) (some #[(shape.size - 1).toUInt64]) true
  let reduced := reduceShape shape (shape.size - 1) true
  mul shape (mul shape xf (rsqrt reduced (addScalar reduced variance eps))) weight

@[extern "lean_torch_dropout"] opaque dropout (shape : Shape)
    (t : @& RawTensor) (p : Float) (training : Bool) : IO RawTensor
@[extern "lean_torch_layer_norm_3d"] opaque layerNorm (batch seq n : UInt64)
    (x weight bias : @& RawTensor) (eps : Float := 1e-5) : RawTensor
@[extern "lean_torch_transpose_for_attention"] opaque splitHeads (batch seq heads dim : UInt64)
    (x : @& RawTensor) : RawTensor
@[extern "lean_torch_transpose_from_attention"] opaque joinHeads (batch heads seq dim : UInt64)
    (x : @& RawTensor) : RawTensor
@[extern "lean_torch_sdpa_4d"] opaque attention (batch heads seq dim : UInt64)
    (q k v : @& RawTensor) (causal : Bool := true) : Except String RawTensor
@[extern "lean_torch_sdpa_4d_io"] opaque attentionIO (batch heads seq dim : UInt64)
    (q k v : @& RawTensor) (dropout : Float) (causal : Bool := true) : IO RawTensor
@[extern "lean_torch_embedding"] opaque embedding (batch seq vocab embed : UInt64)
    (idx weight : @& RawTensor) (paddingIdx : Option Int := none)
    (maxNorm : Option Float := none) (normType : Float := 2.0)
    (scaleGradByFreq : Bool := false) (sparse : Bool := false) : RawTensor
@[extern "lean_torch_embedding_1d"] opaque embedding1d (seq vocab embed : UInt64)
    (idx weight : @& RawTensor) (paddingIdx : Option Int := none)
    (maxNorm : Option Float := none) (normType : Float := 2.0)
    (scaleGradByFreq : Bool := false) (sparse : Bool := false) : RawTensor
@[extern "lean_torch_cross_entropy_2d"] opaque crossEntropy (n c : UInt64)
    (logits targets : @& RawTensor) : RawTensor
@[extern "lean_torch_cross_entropy_2d_checked"] opaque crossEntropyChecked (n c : UInt64)
    (logits targets : @& RawTensor) : Except String RawTensor

end torch.RawTensor

namespace torch.RawTensor

@[extern "lean_torch_clamp_float"] opaque clampFloat (shape : Shape) (t : @& RawTensor) (lo hi : Float) : RawTensor
@[extern "lean_torch_diag"] opaque diag (n : UInt64) (t : @& RawTensor) : RawTensor
@[extern "lean_torch_matrix_exp"] opaque matrixExp (n : UInt64) (t : @& RawTensor) : RawTensor
@[extern "lean_torch_matrix_log"] opaque matrixLog (n : UInt64) (t : @& RawTensor) : RawTensor
@[extern "lean_torch_inv"] opaque inv (n : UInt64) (t : @& RawTensor) : RawTensor
@[extern "lean_torch_frobenius_norm"] opaque frobeniusNorm (m n : UInt64) (t : @& RawTensor) : Float
@[extern "lean_torch_row_norms"] opaque rowNorms (m n : UInt64) (t : @& RawTensor) : RawTensor

@[extern "lean_torch_slice_along_dim"] opaque slice (shape : Shape) (t : @& RawTensor)
    (dim start len : UInt64) : RawTensor

@[extern "lean_torch_max_row_norm"] opaque maxRowNorm (m n : UInt64) (t : @& RawTensor) : Float

@[extern "lean_torch_uniform_like"] opaque uniformLike (shape : Shape) (reference : @& RawTensor)
    (lo hi : Float) : IO RawTensor

@[extern "lean_torch_eye"] opaque eye (n : UInt64) (requiresGrad : Bool := false) (noGrad : requiresGrad = false := by decide) : RawTensor
@[extern "lean_torch_eye_checked"] opaque eyeChecked (n : UInt64) : Except String RawTensor

/-- Complete QR: returns (Q, R) with Q m×m and R m×n. -/
@[extern "lean_torch_qr"] opaque qr (m n : UInt64) (t : @& RawTensor) : RawTensor × RawTensor

/-- Reduced QR: returns (Q, R) with Q m×min(m,n) and R min(m,n)×n. -/
@[extern "lean_torch_qr_reduced"] opaque qrReduced (m n : UInt64) (t : @& RawTensor) : RawTensor × RawTensor

/-- Reduced SVD: returns (U, (S, Vh)) with U m×min(m,n), S of length min(m,n)
    and Vh min(m,n)×n. -/
@[extern "lean_torch_svd"] opaque svd (m n : UInt64) (t : @& RawTensor) : RawTensor × RawTensor × RawTensor

/-- Singular values only, as a vector of length min(m,n). -/
@[extern "lean_torch_svdvals"] opaque svdvals (m n : UInt64) (t : @& RawTensor) : RawTensor

/-- Diagonal matrix from a vector. -/
@[extern "lean_torch_diagflat"] opaque diagflat (n : UInt64) (t : @& RawTensor) : RawTensor

/-- Largest singular value. -/
@[extern "lean_torch_spectral_norm"] opaque spectralNorm (m n : UInt64) (t : @& RawTensor) : Float
@[extern "lean_torch_spectral_norm_checked"] opaque spectralNormChecked (m n : UInt64)
    (t : @& RawTensor) : Except String Float

/-- Sum of the singular values. -/
@[extern "lean_torch_nuclear_norm"] opaque nuclearNorm (m n : UInt64) (t : @& RawTensor) : Float
@[extern "lean_torch_nuclear_norm_checked"] opaque nuclearNormChecked (m n : UInt64)
    (t : @& RawTensor) : Except String Float

@[extern "lean_torch_tensor_atan"] opaque atan (shape : Shape) (t : @& RawTensor) : RawTensor

/-- Maximum over all elements, as a scalar handle. The native side reads no
    shape argument, so none is passed. -/
@[extern "lean_torch_max_all"] opaque maxAll (t : @& RawTensor) : RawTensor

/-- Stack a runtime list of equal vectors along a new leading axis.
    The native bridge consumes the array (but retains shared tensor handles). -/
@[extern "lean_torch_stack_1d"] opaque stack1d (k n : UInt64) (tensors : Array RawTensor)
    (dim : Int64 := 0) : RawTensor

/-- Vector-Jacobian product: gradient of `y` w.r.t. `x` under cotangent `dy`. -/
@[extern "lean_torch_tensor_grad"] opaque grad (sx sy : Shape) (y x dy : @& RawTensor) : RawTensor

/-- 1-D Float32 tensor from a host array. -/
@[extern "lean_torch_from_float_array"] opaque fromFloatArray (arr : @& Array Float) : RawTensor

/-- Draw `numSamples` indices per row of a 2-D probability matrix; Int64 result. -/
@[extern "lean_torch_multinomial"] opaque multinomial (shape : Shape) (probs : @& RawTensor)
    (numSamples : UInt64) (replacement : Bool) : IO RawTensor

/-- In-place-style scatter of `src` into `input` along `dim` at `indices`
    (returns a new tensor). Dimension arguments mirror the C ABI. -/
@[extern "lean_torch_scatter_2d"] opaque scatter2d (batch seq k : UInt64) (input : @& RawTensor)
    (dim : Int64) (indices src : @& RawTensor) : RawTensor

/-- All-close predicate: no shape argument is read natively, so none is passed. -/
@[extern "lean_torch_allclose"] opaque allclose (a b : @& RawTensor) (rtol : Float := 1e-05)
    (atol : Float := 1e-08) : Bool

/-- Whether the tensor is an autograd leaf. -/
@[extern "lean_torch_is_leaf"] opaque isLeaf (t : @& RawTensor) : Bool

/-- Log-softmax along a dimension, spec-preserving. The native side reads no
    shape argument, so none is passed. -/
@[extern "lean_torch_log_softmax"] opaque logSoftmax (t : @& RawTensor) (dim : Int32 := -1) : RawTensor

/-- Backward pass from a scalar loss (gradient = 1.0). -/
@[extern "lean_torch_backward_loss"] opaque backwardLoss (loss : @& RawTensor) : IO Unit

/-- Argmax along a dimension (Int64 indices). The shape argument is unread natively. -/
@[extern "lean_torch_argmax"] opaque argmax (shape : Shape) (t : @& RawTensor) (dim : UInt64) : RawTensor

/-- Recoverable argmax, including dtype, empty-axis and kernel failures. -/
@[extern "lean_torch_argmax_checked"] opaque argmaxChecked (shape : Shape)
    (t : @& RawTensor) (dim : UInt64) : Except String RawTensor

/-- Top-k logit filter (entries outside the top k become -inf), shape-preserving. -/
@[extern "lean_torch_topk_filter"] opaque topKFilter (shape : Shape) (logits : @& RawTensor) (k : UInt64) : RawTensor

/-- Top-p (nucleus) logit filter, shape-preserving. -/
@[extern "lean_torch_topp_filter"] opaque topPFilter (shape : Shape) (logits : @& RawTensor) (p : Float) : RawTensor

/-- Elementwise equality with a scalar; result is a Bool tensor of the same shape. -/
@[extern "lean_torch_eq_scalar"] opaque eqScalar (shape : Shape) (input : @& RawTensor) (scalar : Int64) : RawTensor

/-- 1-D Int64 tensor from a host array. -/
@[extern "lean_torch_from_int64_array"] opaque fromInt64Array (arr : @& Array Int64) : RawTensor

/-- Select entries of a 1-D tensor by Int64 indices along `dim` (0). The
    dimension arguments mirror the C ABI and are unread natively. -/
@[extern "lean_torch_index_select"] opaque indexSelect (n k : UInt64) (data : @& RawTensor)
    (dim : Int64) (indices : @& RawTensor) : RawTensor

/-- Checked vector indexing. Bounds are validated before device indexing so
invalid accelerator indices cannot trigger a device assertion. -/
@[extern "lean_torch_index_select_checked"] opaque indexSelectChecked (n k : UInt64)
    (data : @& RawTensor) (dim : Int64) (indices : @& RawTensor) : Except String RawTensor

/-- Per-element cross entropy (no reduction): logits [n, c], Int64 targets [n]. -/
@[extern "lean_torch_cross_entropy_none"] opaque crossEntropyNone (n c : UInt64)
    (logits : @& RawTensor) (targets : @& RawTensor) : RawTensor
@[extern "lean_torch_cross_entropy_none_checked"] opaque crossEntropyNoneChecked (n c : UInt64)
    (logits targets : @& RawTensor) : Except String RawTensor

/-- Extract an Int64 scalar. The shape argument is unread natively. -/
@[extern "lean_torch_item_int"] opaque itemInt (shape : Shape) (t : @& RawTensor) : Int64

/-- Elementwise logical negation of a Bool tensor. The shape argument is unread natively. -/
@[extern "lean_torch_logical_not"] opaque logicalNot (shape : Shape) (t : @& RawTensor) : RawTensor

/-- Elementwise logical conjunction of two Bool tensors. The shape argument is unread natively. -/
@[extern "lean_torch_logical_and"] opaque logicalAnd (shape : Shape) (a b : @& RawTensor) : RawTensor

@[extern "lean_torch_conv1d"] opaque conv1d (inputShape weightShape : Shape)
    (input weight : @& RawTensor) (stride padding dilation : UInt64) : RawTensor
@[extern "lean_torch_conv1d_checked"] opaque conv1dChecked (inputShape weightShape : Shape)
    (input weight : @& RawTensor) (stride padding dilation : UInt64) : Except String RawTensor
@[extern "lean_torch_tensor_to_float_array_dynamic"] opaque toFloatArray
    (t : @& RawTensor) : IO (Array Float)
@[extern "lean_torch_any"] opaque any (shape : Shape) (t : @& RawTensor) : Bool
@[extern "lean_torch_tensor_to_uint64_array_dynamic"] opaque toUInt64Array
    (t : @& RawTensor) : IO (Array UInt64)
@[extern "lean_torch_randint"] opaque randint (low high : Int64) (shape : Shape)
    (requiresGrad : Bool) (device : Device) : IO RawTensor

end torch.RawTensor

namespace torch.RawTensor
/-- Differentiate a pure scalar function with private fresh leaves. The callback
cannot export the temporary leaves; only detached derivatives are returned. -/
@[extern "lean_torch_functional_grad"] opaque functionalGrad
    (inputs : @& Array RawTensor) (f : Array RawTensor → RawTensor) :
    Except String (Array RawTensor)
end torch.RawTensor

namespace torch.RawTensor
/-- Recover native execution failures without unwinding across Lean frames. -/
@[extern "lean_torch_tensor_add_checked"] opaque addChecked (shape : Shape) (a b : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_tensor_sub_checked"] opaque subChecked (shape : Shape) (a b : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_tensor_mul_checked"] opaque mulChecked (shape : Shape) (a b : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_div_checked"] opaque divChecked (shape : Shape) (a b : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_tensor_relu_checked"] opaque reluChecked (shape : Shape) (t : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_abs_checked"] opaque absChecked (shape : Shape) (t : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_expand_checked"] opaque expandChecked (shape : Shape) (t : @& RawTensor) (target : Array UInt64) : Except String RawTensor

@[extern "lean_torch_reshape_exact_checked"] opaque reshapeChecked (shape : Shape) (t : @& RawTensor) (target : @& Shape) : Except String RawTensor

@[extern "lean_torch_sum_checked"] opaque sumChecked (shape : Shape) (t : @& RawTensor) (dims : Option (Array UInt64)) (keepdim : Bool) : Except String RawTensor

@[extern "lean_torch_mean_checked"] opaque meanChecked (shape : Shape) (t : @& RawTensor) (dims : Option (Array UInt64)) (keepdim : Bool) : Except String RawTensor

@[extern "lean_torch_max_all_checked"] opaque maxAllChecked (t : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_where_checked"] opaque whereSelectChecked (shape : Shape) (condition a b : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_matmul_checked"] opaque matmulChecked (s1 s2 : Shape) (a b : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_mm_checked"] opaque mmChecked (m n k : UInt64) (a b : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_linear_checked"] opaque linearChecked (m n b : UInt64) (a b : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_linear3d_checked"] opaque linear3dChecked (batch seq inDim outDim : UInt64) (a b : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_matmul3d_2d_checked"] opaque matmul3dChecked (batch seq k n : UInt64) (a b : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_bmm4d_checked"] opaque bmm4dChecked (b h m k n : UInt64) (a b : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_row_norms_checked"] opaque rowNormsChecked (m n : UInt64) (t : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_svd_checked"] opaque svdChecked (m n : UInt64) (t : @& RawTensor) : Except String (RawTensor × RawTensor × RawTensor)

@[extern "lean_torch_svdvals_checked"] opaque svdvalsChecked (m n : UInt64) (t : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_add_scalar_checked"] opaque addScalarChecked (shape : Shape) (t : @& RawTensor) (value : Float) : Except String RawTensor

@[extern "lean_torch_sub_scalar_checked"] opaque subScalarChecked (shape : Shape) (t : @& RawTensor) (value : Float) : Except String RawTensor

@[extern "lean_torch_mul_scalar_checked"] opaque mulScalarChecked (shape : Shape) (t : @& RawTensor) (value : Float) : Except String RawTensor

@[extern "lean_torch_div_scalar_checked"] opaque divScalarChecked (shape : Shape) (t : @& RawTensor) (value : Float) : Except String RawTensor

@[extern "lean_torch_l2_norm_checked"] opaque l2NormChecked (n : UInt64) (t : @& RawTensor) : Except String Float

@[extern "lean_torch_frobenius_norm_checked"] opaque frobeniusNormChecked (m n : UInt64) (t : @& RawTensor) : Except String Float

@[extern "lean_torch_max_row_norm_checked"] opaque maxRowNormChecked (m n : UInt64) (t : @& RawTensor) : Except String Float

@[extern "lean_torch_rms_norm_weighted_checked"] opaque rmsNormWeightedChecked (shape weightShape : Shape)
    (x weight : @& RawTensor) (eps : Float := 1e-6) : Except String RawTensor

@[extern "lean_torch_matrix_log_checked"] opaque matrixLogChecked (n : UInt64) (t : @& RawTensor) : Except String RawTensor

def sumAllChecked (shape : Shape) (t : RawTensor) : Except String RawTensor := sumChecked shape t none false

def meanAllChecked (shape : Shape) (t : RawTensor) : Except String RawTensor := meanChecked shape t none false
end torch.RawTensor

namespace torch.RawTensor
@[extern "lean_torch_tensor_exp_checked"] opaque expChecked (shape : Shape) (t : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_sqrt_checked"] opaque sqrtChecked (shape : Shape) (t : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_rsqrt_checked"] opaque rsqrtChecked (shape : Shape) (t : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_tensor_sigmoid_checked"] opaque sigmoidChecked (shape : Shape) (t : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_tensor_tanh_checked"] opaque tanhChecked (shape : Shape) (t : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_tensor_gelu_checked"] opaque geluChecked (shape : Shape) (t : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_tensor_silu_checked"] opaque siluChecked (shape : Shape) (t : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_pow_checked"] opaque powChecked (shape : Shape) (t : @& RawTensor) (exponent : Float) : Except String RawTensor

@[extern "lean_torch_tensor_atan_checked"] opaque atanChecked (shape : Shape) (t : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_tensor_softmax_checked"] opaque softmaxChecked (shape : Shape) (t : @& RawTensor) (dim : Int32 := -1) : Except String RawTensor

@[extern "lean_torch_log_softmax_checked"] opaque logSoftmaxChecked (t : @& RawTensor) (dim : Int32 := -1) : Except String RawTensor
end torch.RawTensor

namespace torch.RawTensor
@[extern "lean_torch_maximum_checked"] opaque maximumChecked (shape : Shape)
    (a b : @& RawTensor) : Except String RawTensor
end torch.RawTensor

namespace torch.RawTensor
@[extern "lean_torch_gt_checked"] opaque gtChecked (shape : Shape) (a b : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_logical_and_checked"] opaque logicalAndChecked (shape : Shape) (a b : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_cat2_checked"] opaque catChecked (s1 s2 : Shape) (a b : @& RawTensor) (dim : Int) : Except String RawTensor
end torch.RawTensor
