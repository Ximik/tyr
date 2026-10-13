import Tyr.Basic

/-! Native libtorch ABI. These handles carry no static guarantees.
Model code should use `torch.Tensor`; only typed adapters call this module. -/
namespace torch.RawTensor

@[extern "lean_torch_zeros"] opaque zeros (shape : Shape) (requiresGrad : Bool := false) (device : Device := .CPU) : RawTensor

@[extern "lean_torch_ones"] opaque ones (shape : Shape) (requiresGrad : Bool := false) (device : Device := .CPU) : RawTensor

@[extern "lean_torch_randn"] opaque randn (shape : Shape) (requiresGrad : Bool := false) (device : Device := .CPU) : IO RawTensor

@[extern "lean_torch_full"] opaque full (shape : Shape) (value : Float) (requiresGrad : Bool := false) (device : Device := .CPU) : RawTensor

@[extern "lean_torch_full_int"] opaque fullInt (shape : Shape) (value : Int64) : RawTensor

@[extern "lean_torch_zeros_like"] opaque zerosLike (shape : Shape) (t : @& RawTensor) : RawTensor

@[extern "lean_torch_ones_like"] opaque onesLike (shape : Shape) (t : @& RawTensor) : RawTensor

@[extern "lean_torch_to_float"] opaque toFloat32 (shape : Shape) (t : @& RawTensor) : RawTensor

@[extern "lean_torch_to_bfloat16"] opaque toBFloat16 (shape : Shape) (t : @& RawTensor) : RawTensor

@[extern "lean_torch_to_long"] opaque toInt64 (shape : Shape) (t : @& RawTensor) : RawTensor

@[extern "lean_torch_to"] opaque toDevice (shape : Shape) (t : @& RawTensor) (device : Device) : RawTensor

@[extern "lean_torch_cast_dtype"] opaque castDType (shape : Shape) (t : @& RawTensor) (tag : @& String) : RawTensor

@[extern "lean_torch_requires_grad"] opaque requiresGrad (shape : Shape) (t : @& RawTensor) : Bool

@[extern "lean_torch_l2_norm"] opaque l2Norm (n : UInt64) (t : @& RawTensor) : Float

@[extern "lean_torch_item"] opaque item (shape : Shape) (t : @& RawTensor) : Float

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

@[extern "lean_torch_grad_of"] opaque gradOf (shape : Shape) (t : @& RawTensor) : RawTensor

@[extern "lean_torch_zero_grad"] opaque zeroGrad (shape : Shape) (t : @& RawTensor) : RawTensor

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

@[extern "lean_torch_transpose"] opaque transpose (shape : Shape) (t : @& RawTensor) (dim0 dim1 : UInt64) : RawTensor

@[extern "lean_torch_unsqueeze"] opaque unsqueeze (shape : Shape) (t : @& RawTensor) (dim : Nat) : RawTensor

@[extern "lean_torch_squeeze"] opaque squeeze (shape : Shape) (t : @& RawTensor) (dim : Nat) : RawTensor

@[extern "lean_torch_cat2"] opaque cat (s1 s2 : Shape) (a b : @& RawTensor) (dim : Int) : RawTensor

@[extern "lean_torch_sum"] opaque sum (shape : Shape) (t : @& RawTensor) (dims : Option (Array UInt64)) (keepdim : Bool) : RawTensor

@[extern "lean_torch_mean"] opaque mean (shape : Shape) (t : @& RawTensor) (dims : Option (Array UInt64)) (keepdim : Bool) : RawTensor

@[extern "lean_torch_set_requires_grad"] opaque setRequiresGrad (shape : Shape) (t : @& RawTensor) (requiresGrad : Bool) : RawTensor

@[extern "lean_torch_affine3d"] opaque affine3d (batch seq inDim outDim : UInt64) (x weight bias : @& RawTensor) : RawTensor


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
    (q k v : @& RawTensor) (dropout : Float) (causal : Bool := true) : RawTensor
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

@[extern "lean_torch_eye"] opaque eye (n : UInt64) (requiresGrad : Bool := false) : RawTensor

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

/-- Sum of the singular values. -/
@[extern "lean_torch_nuclear_norm"] opaque nuclearNorm (m n : UInt64) (t : @& RawTensor) : Float

@[extern "lean_torch_tensor_atan"] opaque atan (shape : Shape) (t : @& RawTensor) : RawTensor

/-- Maximum over all elements, as a scalar handle. The native side reads no
    shape argument, so none is passed. -/
@[extern "lean_torch_max_all"] opaque maxAll (t : @& RawTensor) : RawTensor

/-- Stack a runtime list of equal vectors along a new leading axis. -/
@[extern "lean_torch_stack_1d"] opaque stack1d (k n : UInt64) (tensors : @& Array RawTensor)
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

end torch.RawTensor
