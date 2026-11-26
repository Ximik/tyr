import Tyr.Core.FFI.RawTensor

/-! Native neural-network operations. Only typed adapters consume these handles. -/
namespace torch.RawTensor

@[extern "lean_torch_copy_"] opaque copyInto
    (dst : RawTensor) (src : @& RawTensor) : IO RawTensor

@[extern "lean_torch_conv2d_bias"] opaque conv2dBias!
    (input weight bias : @& RawTensor) (stride padding dilation : Array Nat) : RawTensor
@[extern "lean_torch_cat_dyn"] opaque catMany! (tensors : @& Array RawTensor) (dim : @& Nat) : RawTensor
@[extern "lean_torch_arange"] opaque arange (start stop step : @& Nat) : RawTensor
@[extern "lean_torch_tensor_sin"] opaque sin! (t : @& RawTensor) : RawTensor
@[extern "lean_torch_tensor_cos"] opaque cos! (t : @& RawTensor) : RawTensor
@[extern "lean_torch_sdpa_gqa"] opaque attentionGQA!
    (q k v : @& RawTensor) (causal gqa : Bool) : RawTensor
@[extern "lean_torch_sdpa_gqa_io"] opaque attentionGQAIO
    (q k v : @& RawTensor) (dropout : Float) (causal gqa : Bool) : IO RawTensor
@[extern "lean_torch_sdpa_gqa_mask"] opaque attentionGQAMask!
    (q k v mask : @& RawTensor) (causal gqa : Bool) : RawTensor
@[extern "lean_torch_sdpa_gqa_mask_io"] opaque attentionGQAMaskIO
    (q k v mask : @& RawTensor) (dropout : Float) (causal gqa : Bool) : IO RawTensor
@[extern "lean_torch_sdpa_gqa_mask_qkv"] opaque attentionGQAMaskQKV!
    (query key value mask : @& RawTensor) (gqa : Bool) : RawTensor
@[extern "lean_torch_sdpa_gqa_mask_qkv_io"] opaque attentionGQAMaskQKVIO
    (query key value mask : @& RawTensor) (dropout : Float) (gqa : Bool) : IO RawTensor

@[extern "lean_torch_gather"] opaque gather
    (input : @& RawTensor) (dim : @& Nat) (indices : @& RawTensor) : RawTensor
@[extern "lean_torch_masked_select"] opaque maskedSelect
    (input mask : @& RawTensor) : RawTensor
@[extern "lean_torch_masked_scatter"] opaque maskedScatter
    (input mask source : @& RawTensor) : RawTensor

@[extern "lean_torch_slice_scatter_along_dim"] opaque sliceScatter
    (input : RawTensor) (dim start : @& Nat) (source : @& RawTensor) : RawTensor
@[extern "lean_torch_apply_rotary_emb"] opaque applyRotary
    (input cos sin : @& RawTensor) : RawTensor
@[extern "lean_torch_compute_rotary_freqs_on_device_pure"] opaque rotaryFreqs
    (seq dim : @& Nat) (base : Float) (device : Device) : RawTensor × RawTensor
@[extern "lean_torch_hann_window"] opaque hannWindow (n : @& Nat) : RawTensor
@[extern "lean_torch_stft_1d"] opaque stft (input : @& RawTensor)
    (nFft hopLength winLength : @& Nat) (window : @& RawTensor) (center normalized : Bool) : RawTensor
@[extern "lean_torch_tensor_log10"] opaque log10! (input : @& RawTensor) : RawTensor
@[extern "lean_torch_save_wav"] opaque saveWav (input : @& RawTensor)
    (path : @& String) (sampleRate : @& Nat) : IO Unit

@[extern "lean_torch_lt"] opaque lt! (a b : @& RawTensor) : RawTensor
@[extern "lean_torch_ge"] opaque ge! (a b : @& RawTensor) : RawTensor
@[extern "lean_torch_lt_scalar"] opaque ltScalar (input : @& RawTensor) (value : Float) : RawTensor
@[extern "lean_torch_masked_fill"] opaque maskedFill (input mask : @& RawTensor) (value : Float) : RawTensor

@[extern "lean_torch_conv1d_group_bias"] opaque conv1dGroups!
    (input weight bias : @& RawTensor) (stride padding dilation groups : @& Nat) : RawTensor
@[extern "lean_torch_conv_transpose1d_bias"] opaque convTranspose1d!
    (input weight bias : @& RawTensor) (stride padding outputPadding dilation : @& Nat) : RawTensor
@[extern "lean_torch_sdpa_gqa_window"] opaque attentionWindow!
    (q k v : @& RawTensor) (causal gqa : Bool) (window : @& Nat) : RawTensor
@[extern "lean_torch_sdpa_gqa_window_io"] opaque attentionWindowIO
    (q k v : @& RawTensor) (dropout : Float) (causal gqa : Bool) (window : @& Nat) : IO RawTensor

@[extern "lean_torch_tensor_elu"] opaque elu! (input : @& RawTensor) : RawTensor

@[extern "lean_torch_tensor_log"] opaque log! (input : @& RawTensor) : RawTensor

@[extern "lean_torch_clip_grad_norm_"] opaque clipGradNorm
    (input : @& RawTensor) (maxNorm : Float) : IO Float
@[extern "lean_torch_save_tensor"] opaque saveTensor
    (input : @& RawTensor) (path : @& String) : IO Unit
@[extern "lean_torch_load_tensor_exact"] opaque loadTensorExact (shape : Shape)
    (path : @& String) : IO RawTensor

@[extern "lean_torch_wav_begin"] opaque wavBegin (path : @& String) (sampleRate : Nat) : IO Unit
@[extern "lean_torch_wav_append"] opaque wavAppend
    (input : @& RawTensor) (path : @& String) : IO Unit
@[extern "lean_torch_wav_finalize"] opaque wavFinalize (path : @& String) : IO Unit

@[extern "lean_torch_load_u16_bin"] opaque loadU16Bin (count : @& Nat) (path : @& String) : IO RawTensor

@[extern "lean_torch_linspace"] opaque linspace (start stop : Float) (count : @& Nat) (requiresGrad : Bool) (noGrad : requiresGrad = false := by decide) : RawTensor
@[extern "lean_torch_max_dim_3d"] opaque maxLast3d!
    (input : @& RawTensor) (dim : @& Nat) : RawTensor × RawTensor
@[extern "lean_torch_topk_2d"] opaque topKLast2d!
    (input : @& RawTensor) (k dim : @& Nat) : RawTensor × RawTensor

@[extern "lean_torch_get"] opaque getOp (input : @& RawTensor) (idx : @& Nat) : RawTensor
@[extern "lean_torch_unbind"] opaque unbind (input : @& RawTensor) (dim : @& Nat) : Array RawTensor
@[extern "lean_torch_logspace"] opaque logspace (start stop : Float) (steps : @& Nat)
    (base : Float) (requiresGrad : Bool) (noGrad : requiresGrad = false := by decide) : RawTensor

@[extern "lean_torch_backward_unit"] opaque backward
    (output cotangent : @& RawTensor) (retainGraph createGraph : Bool) : IO Unit
@[extern "lean_torch_has_grad_fn"] opaque hasGradFn
    (input : @& RawTensor) : Bool

@[extern "lean_torch_find_bos_positions"] opaque findTokenPositions
    (tokens : @& RawTensor) (token : Int64) : IO RawTensor

@[extern "lean_torch_tensor_softplus"] opaque softplus (input : @& RawTensor) : RawTensor
@[extern "lean_torch_leaky_relu"] opaque leakyRelu (input : @& RawTensor) (slope : Float) : RawTensor
@[extern "lean_torch_softcap"] opaque softcap (input : @& RawTensor) (cap : Float) : RawTensor
@[extern "lean_torch_tensor_floor"] opaque floor (input : @& RawTensor) : RawTensor
@[extern "lean_torch_atan2"] opaque atan2 (y x : @& RawTensor) : RawTensor
@[extern "lean_torch_cumsum"] opaque cumsum (input : @& RawTensor) (dim : @& Nat) : RawTensor
@[extern "lean_torch_scatter_add"] opaque scatterAdd
    (input : @& RawTensor) (dim : @& Nat) (indices src : @& RawTensor) : RawTensor
@[extern "lean_torch_group_norm"] opaque groupNorm (input : @& RawTensor) (groups : @& Nat)
    (weight bias : Option RawTensor) (eps : Float) : RawTensor
@[extern "lean_torch_avg_pool2d"] opaque avgPool2d
    (input : @& RawTensor) (kernel stride padding : Array Nat) : RawTensor
@[extern "lean_torch_interpolate"] opaque interpolate (input : @& RawTensor)
    (size : Array Nat) (mode : String) (alignCorners : Bool) : RawTensor
@[extern "lean_torch_istft_1d"] opaque istft (input : @& RawTensor)
    (nFft hopLength winLength : @& Nat) (window : @& RawTensor)
    (center normalized : Bool) (length : @& Nat) : RawTensor
@[extern "lean_torch_timestep_embedding"] opaque timestepEmbedding (t : @& RawTensor) (dim : @& Nat)
    (maxPeriod timeFactor : Float) : RawTensor
@[extern "lean_torch_save_ppm"] opaque savePPM (image : @& RawTensor) (path : @& String) : IO Unit

end torch.RawTensor

namespace torch.RawTensor
@[extern "lean_torch_cat_dyn_try"] opaque catMany? (tensors : @& Array RawTensor) (dim : @& Nat) : Except String RawTensor
end torch.RawTensor
