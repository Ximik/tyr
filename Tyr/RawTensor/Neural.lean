import Tyr.RawTensor

/-! Native neural-network operations. Only typed adapters consume these handles. -/
namespace torch.RawTensor

@[extern "lean_torch_copy_"] opaque copyInto
    (dst : RawTensor) (src : @& RawTensor) : IO RawTensor

@[extern "lean_torch_conv2d_bias"] opaque conv2dBias
    (input weight bias : @& RawTensor) (stride padding dilation : Array UInt64) : RawTensor
@[extern "lean_torch_conv2d_bias_checked"] opaque conv2dBiasChecked
    (input weight bias : @& RawTensor) (stride padding dilation : Array UInt64) : Except String RawTensor
@[extern "lean_torch_cat_dyn"] opaque catMany (tensors : @& Array RawTensor) (dim : Int) : RawTensor
@[extern "lean_torch_arange"] opaque arange (start stop step : UInt64) : RawTensor
@[extern "lean_torch_tensor_sin"] opaque sin (t : @& RawTensor) : RawTensor
@[extern "lean_torch_tensor_cos"] opaque cos (t : @& RawTensor) : RawTensor
@[extern "lean_torch_sdpa_gqa"] opaque attentionGQA
    (q k v : @& RawTensor) (causal gqa : Bool) : Except String RawTensor
@[extern "lean_torch_sdpa_gqa_io"] opaque attentionGQAIO
    (q k v : @& RawTensor) (dropout : Float) (causal gqa : Bool) : IO RawTensor
@[extern "lean_torch_sdpa_gqa_mask"] opaque attentionGQAMask
    (q k v mask : @& RawTensor) (causal gqa : Bool) : Except String RawTensor
@[extern "lean_torch_sdpa_gqa_mask_io"] opaque attentionGQAMaskIO
    (q k v mask : @& RawTensor) (dropout : Float) (causal gqa : Bool) : IO RawTensor

@[extern "lean_torch_gather"] opaque gather
    (input : @& RawTensor) (dim : Int64) (indices : @& RawTensor) : RawTensor
@[extern "lean_torch_masked_select"] opaque maskedSelect
    (input mask : @& RawTensor) : RawTensor
@[extern "lean_torch_masked_scatter"] opaque maskedScatter
    (input mask source : @& RawTensor) : RawTensor

@[extern "lean_torch_sdpa_gqa_qkv"] opaque attentionGQAQKV
    (q k v : @& RawTensor) (causal gqa : Bool) : Except String RawTensor
@[extern "lean_torch_sdpa_gqa_qkv_io"] opaque attentionGQAQKVIO
    (q k v : @& RawTensor) (dropout : Float) (causal gqa : Bool) : IO RawTensor
@[extern "lean_torch_tyr_flash_attn_4d"] opaque flashAttention
    (q k v : @& RawTensor) (mask : Option RawTensor)
    (causal : Bool) (scale : Option Float) (gqa : Bool) : Except String RawTensor
@[extern "lean_torch_tyr_flash_attn_4d_io"] opaque flashAttentionIO
    (q k v : @& RawTensor) (mask : Option RawTensor) (dropout : Float)
    (causal : Bool) (scale : Option Float) (gqa : Bool) : IO RawTensor
@[extern "lean_torch_slice_scatter_along_dim"] opaque sliceScatter
    (input : RawTensor) (dim start : UInt64) (source : @& RawTensor) : RawTensor
@[extern "lean_torch_apply_rotary_emb"] opaque applyRotary
    (input cos sin : @& RawTensor) : RawTensor
@[extern "lean_torch_compute_rotary_freqs_on_device_pure"] opaque rotaryFreqs
    (seq dim : UInt64) (base : Float) (device : Device) : RawTensor × RawTensor
@[extern "lean_torch_hann_window"] opaque hannWindow (n : UInt64) : RawTensor
@[extern "lean_torch_stft_1d"] opaque stft (input : @& RawTensor)
    (nFft hopLength winLength : UInt64) (window : @& RawTensor) (center normalized : Bool) : RawTensor
/-- STFT with all native validation and kernel failures contained in `Except`. -/
@[extern "lean_torch_stft_1d_checked"] opaque stftChecked (input : @& RawTensor)
    (nFft hopLength winLength : UInt64) (window : @& RawTensor) (center normalized : Bool) :
    Except String RawTensor
@[extern "lean_torch_tensor_log10"] opaque log10 (input : @& RawTensor) : RawTensor
@[extern "lean_torch_save_wav"] opaque saveWav (input : @& RawTensor)
    (path : @& String) (sampleRate : UInt64) : IO Unit

@[extern "lean_torch_lt"] opaque lt (a b : @& RawTensor) : RawTensor
@[extern "lean_torch_ge"] opaque ge (a b : @& RawTensor) : RawTensor
@[extern "lean_torch_lt_scalar"] opaque ltScalar (input : @& RawTensor) (value : Float) : RawTensor
@[extern "lean_torch_masked_fill"] opaque maskedFill (input mask : @& RawTensor) (value : Float) : RawTensor

@[extern "lean_torch_conv1d_group_bias"] opaque conv1dGroups
    (input weight bias : @& RawTensor) (stride padding dilation groups : UInt64) : RawTensor
@[extern "lean_torch_conv1d_group_bias_checked"] opaque conv1dGroupsChecked
    (input weight bias : @& RawTensor) (stride padding dilation groups : UInt64) : Except String RawTensor
@[extern "lean_torch_conv_transpose1d_bias"] opaque convTranspose1d
    (input weight bias : @& RawTensor) (stride padding outputPadding dilation : UInt64) : RawTensor
@[extern "lean_torch_conv_transpose1d_bias_checked"] opaque convTranspose1dChecked
    (input weight bias : @& RawTensor) (stride padding outputPadding dilation : UInt64) : Except String RawTensor
@[extern "lean_torch_sdpa_gqa_window"] opaque attentionWindow
    (q k v : @& RawTensor) (causal gqa : Bool) (window : UInt64) : Except String RawTensor
@[extern "lean_torch_sdpa_gqa_window_io"] opaque attentionWindowIO
    (q k v : @& RawTensor) (dropout : Float) (causal gqa : Bool) (window : UInt64) : IO RawTensor

@[extern "lean_torch_tensor_elu"] opaque elu (input : @& RawTensor) : RawTensor

@[extern "lean_torch_tensor_log"] opaque log (input : @& RawTensor) : RawTensor

@[extern "lean_torch_clip_grad_norm_"] opaque clipGradNorm
    (input : @& RawTensor) (maxNorm : Float) : IO Float
@[extern "lean_torch_save_tensor"] opaque saveTensor
    (input : @& RawTensor) (path : @& String) : IO Unit
@[extern "lean_torch_load_tensor_exact"] opaque loadTensorExact (shape : Shape)
    (path : @& String) : IO RawTensor

@[extern "lean_torch_wav_begin"] opaque wavBegin (path : @& String) (sampleRate : UInt64) : IO Unit
@[extern "lean_torch_wav_append"] opaque wavAppend
    (input : @& RawTensor) (path : @& String) : IO Unit
@[extern "lean_torch_wav_finalize"] opaque wavFinalize (path : @& String) : IO Unit

@[extern "lean_torch_load_u16_bin"] opaque loadU16Bin (count : UInt64) (path : @& String) : IO RawTensor

@[extern "lean_torch_linspace"] opaque linspace (start stop : Float) (count : UInt64) (requiresGrad : Bool) (noGrad : requiresGrad = false := by decide) : RawTensor
@[extern "lean_torch_max_dim_3d"] opaque maxLast3d
    (input : @& RawTensor) (dim : UInt64) : RawTensor × RawTensor
@[extern "lean_torch_max_dim_3d_checked"] opaque maxLast3dChecked
    (input : @& RawTensor) (dim : UInt64) : Except String (RawTensor × RawTensor)
@[extern "lean_torch_topk_2d"] opaque topKLast2d
    (input : @& RawTensor) (k dim : UInt64) : RawTensor × RawTensor
@[extern "lean_torch_topk_2d_checked"] opaque topKLast2dChecked
    (input : @& RawTensor) (k dim : UInt64) : Except String (RawTensor × RawTensor)

@[extern "lean_torch_get"] opaque getOp (input : @& RawTensor) (idx : Int) : RawTensor
@[extern "lean_torch_unbind"] opaque unbind (input : @& RawTensor) (dim : Nat) : Array RawTensor
@[extern "lean_torch_logspace"] opaque logspace (start stop : Float) (steps : UInt64)
    (base : Float) (requiresGrad : Bool) (noGrad : requiresGrad = false := by decide) : RawTensor

@[extern "lean_torch_backward_unit"] opaque backward
    (output cotangent : @& RawTensor) (retainGraph createGraph : Bool) : IO Unit
@[extern "lean_torch_has_grad_fn"] opaque hasGradFn
    (input : @& RawTensor) : Bool

@[extern "lean_torch_find_bos_positions"] opaque findTokenPositions
    (tokens : @& RawTensor) (token : Int64) : IO RawTensor

end torch.RawTensor

namespace torch.RawTensor
@[extern "lean_torch_cat_dyn_checked"] opaque catManyChecked (tensors : @& Array RawTensor) (dim : Int) : Except String RawTensor
end torch.RawTensor

namespace torch.RawTensor
@[extern "lean_torch_tensor_log_checked"] opaque logChecked (input : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_tensor_log10_checked"] opaque log10Checked (input : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_tensor_sin_checked"] opaque sinChecked (t : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_tensor_cos_checked"] opaque cosChecked (t : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_tensor_elu_checked"] opaque eluChecked (input : @& RawTensor) : Except String RawTensor
end torch.RawTensor

namespace torch.RawTensor
@[extern "lean_torch_lt_checked"] opaque ltChecked (a b : @& RawTensor) : Except String RawTensor

@[extern "lean_torch_ge_checked"] opaque geChecked (a b : @& RawTensor) : Except String RawTensor
end torch.RawTensor
