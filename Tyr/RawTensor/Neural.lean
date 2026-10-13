import Tyr.RawTensor

/-! Native neural-network operations. Only typed adapters consume these handles. -/
namespace torch.RawTensor

@[extern "lean_torch_conv2d_bias"] opaque conv2dBias (inputShape weightShape biasShape : Shape)
    (input weight bias : @& RawTensor) (stride padding dilation : Array UInt64) : RawTensor
@[extern "lean_torch_cat_dyn"] opaque catMany (tensors : @& Array RawTensor) (dim : Int) : RawTensor
@[extern "lean_torch_arange"] opaque arange (start stop step : UInt64) : RawTensor
@[extern "lean_torch_tensor_sin"] opaque sin (shape : Shape) (t : @& RawTensor) : RawTensor
@[extern "lean_torch_tensor_cos"] opaque cos (shape : Shape) (t : @& RawTensor) : RawTensor
@[extern "lean_torch_sdpa_gqa"] opaque attentionGQA (batch heads kvHeads seq dim : UInt64)
    (q k v : @& RawTensor) (dropout : Float) (causal gqa : Bool) : RawTensor
@[extern "lean_torch_sdpa_gqa_mask"] opaque attentionGQAMask (batch heads kvHeads seq dim : UInt64)
    (q k v mask : @& RawTensor) (dropout : Float) (causal gqa : Bool) : RawTensor

@[extern "lean_torch_gather"] opaque gather (inputShape indexShape : Shape)
    (input : @& RawTensor) (dim : Int64) (indices : @& RawTensor) : RawTensor
@[extern "lean_torch_masked_select"] opaque maskedSelect (shape : Shape)
    (input mask : @& RawTensor) : RawTensor
@[extern "lean_torch_masked_scatter"] opaque maskedScatter (shape sourceShape : Shape)
    (input mask source : @& RawTensor) : RawTensor

@[extern "lean_torch_sdpa_gqa_qkv"] opaque attentionGQAQKV (batch heads kvHeads qSeq kvSeq dim : UInt64)
    (q k v : @& RawTensor) (dropout : Float) (causal gqa : Bool) : RawTensor
@[extern "lean_torch_tyr_flash_attn_4d"] opaque flashAttention (batch heads kvHeads qSeq kvSeq dim : UInt64)
    (q k v : @& RawTensor) (mask : Option RawTensor) (dropout : Float)
    (causal : Bool) (scale : Option Float) (gqa : Bool) : RawTensor
@[extern "lean_torch_slice_scatter_along_dim"] opaque sliceScatter (shape sourceShape : Shape)
    (input : @& RawTensor) (dim start : UInt64) (source : @& RawTensor) : RawTensor
@[extern "lean_torch_apply_rotary_emb"] opaque applyRotary (batch seq heads dim : UInt64)
    (input cos sin : @& RawTensor) : RawTensor
@[extern "lean_torch_compute_rotary_freqs_on_device_pure"] opaque rotaryFreqs
    (seq dim : UInt64) (base : Float) (device : Device) : RawTensor × RawTensor
@[extern "lean_torch_hann_window"] opaque hannWindow (n : UInt64) : RawTensor
@[extern "lean_torch_stft_1d"] opaque stft (n : UInt64) (input : @& RawTensor)
    (nFft hopLength winLength : UInt64) (window : @& RawTensor) (center normalized : Bool) : RawTensor
@[extern "lean_torch_tensor_log10"] opaque log10 (shape : Shape) (input : @& RawTensor) : RawTensor
@[extern "lean_torch_save_wav"] opaque saveWav (shape : Shape) (input : @& RawTensor)
    (path : @& String) (sampleRate : UInt64) : IO Unit

@[extern "lean_torch_lt"] opaque lt (shape : Shape) (a b : @& RawTensor) : RawTensor
@[extern "lean_torch_ge"] opaque ge (shape : Shape) (a b : @& RawTensor) : RawTensor
@[extern "lean_torch_lt_scalar"] opaque ltScalar (shape : Shape) (input : @& RawTensor) (value : Float) : RawTensor
@[extern "lean_torch_masked_fill"] opaque maskedFill (shape : Shape) (input mask : @& RawTensor) (value : Float) : RawTensor

@[extern "lean_torch_conv1d_group_bias"] opaque conv1dGroups (inputShape weightShape biasShape : Shape)
    (input weight bias : @& RawTensor) (stride padding dilation groups : UInt64) : RawTensor
@[extern "lean_torch_conv_transpose1d_bias"] opaque convTranspose1d (inputShape weightShape biasShape : Shape)
    (input weight bias : @& RawTensor) (stride padding outputPadding dilation : UInt64) : RawTensor
@[extern "lean_torch_sdpa_gqa_window"] opaque attentionWindow (batch heads kvHeads seq dim : UInt64)
    (q k v : @& RawTensor) (dropout : Float) (causal gqa : Bool) (window : UInt64) : RawTensor

@[extern "lean_torch_tensor_elu"] opaque elu (shape : Shape) (input : @& RawTensor) : RawTensor

@[extern "lean_torch_tensor_log"] opaque log (shape : Shape) (input : @& RawTensor) : RawTensor
@[extern "lean_torch_safetensors_save"] opaque saveSafeTensor (shape : Shape)
    (path name : @& String) (input : @& RawTensor) : IO Unit

@[extern "lean_torch_clip_grad_norm_"] opaque clipGradNorm (shape : Shape)
    (input : @& RawTensor) (maxNorm : Float) : IO Float
@[extern "lean_torch_save_tensor"] opaque saveTensor (shape : Shape)
    (input : @& RawTensor) (path : @& String) : IO Unit
@[extern "lean_torch_load_tensor_exact"] opaque loadTensorExact (shape : Shape)
    (path : @& String) : IO RawTensor

@[extern "lean_torch_wav_begin"] opaque wavBegin (path : @& String) (sampleRate : UInt64) : IO Unit
@[extern "lean_torch_wav_append"] opaque wavAppend (shape : Shape)
    (input : @& RawTensor) (path : @& String) : IO Unit
@[extern "lean_torch_wav_finalize"] opaque wavFinalize (path : @& String) : IO Unit

end torch.RawTensor
