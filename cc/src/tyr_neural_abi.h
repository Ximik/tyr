#pragma once
#include <lean/lean.h>

// Audited against the generated Tyr/RawTensor/Neural.c declarations.
#ifdef __cplusplus
extern "C" {
#endif
lean_object* lean_torch_conv2d_bias(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_cat_dyn(lean_object*, lean_object*);
lean_object* lean_torch_arange(uint64_t, uint64_t, uint64_t);
lean_object* lean_torch_tensor_sin(lean_object*, lean_object*);
lean_object* lean_torch_tensor_cos(lean_object*, lean_object*);
lean_object* lean_torch_sdpa_gqa(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*, lean_object*, double, uint8_t, uint8_t);
lean_object* lean_torch_sdpa_gqa_mask(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*, lean_object*, lean_object*, double, uint8_t, uint8_t);
lean_object* lean_torch_gather(lean_object*, lean_object*, lean_object*, uint64_t, lean_object*);
lean_object* lean_torch_masked_select(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_masked_scatter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_sdpa_gqa_qkv(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*, lean_object*, double, uint8_t, uint8_t);
lean_object* lean_torch_tyr_flash_attn_4d(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*, lean_object*, lean_object*, double, uint8_t, lean_object*, uint8_t);
lean_object* lean_torch_slice_scatter_along_dim(lean_object*, lean_object*, lean_object*, uint64_t, uint64_t, lean_object*);
lean_object* lean_torch_apply_rotary_emb(uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_compute_rotary_freqs_on_device_pure(uint64_t, uint64_t, double, lean_object*);
lean_object* lean_torch_hann_window(uint64_t);
lean_object* lean_torch_stft_1d(uint64_t, lean_object*, uint64_t, uint64_t, uint64_t, lean_object*, uint8_t, uint8_t);
lean_object* lean_torch_tensor_log10(lean_object*, lean_object*);
lean_object* lean_torch_save_wav(lean_object*, lean_object*, lean_object*, uint64_t);
lean_object* lean_torch_lt(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_ge(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_lt_scalar(lean_object*, lean_object*, double);
lean_object* lean_torch_masked_fill(lean_object*, lean_object*, lean_object*, double);
lean_object* lean_torch_conv1d_group_bias(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint64_t, uint64_t, uint64_t, uint64_t);
lean_object* lean_torch_conv_transpose1d_bias(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint64_t, uint64_t, uint64_t, uint64_t);
lean_object* lean_torch_sdpa_gqa_window(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*, lean_object*, double, uint8_t, uint8_t, uint64_t);
lean_object* lean_torch_tensor_elu(lean_object*, lean_object*);
lean_object* lean_torch_tensor_log(lean_object*, lean_object*);
lean_object* lean_torch_safetensors_save(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_clip_grad_norm_(lean_object*, lean_object*, double);
lean_object* lean_torch_save_tensor(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_load_tensor_exact(lean_object*, lean_object*);
#ifdef __cplusplus
}
#endif
