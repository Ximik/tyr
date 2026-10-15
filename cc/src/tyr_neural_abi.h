#pragma once
#include <lean/lean.h>

// Audited against the generated Tyr/RawTensor/Neural.c declarations.
#ifdef __cplusplus
extern "C" {
#endif
// copy_ consumes destination, borrows source, and returns IO RawTensor.
lean_object* lean_torch_copy_(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_conv2d_bias(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_conv2d_bias_checked(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_cat_dyn(lean_object*, lean_object*);
lean_object* lean_torch_arange(uint64_t, uint64_t, uint64_t);
lean_object* lean_torch_tensor_sin(lean_object*, lean_object*);
lean_object* lean_torch_tensor_cos(lean_object*, lean_object*);
lean_object* lean_torch_sdpa_gqa(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t);
lean_object* lean_torch_sdpa_gqa_io(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*, lean_object*, double, uint8_t, uint8_t);
lean_object* lean_torch_sdpa_gqa_mask(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t);
lean_object* lean_torch_sdpa_gqa_mask_io(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*, lean_object*, lean_object*, double, uint8_t, uint8_t);
lean_object* lean_torch_gather(lean_object*, lean_object*, lean_object*, uint64_t, lean_object*);
lean_object* lean_torch_masked_select(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_masked_scatter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_sdpa_gqa_qkv(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t);
lean_object* lean_torch_sdpa_gqa_qkv_io(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*, lean_object*, double, uint8_t, uint8_t);
// Except String RawTensor; borrows Q/K/V and consumes both Option arguments.
lean_object* lean_torch_tyr_flash_attn_4d(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, uint8_t);
lean_object* lean_torch_tyr_flash_attn_4d_io(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*, lean_object*, lean_object*, double, uint8_t, lean_object*, uint8_t);
// slice_scatter consumes input, borrows source, and returns a tensor value.
lean_object* lean_torch_slice_scatter_along_dim(lean_object*, lean_object*, lean_object*, uint64_t, uint64_t, lean_object*);
lean_object* lean_torch_apply_rotary_emb(uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_compute_rotary_freqs_on_device_pure(uint64_t, uint64_t, double, lean_object*);
lean_object* lean_torch_hann_window(uint64_t);
lean_object* lean_torch_stft_1d(uint64_t, lean_object*, uint64_t, uint64_t, uint64_t, lean_object*, uint8_t, uint8_t);
// Except String RawTensor; borrows input and window.
lean_object* lean_torch_stft_1d_checked(uint64_t, lean_object*, uint64_t, uint64_t, uint64_t, lean_object*, uint8_t, uint8_t);
lean_object* lean_torch_tensor_log10(lean_object*, lean_object*);
lean_object* lean_torch_save_wav(lean_object*, lean_object*, lean_object*, uint64_t);
lean_object* lean_torch_lt(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_ge(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_lt_scalar(lean_object*, lean_object*, double);
lean_object* lean_torch_masked_fill(lean_object*, lean_object*, lean_object*, double);
lean_object* lean_torch_conv1d_group_bias(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint64_t, uint64_t, uint64_t, uint64_t);
lean_object* lean_torch_conv1d_group_bias_checked(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint64_t, uint64_t, uint64_t, uint64_t);
lean_object* lean_torch_conv_transpose1d_bias(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint64_t, uint64_t, uint64_t, uint64_t);
lean_object* lean_torch_conv_transpose1d_bias_checked(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint64_t, uint64_t, uint64_t, uint64_t);
lean_object* lean_torch_sdpa_gqa_window(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint64_t);
lean_object* lean_torch_sdpa_gqa_window_io(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*, lean_object*, double, uint8_t, uint8_t, uint64_t);
lean_object* lean_torch_tensor_elu(lean_object*, lean_object*);
lean_object* lean_torch_tensor_log(lean_object*, lean_object*);
lean_object* lean_torch_safetensors_save(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_clip_grad_norm_(lean_object*, lean_object*, double);
lean_object* lean_torch_save_tensor(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_load_tensor_exact(lean_object*, lean_object*);
lean_object* lean_torch_wav_begin(lean_object*, uint64_t);
lean_object* lean_torch_wav_append(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_wav_finalize(lean_object*);
lean_object* lean_torch_load_u16_bin(uint64_t, lean_object*);
lean_object* lean_torch_linspace(double, double, uint64_t, uint8_t);
lean_object* lean_torch_max_dim_3d(uint64_t, uint64_t, uint64_t, lean_object*, uint64_t);
lean_object* lean_torch_max_dim_3d_checked(uint64_t, uint64_t, uint64_t, lean_object*, uint64_t);
lean_object* lean_torch_topk_2d(uint64_t, uint64_t, lean_object*, uint64_t, uint64_t);
lean_object* lean_torch_topk_2d_checked(uint64_t, uint64_t, lean_object*, uint64_t, uint64_t);
lean_object* lean_torch_get(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_unbind(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_logspace(double, double, uint64_t, double, uint8_t);
lean_object* lean_torch_backward_unit(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t);
uint8_t lean_torch_has_grad_fn(lean_object*, lean_object*);
lean_object* lean_torch_find_bos_positions(uint64_t, lean_object*, uint64_t);
// Checked variants return Except String with the same payload and argument ownership.
lean_object* lean_torch_cat_dyn_checked(lean_object*, lean_object*);
lean_object* lean_torch_tensor_log_checked(lean_object*, lean_object*);
lean_object* lean_torch_tensor_log10_checked(lean_object*, lean_object*);
lean_object* lean_torch_tensor_sin_checked(lean_object*, lean_object*);
lean_object* lean_torch_tensor_cos_checked(lean_object*, lean_object*);
lean_object* lean_torch_tensor_elu_checked(lean_object*, lean_object*);
lean_object* lean_torch_lt_checked(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_ge_checked(lean_object*, lean_object*, lean_object*);
#ifdef __cplusplus
}
#endif
