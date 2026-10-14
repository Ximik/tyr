#pragma once
#include <lean/lean.h>

// Native calls used by Tyr.RawTensor. Check against generated RawTensor.c
// with scripts/check_ffi_abi.py; including this header checks the C++ side.
#ifdef __cplusplus
extern "C" {
#endif

// Basic metadata calls consume their legacy shape argument and borrow the tensor.
lean_object* lean_torch_to_string(lean_object*, lean_object*);
lean_object* lean_torch_get_shape(lean_object*, lean_object*);
lean_object* lean_torch_get_dtype(lean_object*, lean_object*);
lean_object* lean_torch_get_device_enum(lean_object*, lean_object*);
lean_object* lean_torch_get_device(lean_object*, lean_object*);
lean_object* lean_torch_get_values(lean_object*, lean_object*, uint64_t);
lean_object* lean_torch_get_stats(lean_object*, lean_object*);

lean_object* lean_torch_slice_along_dim(lean_object*, lean_object*, uint64_t, uint64_t, uint64_t);
lean_object* lean_torch_clamp_float(lean_object*, lean_object*, double, double);
lean_object* lean_torch_diag(uint64_t, lean_object*);
lean_object* lean_torch_matrix_exp(uint64_t, lean_object*);
lean_object* lean_torch_matrix_log(uint64_t, lean_object*);
lean_object* lean_torch_inv(uint64_t, lean_object*);
lean_object* lean_torch_eye(uint64_t, uint8_t);
lean_object* lean_torch_qr(uint64_t, uint64_t, lean_object*);
lean_object* lean_torch_qr_reduced(uint64_t, uint64_t, lean_object*);
lean_object* lean_torch_svd(uint64_t, uint64_t, lean_object*);
lean_object* lean_torch_svdvals(uint64_t, uint64_t, lean_object*);
lean_object* lean_torch_diagflat(uint64_t, lean_object*);
double lean_torch_spectral_norm(uint64_t, uint64_t, lean_object*);
double lean_torch_nuclear_norm(uint64_t, uint64_t, lean_object*);
lean_object* lean_torch_tensor_atan(lean_object*, lean_object*);
lean_object* lean_torch_max_all(lean_object*);
lean_object* lean_torch_stack_1d(uint64_t, uint64_t, lean_object*, uint64_t);
lean_object* lean_torch_tensor_grad(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_from_float_array(lean_object*);
lean_object* lean_torch_multinomial(lean_object*, lean_object*, uint64_t, uint8_t);
lean_object* lean_torch_scatter_2d(uint64_t, uint64_t, uint64_t, lean_object*, uint64_t, lean_object*, lean_object*);
uint8_t lean_torch_allclose(lean_object*, lean_object*, double, double);
uint8_t lean_torch_is_leaf(lean_object*);
lean_object* lean_torch_log_softmax(lean_object*, uint32_t);
lean_object* lean_torch_backward_loss(lean_object*);
double lean_torch_frobenius_norm(uint64_t, uint64_t, lean_object*);
lean_object* lean_torch_row_norms(uint64_t, uint64_t, lean_object*);
double lean_torch_max_row_norm(uint64_t, uint64_t, lean_object*);
lean_object* lean_torch_uniform_like(lean_object*, lean_object*, double, double);
lean_object* lean_torch_abs(lean_object*, lean_object*);
lean_object* lean_torch_to(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_cast_dtype(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_convert_tensor(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_torch_requires_grad(lean_object*, lean_object*);
double lean_torch_l2_norm(uint64_t, lean_object*);
double lean_torch_item(lean_object*, lean_object*);
lean_object* lean_torch_gt(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_where(lean_object*, lean_object*, lean_object*, lean_object*);
// grad_of: borrowed tensor -> IO detached snapshot; zero_grad: borrowed tensor -> IO Unit.
lean_object* lean_torch_grad_of(lean_object*, lean_object*);
lean_object* lean_torch_zero_grad(lean_object*, lean_object*);
lean_object* lean_torch_add_scalar(lean_object*, lean_object*, double);
lean_object* lean_torch_affine3d(uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_bmm4d(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*);
lean_object* lean_torch_cat2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_cross_entropy_2d(uint64_t, uint64_t, lean_object*, lean_object*);
lean_object* lean_torch_detach(lean_object*, lean_object*);
lean_object* lean_torch_div(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_div_scalar(lean_object*, lean_object*, double);
lean_object* lean_torch_dropout(lean_object*, lean_object*, double, uint8_t);
lean_object* lean_torch_embedding(uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*, lean_object*, lean_object*, double, uint8_t, uint8_t);
lean_object* lean_torch_embedding_1d(uint64_t, uint64_t, uint64_t, lean_object*, lean_object*, lean_object*, lean_object*, double, uint8_t, uint8_t);
lean_object* lean_torch_expand(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_full(lean_object*, double, uint8_t, lean_object*);
lean_object* lean_torch_full_int(lean_object*, uint64_t);
lean_object* lean_torch_layer_norm_3d(uint64_t, uint64_t, uint64_t, lean_object*, lean_object*, lean_object*, double);
lean_object* lean_torch_linear(uint64_t, uint64_t, uint64_t, lean_object*, lean_object*);
lean_object* lean_torch_linear3d(uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*);
lean_object* lean_torch_matmul(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_matmul3d_2d(uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*);
lean_object* lean_torch_mean(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* lean_torch_mm(uint64_t, uint64_t, uint64_t, lean_object*, lean_object*);
lean_object* lean_torch_mul_scalar(lean_object*, lean_object*, double);
lean_object* lean_torch_ones(lean_object*, uint8_t, lean_object*);
lean_object* lean_torch_ones_like(lean_object*, lean_object*);
lean_object* lean_torch_permute(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_pow(lean_object*, lean_object*, double);
lean_object* lean_torch_randn(lean_object*, uint8_t, lean_object*);
lean_object* lean_torch_reshape_exact(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_rsqrt(lean_object*, lean_object*);
lean_object* lean_torch_sdpa_4d(uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*, lean_object*, double, uint8_t);
// set_requires_grad consumes the tensor and returns its updated value.
lean_object* lean_torch_set_requires_grad(lean_object*, lean_object*, uint8_t);
lean_object* lean_torch_sqrt(lean_object*, lean_object*);
lean_object* lean_torch_squeeze(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_sub_scalar(lean_object*, lean_object*, double);
lean_object* lean_torch_sum(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* lean_torch_tensor_add(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_tensor_exp(lean_object*, lean_object*);
lean_object* lean_torch_tensor_gelu(lean_object*, lean_object*);
lean_object* lean_torch_tensor_mul(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_tensor_relu(lean_object*, lean_object*);
lean_object* lean_torch_tensor_sigmoid(lean_object*, lean_object*);
lean_object* lean_torch_tensor_silu(lean_object*, lean_object*);
lean_object* lean_torch_tensor_softmax(lean_object*, lean_object*, uint32_t);
lean_object* lean_torch_tensor_sub(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_tensor_tanh(lean_object*, lean_object*);
lean_object* lean_torch_to_bfloat16(lean_object*, lean_object*);
lean_object* lean_torch_to_float(lean_object*, lean_object*);
lean_object* lean_torch_to_long(lean_object*, lean_object*);
lean_object* lean_torch_transpose(lean_object*, lean_object*, uint64_t, uint64_t);
lean_object* lean_torch_transpose_2d(uint64_t, uint64_t, lean_object*);
lean_object* lean_torch_transpose_for_attention(uint64_t, uint64_t, uint64_t, uint64_t, lean_object*);
lean_object* lean_torch_transpose_from_attention(uint64_t, uint64_t, uint64_t, uint64_t, lean_object*);
lean_object* lean_torch_unsqueeze(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_zeros(lean_object*, uint8_t, lean_object*);
lean_object* lean_torch_zeros_like(lean_object*, lean_object*);
lean_object* lean_torch_argmax(lean_object*, lean_object*, uint64_t);
lean_object* lean_torch_topk_filter(lean_object*, lean_object*, uint64_t);
lean_object* lean_torch_topp_filter(lean_object*, lean_object*, double);
lean_object* lean_torch_eq_scalar(lean_object*, lean_object*, uint64_t);
lean_object* lean_torch_from_int64_array(lean_object*);
lean_object* lean_torch_index_select(uint64_t, uint64_t, lean_object*, uint64_t, lean_object*);
lean_object* lean_torch_cross_entropy_none(uint64_t, uint64_t, lean_object*, lean_object*);
uint64_t lean_torch_item_int(lean_object*, lean_object*);
lean_object* lean_torch_set_grad_enabled(uint8_t);
lean_object* lean_torch_is_grad_enabled();
lean_object* lean_torch_logical_not(lean_object*, lean_object*);
lean_object* lean_torch_logical_and(lean_object*, lean_object*, lean_object*);

lean_object* lean_torch_conv1d(lean_object*, lean_object*, lean_object*, lean_object*, uint64_t, uint64_t, uint64_t);
lean_object* lean_torch_tensor_to_float_array_dynamic(lean_object*);

uint8_t lean_torch_any(lean_object*, lean_object*);

lean_object* lean_torch_tensor_to_uint64_array_dynamic(lean_object*);

lean_object* lean_torch_randint(uint64_t, uint64_t, lean_object*, uint8_t, lean_object*);

#ifdef __cplusplus
}
#endif
