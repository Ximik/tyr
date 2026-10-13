#include "tyr_qwen35_abi.h"
#include "tyr_tensor.h"
#include <torch/torch.h>

using namespace torch::indexing;

static torch::Tensor qwen35_l2norm(const torch::Tensor& x, int64_t dim = -1, double eps = 1e-6) {
  auto inv_norm = torch::rsqrt((x * x).sum(dim, /*keepdim=*/true) + eps);
  return x * inv_norm;
}

static std::pair<torch::Tensor, torch::Tensor> qwen35_chunk_gated_delta_rule_impl(
    const torch::Tensor& query_in,
    const torch::Tensor& key_in,
    const torch::Tensor& value_in,
    const torch::Tensor& g_in,
    const torch::Tensor& beta_in,
    int64_t chunk_size = 64) {

  auto initial_dtype = query_in.dtype();
  auto query_norm = qwen35_l2norm(query_in, -1, 1e-6);
  auto key_norm = qwen35_l2norm(key_in, -1, 1e-6);

  auto query = query_norm.transpose(1, 2).contiguous().to(torch::kFloat32);
  auto key = key_norm.transpose(1, 2).contiguous().to(torch::kFloat32);
  auto value = value_in.transpose(1, 2).contiguous().to(torch::kFloat32);
  auto beta = beta_in.transpose(1, 2).contiguous().to(torch::kFloat32);
  auto g = g_in.transpose(1, 2).contiguous().to(torch::kFloat32);

  const int64_t batch_size = key.size(0);
  const int64_t num_heads = key.size(1);
  const int64_t sequence_length = key.size(2);
  const int64_t k_head_dim = key.size(3);
  const int64_t v_head_dim = value.size(3);
  const int64_t pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size;
  const int64_t total_sequence_length = sequence_length + pad_size;

  if (pad_size > 0) {
    query = torch::constant_pad_nd(query, {0, 0, 0, pad_size}, 0.0);
    key = torch::constant_pad_nd(key, {0, 0, 0, pad_size}, 0.0);
    value = torch::constant_pad_nd(value, {0, 0, 0, pad_size}, 0.0);
    beta = torch::constant_pad_nd(beta, {0, pad_size}, 0.0);
    g = torch::constant_pad_nd(g, {0, pad_size}, 0.0);
  }

  const double scale = 1.0 / std::sqrt(static_cast<double>(query.size(-1)));
  query = query * scale;

  auto v_beta = value * beta.unsqueeze(-1);
  auto k_beta = key * beta.unsqueeze(-1);

  const int64_t num_chunks = total_sequence_length / chunk_size;
  query = query.reshape({batch_size, num_heads, num_chunks, chunk_size, k_head_dim});
  key = key.reshape({batch_size, num_heads, num_chunks, chunk_size, k_head_dim});
  value = value.reshape({batch_size, num_heads, num_chunks, chunk_size, v_head_dim});
  k_beta = k_beta.reshape({batch_size, num_heads, num_chunks, chunk_size, k_head_dim});
  v_beta = v_beta.reshape({batch_size, num_heads, num_chunks, chunk_size, v_head_dim});
  g = g.reshape({batch_size, num_heads, num_chunks, chunk_size});

  auto strict_upper_with_diag =
      torch::triu(torch::ones({chunk_size, chunk_size}, torch::TensorOptions().dtype(torch::kBool).device(query.device())), 0);
  auto strict_upper =
      torch::triu(torch::ones({chunk_size, chunk_size}, torch::TensorOptions().dtype(torch::kBool).device(query.device())), 1);

  g = g.cumsum(-1);
  auto decay_mask =
      torch::tril(g.unsqueeze(-1) - g.unsqueeze(-2)).exp().to(torch::kFloat32);

  auto attn =
      -(torch::matmul(k_beta, key.transpose(-1, -2)) * decay_mask)
           .masked_fill(strict_upper_with_diag.view({1, 1, 1, chunk_size, chunk_size}), 0.0);
  for (int64_t i = 1; i < chunk_size; ++i) {
    auto row = attn.index({Slice(), Slice(), Slice(), i, Slice(None, i)}).clone();
    auto sub = attn.index({Slice(), Slice(), Slice(), Slice(None, i), Slice(None, i)}).clone();
    auto updated = row + (row.unsqueeze(-1) * sub).sum(-2);
    attn.index_put_({Slice(), Slice(), Slice(), i, Slice(None, i)}, updated);
  }
  auto eye = torch::eye(chunk_size, torch::TensorOptions().dtype(attn.dtype()).device(attn.device()))
                 .view({1, 1, 1, chunk_size, chunk_size});
  attn = attn + eye;

  value = torch::matmul(attn, v_beta);
  auto k_cumdecay = torch::matmul(attn, k_beta * g.exp().unsqueeze(-1));

  auto last_recurrent_state =
      torch::zeros({batch_size, num_heads, k_head_dim, v_head_dim}, value.options());
  auto core_attn_out = torch::zeros_like(value);

  for (int64_t i = 0; i < num_chunks; ++i) {
    auto q_i = query.index({Slice(), Slice(), i});
    auto k_i = key.index({Slice(), Slice(), i});
    auto v_i = value.index({Slice(), Slice(), i});
    auto g_i = g.index({Slice(), Slice(), i});
    auto decay_i = decay_mask.index({Slice(), Slice(), i});

    auto intra_attn =
        (torch::matmul(q_i, k_i.transpose(-1, -2)) * decay_i)
            .masked_fill(strict_upper.view({1, 1, chunk_size, chunk_size}), 0.0);
    auto v_prime = torch::matmul(k_cumdecay.index({Slice(), Slice(), i}), last_recurrent_state);
    auto v_new = v_i - v_prime;
    auto attn_inter = torch::matmul(q_i * g_i.exp().unsqueeze(-1), last_recurrent_state);
    core_attn_out.index_put_({Slice(), Slice(), i}, attn_inter + torch::matmul(intra_attn, v_new));

    auto g_last = g_i.index({Slice(), Slice(), chunk_size - 1}).unsqueeze(-1);
    auto decayed_k =
        k_i * (g_last - g_i).exp().unsqueeze(-1);
    last_recurrent_state =
        last_recurrent_state * g_last.unsqueeze(-1).exp() +
        torch::matmul(decayed_k.transpose(-1, -2), v_new);
  }

  core_attn_out =
      core_attn_out.reshape({batch_size, num_heads, total_sequence_length, v_head_dim})
          .slice(2, 0, sequence_length)
          .transpose(1, 2)
          .contiguous()
          .to(initial_dtype);
  return {core_attn_out, last_recurrent_state};
}

static std::pair<torch::Tensor, torch::Tensor> qwen35_recurrent_gated_delta_rule_impl(
    const torch::Tensor& query_in,
    const torch::Tensor& key_in,
    const torch::Tensor& value_in,
    const torch::Tensor& g_in,
    const torch::Tensor& beta_in,
    const torch::Tensor& initial_state_in) {

  auto initial_dtype = query_in.dtype();
  auto query_norm = qwen35_l2norm(query_in, -1, 1e-6);
  auto key_norm = qwen35_l2norm(key_in, -1, 1e-6);

  auto query = query_norm.transpose(1, 2).contiguous().to(torch::kFloat32);
  auto key = key_norm.transpose(1, 2).contiguous().to(torch::kFloat32);
  auto value = value_in.transpose(1, 2).contiguous().to(torch::kFloat32);
  auto beta = beta_in.transpose(1, 2).contiguous().to(torch::kFloat32);
  auto g = g_in.transpose(1, 2).contiguous().to(torch::kFloat32);

  const int64_t batch_size = key.size(0);
  const int64_t num_heads = key.size(1);
  const int64_t sequence_length = key.size(2);
  const int64_t v_head_dim = value.size(3);

  const double scale = 1.0 / std::sqrt(static_cast<double>(query.size(-1)));
  query = query * scale;

  auto core_attn_out =
      torch::zeros({batch_size, num_heads, sequence_length, v_head_dim}, value.options());
  auto last_recurrent_state = initial_state_in.to(value);

  for (int64_t i = 0; i < sequence_length; ++i) {
    auto q_t = query.index({Slice(), Slice(), i});
    auto k_t = key.index({Slice(), Slice(), i});
    auto v_t = value.index({Slice(), Slice(), i});
    auto g_t = g.index({Slice(), Slice(), i}).exp().unsqueeze(-1).unsqueeze(-1);
    auto beta_t = beta.index({Slice(), Slice(), i}).unsqueeze(-1);

    last_recurrent_state = last_recurrent_state * g_t;
    auto kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(-2);
    auto delta = (v_t - kv_mem) * beta_t;
    last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2);
    core_attn_out.index_put_(
        {Slice(), Slice(), i},
        (last_recurrent_state * q_t.unsqueeze(-1)).sum(-2));
  }

  core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype);
  return {core_attn_out, last_recurrent_state};
}

extern "C" {

lean_object* lean_torch_qwen35_chunk_gated_delta_rule(
    uint64_t /*batch*/,
    uint64_t /*seq*/,
    uint64_t /*n_head*/,
    uint64_t /*kdim*/,
    uint64_t /*vdim*/,
    b_lean_obj_arg query,
    b_lean_obj_arg key,
    b_lean_obj_arg value,
    b_lean_obj_arg g,
    b_lean_obj_arg beta) {
  auto query_ = borrowTensor(query);
  auto key_ = borrowTensor(key);
  auto value_ = borrowTensor(value);
  auto g_ = borrowTensor(g);
  auto beta_ = borrowTensor(beta);
  auto [out, state] = qwen35_chunk_gated_delta_rule_impl(query_, key_, value_, g_, beta_);
  lean_object* result = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(result, 0, fromTorchTensor(out));
  lean_ctor_set(result, 1, fromTorchTensor(state));
  return result;
}

lean_object* lean_torch_qwen35_recurrent_gated_delta_rule(
    uint64_t /*batch*/,
    uint64_t /*seq*/,
    uint64_t /*n_head*/,
    uint64_t /*kdim*/,
    uint64_t /*vdim*/,
    b_lean_obj_arg query,
    b_lean_obj_arg key,
    b_lean_obj_arg value,
    b_lean_obj_arg g,
    b_lean_obj_arg beta,
    b_lean_obj_arg initial_state) {
  auto query_ = borrowTensor(query);
  auto key_ = borrowTensor(key);
  auto value_ = borrowTensor(value);
  auto g_ = borrowTensor(g);
  auto beta_ = borrowTensor(beta);
  auto initial_state_ = borrowTensor(initial_state);
  auto [out, state] =
      qwen35_recurrent_gated_delta_rule_impl(query_, key_, value_, g_, beta_, initial_state_);
  lean_object* result = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(result, 0, fromTorchTensor(out));
  lean_ctor_set(result, 1, fromTorchTensor(state));
  return result;
}

} // extern "C"
