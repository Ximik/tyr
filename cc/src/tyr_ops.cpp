#include "tyr_neural_abi.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <lean/lean.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include "tyr_attention.h"
#include "tyr_checked_result.h"

#ifndef TYR_OPS_HAS_CUDA_TOOLKIT
#define TYR_OPS_HAS_CUDA_TOOLKIT 0
#endif

#if TYR_OPS_HAS_CUDA_TOOLKIT
#if defined(__has_include)
#if __has_include(<c10/cuda/CUDAStream.h>) && __has_include(<c10/cuda/CUDAFunctions.h>)
#define TYR_OPS_HAS_CUDA_STREAM 1
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAFunctions.h>
#include <ATen/cuda/CUDAContext.h>
#else
#define TYR_OPS_HAS_CUDA_STREAM 0
#endif
#else
#define TYR_OPS_HAS_CUDA_STREAM 0
#endif
#else
#define TYR_OPS_HAS_CUDA_STREAM 0
#endif

namespace tyr_ops {
// The TK H100 attention/decode kernels emit `wgmma.*` instructions which are
// sm_90a-only — they fail to JIT on Blackwell devices (e.g. GB10/B200,
// compute cap 10/12). Gate the native routes on Hopper.
static bool device_supports_tk_hopper(const torch::Tensor& t) {
#if TYR_OPS_HAS_CUDA_STREAM
  if (!t.is_cuda()) return false;
  auto* props = at::cuda::getDeviceProperties(t.get_device());
  return props != nullptr && props->major == 9;
#else
  (void)t;
  return false;
#endif
}
} // namespace tyr_ops

// Shared tensor/Lean interop helpers defined in tyr.cpp.
torch::Tensor borrowTensor(b_lean_obj_arg o);
lean_object *giveTensor(torch::Tensor t);
lean_object *fromTorchTensor(torch::Tensor t);

extern "C" lean_object* lean_launch_Tyr_GPU_Kernels_tkMhaH100Fwd2Block(
    b_lean_obj_arg q_ptr, b_lean_obj_arg k_ptr, b_lean_obj_arg v_ptr,
    b_lean_obj_arg o_ptr, b_lean_obj_arg l_ptr, uint64_t seq_len, uint64_t head_dim,
    uint64_t grid_x, uint64_t grid_y, uint64_t grid_z,
    uint64_t block_x, uint64_t block_y, uint64_t block_z,
    uint64_t shared_mem, uint64_t stream);
extern "C" lean_object* lean_launch_Tyr_GPU_Kernels_tkMhaH100Fwd12Block(
    b_lean_obj_arg q_ptr, b_lean_obj_arg k_ptr, b_lean_obj_arg v_ptr,
    b_lean_obj_arg o_ptr, b_lean_obj_arg l_ptr, uint64_t seq_len, uint64_t head_dim,
    uint64_t grid_x, uint64_t grid_y, uint64_t grid_z,
    uint64_t block_x, uint64_t block_y, uint64_t block_z,
    uint64_t shared_mem, uint64_t stream);
extern "C" lean_object* lean_launch_Tyr_GPU_Kernels_tkMhaH100DecodeFwd(
    b_lean_obj_arg q_ptr, b_lean_obj_arg k_ptr, b_lean_obj_arg v_ptr,
    b_lean_obj_arg o_ptr,
    uint64_t batch, uint64_t q_heads, uint64_t kv_heads,
    uint64_t kv_seq, uint64_t head_dim,
    uint64_t grid_x, uint64_t grid_y, uint64_t grid_z,
    uint64_t block_x, uint64_t block_y, uint64_t block_z,
    uint64_t shared_mem, uint64_t stream);
extern "C" lean_object* lean_launch_Tyr_GPU_Kernels_tkMhaH100DecodeFwd64(
    b_lean_obj_arg q_ptr, b_lean_obj_arg k_ptr, b_lean_obj_arg v_ptr,
    b_lean_obj_arg o_ptr,
    uint64_t batch, uint64_t q_heads, uint64_t kv_heads,
    uint64_t kv_seq, uint64_t head_dim,
    uint64_t grid_x, uint64_t grid_y, uint64_t grid_z,
    uint64_t block_x, uint64_t block_y, uint64_t block_z,
    uint64_t shared_mem, uint64_t stream);
extern "C" lean_object* lean_launch_Tyr_GPU_Kernels_tkMhaH100DecodeFwd256(
    b_lean_obj_arg q_ptr, b_lean_obj_arg k_ptr, b_lean_obj_arg v_ptr,
    b_lean_obj_arg o_ptr,
    uint64_t batch, uint64_t q_heads, uint64_t kv_heads,
    uint64_t kv_seq, uint64_t head_dim,
    uint64_t grid_x, uint64_t grid_y, uint64_t grid_z,
    uint64_t block_x, uint64_t block_y, uint64_t block_z,
    uint64_t shared_mem, uint64_t stream);

// V2 D03 — GQA-packed decode forward kernels. Pack
// `R = q_heads / kv_heads` query heads into one CTA, so K/V is loaded
// once per group instead of R times. Grid: `batch * kv_heads`.
// Eligible when R ∈ {2, 4, 8, 16} (so R | 64).
extern "C" lean_object* lean_launch_Tyr_GPU_Kernels_tkMhaH100DecodeFwdGqa(
    b_lean_obj_arg q_ptr, b_lean_obj_arg k_ptr, b_lean_obj_arg v_ptr,
    b_lean_obj_arg o_ptr,
    uint64_t batch, uint64_t q_heads, uint64_t kv_heads,
    uint64_t kv_seq, uint64_t head_dim,
    uint64_t grid_x, uint64_t grid_y, uint64_t grid_z,
    uint64_t block_x, uint64_t block_y, uint64_t block_z,
    uint64_t shared_mem, uint64_t stream);
extern "C" lean_object* lean_launch_Tyr_GPU_Kernels_tkMhaH100DecodeFwdGqa64(
    b_lean_obj_arg q_ptr, b_lean_obj_arg k_ptr, b_lean_obj_arg v_ptr,
    b_lean_obj_arg o_ptr,
    uint64_t batch, uint64_t q_heads, uint64_t kv_heads,
    uint64_t kv_seq, uint64_t head_dim,
    uint64_t grid_x, uint64_t grid_y, uint64_t grid_z,
    uint64_t block_x, uint64_t block_y, uint64_t block_z,
    uint64_t shared_mem, uint64_t stream);
extern "C" lean_object* lean_launch_Tyr_GPU_Kernels_tkMhaH100DecodeFwdGqa256(
    b_lean_obj_arg q_ptr, b_lean_obj_arg k_ptr, b_lean_obj_arg v_ptr,
    b_lean_obj_arg o_ptr,
    uint64_t batch, uint64_t q_heads, uint64_t kv_heads,
    uint64_t kv_seq, uint64_t head_dim,
    uint64_t grid_x, uint64_t grid_y, uint64_t grid_z,
    uint64_t block_x, uint64_t block_y, uint64_t block_z,
    uint64_t shared_mem, uint64_t stream);
extern "C" lean_object* lean_launch_Tyr_GPU_Kernels_tkMhaH100BwdPrep2Block(
    b_lean_obj_arg dO_ptr, b_lean_obj_arg o_ptr, b_lean_obj_arg d_ptr,
    uint64_t seq_len, uint64_t head_dim,
    uint64_t grid_x, uint64_t grid_y, uint64_t grid_z,
    uint64_t block_x, uint64_t block_y, uint64_t block_z,
    uint64_t shared_mem, uint64_t stream);
extern "C" lean_object* lean_launch_Tyr_GPU_Kernels_tkMhaH100Bwd2BlockPartials(
    b_lean_obj_arg q_ptr, b_lean_obj_arg k_ptr, b_lean_obj_arg v_ptr,
    b_lean_obj_arg dO_ptr, b_lean_obj_arg l_ptr, b_lean_obj_arg d_ptr,
    b_lean_obj_arg dQ_ptr, b_lean_obj_arg dK_part_ptr, b_lean_obj_arg dV_part_ptr,
    uint64_t seq_len, uint64_t head_dim,
    uint64_t grid_x, uint64_t grid_y, uint64_t grid_z,
    uint64_t block_x, uint64_t block_y, uint64_t block_z,
    uint64_t shared_mem, uint64_t stream);
extern "C" lean_object* lean_launch_Tyr_GPU_Kernels_tkMhaH100Bwd12BlockPartials(
    b_lean_obj_arg q_ptr, b_lean_obj_arg k_ptr, b_lean_obj_arg v_ptr,
    b_lean_obj_arg dO_ptr, b_lean_obj_arg l_ptr, b_lean_obj_arg d_ptr,
    b_lean_obj_arg dQ_ptr, b_lean_obj_arg dK_part_ptr, b_lean_obj_arg dV_part_ptr,
    uint64_t seq_len, uint64_t head_dim,
    uint64_t grid_x, uint64_t grid_y, uint64_t grid_z,
    uint64_t block_x, uint64_t block_y, uint64_t block_z,
    uint64_t shared_mem, uint64_t stream);
extern "C" lean_object* lean_launch_Tyr_GPU_Kernels_tkMhaH100Bwd2BlockDq(
    b_lean_obj_arg q_ptr, b_lean_obj_arg k_ptr, b_lean_obj_arg v_ptr,
    b_lean_obj_arg dO_ptr, b_lean_obj_arg l_ptr, b_lean_obj_arg d_ptr,
    b_lean_obj_arg dQ_ptr,
    uint64_t seq_len, uint64_t head_dim,
    uint64_t grid_x, uint64_t grid_y, uint64_t grid_z,
    uint64_t block_x, uint64_t block_y, uint64_t block_z,
    uint64_t shared_mem, uint64_t stream);
extern "C" lean_object* lean_launch_Tyr_GPU_Kernels_tkMhaH100Bwd12BlockDq(
    b_lean_obj_arg q_ptr, b_lean_obj_arg k_ptr, b_lean_obj_arg v_ptr,
    b_lean_obj_arg dO_ptr, b_lean_obj_arg l_ptr, b_lean_obj_arg d_ptr,
    b_lean_obj_arg dQ_ptr,
    uint64_t seq_len, uint64_t head_dim,
    uint64_t grid_x, uint64_t grid_y, uint64_t grid_z,
    uint64_t block_x, uint64_t block_y, uint64_t block_z,
    uint64_t shared_mem, uint64_t stream);
extern "C" lean_object* lean_launch_Tyr_GPU_Kernels_tkMhaH100Bwd2BlockKvSweep(
    b_lean_obj_arg q_ptr, b_lean_obj_arg k_ptr, b_lean_obj_arg v_ptr,
    b_lean_obj_arg dO_ptr, b_lean_obj_arg l_ptr, b_lean_obj_arg d_ptr,
    b_lean_obj_arg dQ_ptr, b_lean_obj_arg dK_ptr, b_lean_obj_arg dV_ptr,
    uint64_t seq_len, uint64_t head_dim,
    uint64_t grid_x, uint64_t grid_y, uint64_t grid_z,
    uint64_t block_x, uint64_t block_y, uint64_t block_z,
    uint64_t shared_mem, uint64_t stream);
extern "C" lean_object* lean_launch_Tyr_GPU_Kernels_tkMhaH100Bwd12BlockKvSweep(
    b_lean_obj_arg q_ptr, b_lean_obj_arg k_ptr, b_lean_obj_arg v_ptr,
    b_lean_obj_arg dO_ptr, b_lean_obj_arg l_ptr, b_lean_obj_arg d_ptr,
    b_lean_obj_arg dQ_ptr, b_lean_obj_arg dK_ptr, b_lean_obj_arg dV_ptr,
    uint64_t seq_len, uint64_t head_dim,
    uint64_t grid_x, uint64_t grid_y, uint64_t grid_z,
    uint64_t block_x, uint64_t block_y, uint64_t block_z,
    uint64_t shared_mem, uint64_t stream);

std::vector<at::Tensor> tyr_tk_attention_forward_nosync(
    at::Tensor q, at::Tensor k, at::Tensor v, bool causal);
std::vector<at::Tensor> tyr_tk_attention_backward_nosync(
    at::Tensor q, at::Tensor k, at::Tensor v,
    at::Tensor o, at::Tensor l_vec, at::Tensor og, bool causal);

namespace tyr_ops {

enum class FlashAttnRoute {
  Portable,
  TkMhaH100Decode,
  TkMhaH1002Block,
  TkMhaH10012Block,
};

enum class FlashAttnImpl {
  Generated,
  VendoredTk,
};

struct LaunchConfig {
  uint64_t grid_x{1};
  uint64_t grid_y{1};
  uint64_t grid_z{1};
  uint64_t block_x{128};
  uint64_t block_y{1};
  uint64_t block_z{1};
  uint64_t shared_mem{0};
  uint64_t stream{0};
};

struct LeanTensorRef {
  lean_object* obj;
  explicit LeanTensorRef(const torch::Tensor& t) : obj(giveTensor(t)) {}
  ~LeanTensorRef() {
    if (obj != nullptr) {
      lean_dec(obj);
    }
  }
  LeanTensorRef(const LeanTensorRef&) = delete;
  LeanTensorRef& operator=(const LeanTensorRef&) = delete;
};

static inline double default_scale_for_dim(int64_t head_dim) {
  return 1.0 / std::sqrt(static_cast<double>(head_dim));
}

static inline bool scale_matches_default(
    int64_t head_dim,
    const c10::optional<double>& scale) {
  if (!scale.has_value()) {
    return true;
  }
  return std::abs(scale.value() - default_scale_for_dim(head_dim)) <= 1.0e-6;
}

static inline bool native_decode_autograd_safe(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value) {
  return !(query.requires_grad() || key.requires_grad() || value.requires_grad());
}

static inline bool valid_gqa_heads(int64_t q_heads, int64_t kv_heads, bool enable_gqa) {
  if (q_heads <= 0 || kv_heads <= 0) {
    return false;
  }
  if (q_heads == kv_heads) {
    return true;
  }
  return enable_gqa && q_heads % kv_heads == 0;
}

static uint64_t generated_decode_shared_mem(
    int64_t head_dim,
    uint64_t block_threads = 256) {
  return (block_threads + 2 * static_cast<uint64_t>(head_dim)) * sizeof(float);
}

// Shared-memory budget for the TK-style tile-based decode kernel.
// Tiles: qShared/kShared/vShared/oShared at 64x128 BF16 each = 4 * 64 * 128 * 2 = 64 KiB,
// plus a small overhead for semaphores and tile metadata. Held well under the
// generated_forward_shared_mem() ceiling.
static uint64_t generated_decode_tk_shared_mem(int64_t head_dim) {
  const uint64_t bf16_bytes = sizeof(uint16_t);
  const uint64_t tile_m = 64;
  const uint64_t tile_n = 64;
  const uint64_t hdim = static_cast<uint64_t>(head_dim);
  const uint64_t tiles_bytes =
      (tile_m * hdim + tile_n * hdim + tile_n * hdim + tile_m * hdim) * bf16_bytes;
  // 4 KiB headroom for semaphores, padding, and TMA descriptors.
  return tiles_bytes + 4096;
}

static uint64_t generated_forward_shared_mem();

static void check_flash_attn_args(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const c10::optional<torch::Tensor>& attn_mask) {
  TORCH_CHECK(query.dim() == 4, "tyr::flash_attn: query must be rank-4 [B, Hq, Q, D]");
  TORCH_CHECK(key.dim() == 4, "tyr::flash_attn: key must be rank-4 [B, Hkv, K, D]");
  TORCH_CHECK(value.dim() == 4, "tyr::flash_attn: value must be rank-4 [B, Hkv, K, D]");
  TORCH_CHECK(query.device() == key.device() && key.device() == value.device(),
    "tyr::flash_attn: Q/K/V must be on the same device");
  TORCH_CHECK(query.scalar_type() == key.scalar_type() && key.scalar_type() == value.scalar_type(),
    "tyr::flash_attn: Q/K/V must have the same dtype");
  TORCH_CHECK(query.size(0) == key.size(0) && key.size(0) == value.size(0),
    "tyr::flash_attn: batch mismatch between Q/K/V");
  TORCH_CHECK(key.size(1) == value.size(1),
    "tyr::flash_attn: KV-head mismatch between K and V");
  TORCH_CHECK(key.size(2) == value.size(2),
    "tyr::flash_attn: KV-sequence mismatch between K and V");
  TORCH_CHECK(query.size(3) == key.size(3) && key.size(3) == value.size(3),
    "tyr::flash_attn: head-dimension mismatch between Q/K/V");
  if (attn_mask.has_value() && attn_mask->defined()) {
    TORCH_CHECK(attn_mask->dim() == 2,
      "tyr::flash_attn: attn_mask must be rank-2 [batch, kv_seq]");
    TORCH_CHECK(attn_mask->size(0) == query.size(0) && attn_mask->size(1) == key.size(2),
      "tyr::flash_attn: attn_mask shape must be [batch, kv_seq]");
  }
}

static inline FlashAttnRoute select_route(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const c10::optional<torch::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    const c10::optional<double>& scale,
    bool enable_gqa) {
  const bool mask_ok = !(attn_mask.has_value() && attn_mask->defined());
  const bool device_ok = query.is_cuda() && device_supports_tk_hopper(query);
  const bool dtype_ok = query.scalar_type() == torch::kBFloat16;
  // V1 of the TK-style decode kernel supports head_dim ∈ {64, 128, 256} and
  // any positive KV sequence length; the kernel iterates ceil(kv_seq/64)
  // blocks and applies a runtime tail mask (TK `right_fill`) on the last
  // block. head_dim=256 covers the Qwen 3.5/3.6 family and Gemma-2 27B.
  const bool decode_shape_ok =
      query.size(2) == 1 &&
      key.size(2) == value.size(2) &&
      key.size(2) > 0 &&
      query.size(3) == key.size(3) &&
      (query.size(3) == 128 || query.size(3) == 64 || query.size(3) == 256) &&
      valid_gqa_heads(query.size(1), key.size(1), enable_gqa);
  const bool decode_semantics_ok =
      mask_ok &&
      dropout_p == 0.0 &&
      !is_causal &&
      scale_matches_default(query.size(3), scale) &&
      native_decode_autograd_safe(query, key, value) &&
      generated_decode_tk_shared_mem(query.size(3)) <= generated_forward_shared_mem();
  if (device_ok && dtype_ok && decode_shape_ok && decode_semantics_ok) {
    return FlashAttnRoute::TkMhaH100Decode;
  }

  const bool shape_ok =
      query.size(0) == 1 &&
      query.size(1) == 1 &&
      key.size(1) == 1 &&
      value.size(1) == 1 &&
      query.size(2) == key.size(2) &&
      query.size(2) == value.size(2) &&
      query.size(3) == 64;
  const bool semantics_ok =
      mask_ok &&
      dropout_p == 0.0 &&
      !is_causal &&
      !enable_gqa &&
      scale_matches_default(query.size(3), scale);
  if (!(device_ok && dtype_ok && shape_ok && semantics_ok)) {
    return FlashAttnRoute::Portable;
  }
  if (query.size(2) == 128) {
    return FlashAttnRoute::TkMhaH1002Block;
  }
  if (query.size(2) == 768) {
    return FlashAttnRoute::TkMhaH10012Block;
  }
  return FlashAttnRoute::Portable;
}

static inline uint64_t current_stream_handle(const torch::Tensor& t) {
#if TYR_OPS_HAS_CUDA_STREAM
  if (t.is_cuda()) {
    return reinterpret_cast<uint64_t>(c10::cuda::getCurrentCUDAStream(t.get_device()).stream());
  }
#endif
  return 0;
}

static void throw_on_launcher_error(lean_object* io_result, const char* launcher) {
  if (lean_io_result_is_error(io_result)) {
    std::cerr << "tyr::flash_attn: native launcher failed: " << launcher << std::endl;
    lean_io_result_show_error(io_result);
    lean_dec(io_result);
    throw std::runtime_error(std::string("tyr::flash_attn: native launcher failed: ") + launcher);
  }
  lean_dec(io_result);
}

static LaunchConfig launch_config_for(const torch::Tensor& query, FlashAttnRoute route) {
  LaunchConfig cfg;
  if (route == FlashAttnRoute::TkMhaH10012Block) {
    cfg.grid_x = static_cast<uint64_t>(query.size(2) / (3 * 64));
    cfg.grid_y = static_cast<uint64_t>(query.size(1));
    cfg.block_x = 512;
  } else {
    cfg.grid_y = 2;
  }
  cfg.stream = current_stream_handle(query);
  return cfg;
}

static int64_t kv_blocks_for(FlashAttnRoute route) {
  return route == FlashAttnRoute::TkMhaH10012Block ? 12 : 2;
}

static uint64_t generated_forward_shared_mem() {
  return 227 * 1024 - 1024;
}

static uint64_t generated_backward_prep_shared_mem() {
  return 227 * 1024 - 1024;
}

static uint64_t generated_backward_sweep_shared_mem() {
  return 117760;
}

static std::pair<torch::Tensor, torch::Tensor> generated_forward(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    FlashAttnRoute route) {
  const auto q = query.contiguous();
  const auto k = key.contiguous();
  const auto v = value.contiguous();
  auto out = torch::empty_like(q);
  auto l = torch::empty(
      {1, 1, 1, q.size(2)},
      q.options().dtype(torch::kFloat32));

  LeanTensorRef q_ref(q);
  LeanTensorRef k_ref(k);
  LeanTensorRef v_ref(v);
  LeanTensorRef out_ref(out);
  LeanTensorRef l_ref(l);

  auto cfg = launch_config_for(q, route);
  cfg.shared_mem = generated_forward_shared_mem();
  lean_object* result = nullptr;
  if (route == FlashAttnRoute::TkMhaH1002Block) {
    result = lean_launch_Tyr_GPU_Kernels_tkMhaH100Fwd2Block(
        q_ref.obj, k_ref.obj, v_ref.obj, out_ref.obj, l_ref.obj,
        static_cast<uint64_t>(q.size(2)), static_cast<uint64_t>(q.size(3)),
        cfg.grid_x, cfg.grid_y, cfg.grid_z,
        cfg.block_x, cfg.block_y, cfg.block_z,
        cfg.shared_mem, cfg.stream);
    throw_on_launcher_error(result, "tkMhaH100Fwd2Block");
  } else {
    result = lean_launch_Tyr_GPU_Kernels_tkMhaH100Fwd12Block(
        q_ref.obj, k_ref.obj, v_ref.obj, out_ref.obj, l_ref.obj,
        static_cast<uint64_t>(q.size(2)), static_cast<uint64_t>(q.size(3)),
        cfg.grid_x, cfg.grid_y, cfg.grid_z,
        cfg.block_x, cfg.block_y, cfg.block_z,
        cfg.shared_mem, cfg.stream);
    throw_on_launcher_error(result, "tkMhaH100Fwd12Block");
  }

  return {out, l};
}

static torch::Tensor generated_decode_forward(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value) {
  const auto q = query.contiguous();
  const auto k = key.contiguous();
  const auto v = value.contiguous();
  auto out = torch::empty_like(q);

  LeanTensorRef q_ref(q);
  LeanTensorRef k_ref(k);
  LeanTensorRef v_ref(v);
  LeanTensorRef out_ref(out);

  LaunchConfig cfg;
  cfg.block_x = 128;
  cfg.shared_mem = generated_decode_tk_shared_mem(q.size(3));
  cfg.stream = current_stream_handle(q);

  const int64_t q_heads = q.size(1);
  const int64_t kv_heads = k.size(1);
  const int64_t head_dim = q.size(3);
  // V2 D03: when R = q_heads / kv_heads ∈ {2, 4, 8, 16}, use the
  // GQA-packed kernel (one CTA per kv_head, R query heads packed into
  // one Q tile). Drops K/V bandwidth by Rx vs V1; falls back to V1
  // for ratio=1 or other ratios.
  const bool gqa_packed_eligible =
      kv_heads > 0 &&
      q_heads % kv_heads == 0 &&
      ((q_heads / kv_heads) == 2 ||
       (q_heads / kv_heads) == 4 ||
       (q_heads / kv_heads) == 8 ||
       (q_heads / kv_heads) == 16);

  cfg.grid_x = static_cast<uint64_t>(
      q.size(0) * (gqa_packed_eligible ? kv_heads : q_heads));

  auto launch = [&](auto fn, const char* name) {
    auto result = fn(
        q_ref.obj, k_ref.obj, v_ref.obj, out_ref.obj,
        static_cast<uint64_t>(q.size(0)),
        static_cast<uint64_t>(q.size(1)),
        static_cast<uint64_t>(k.size(1)),
        static_cast<uint64_t>(k.size(2)),
        static_cast<uint64_t>(q.size(3)),
        cfg.grid_x, cfg.grid_y, cfg.grid_z,
        cfg.block_x, cfg.block_y, cfg.block_z,
        cfg.shared_mem, cfg.stream);
    throw_on_launcher_error(result, name);
  };
  if (gqa_packed_eligible) {
    if (head_dim == 64) {
      launch(lean_launch_Tyr_GPU_Kernels_tkMhaH100DecodeFwdGqa64,
             "tkMhaH100DecodeFwdGqa64");
    } else if (head_dim == 256) {
      launch(lean_launch_Tyr_GPU_Kernels_tkMhaH100DecodeFwdGqa256,
             "tkMhaH100DecodeFwdGqa256");
    } else {
      launch(lean_launch_Tyr_GPU_Kernels_tkMhaH100DecodeFwdGqa,
             "tkMhaH100DecodeFwdGqa");
    }
  } else if (head_dim == 64) {
    launch(lean_launch_Tyr_GPU_Kernels_tkMhaH100DecodeFwd64,
           "tkMhaH100DecodeFwd64");
  } else if (head_dim == 256) {
    launch(lean_launch_Tyr_GPU_Kernels_tkMhaH100DecodeFwd256,
           "tkMhaH100DecodeFwd256");
  } else {
    launch(lean_launch_Tyr_GPU_Kernels_tkMhaH100DecodeFwd,
           "tkMhaH100DecodeFwd");
  }
  return out;
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> generated_backward(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& out,
    const torch::Tensor& l,
    const torch::Tensor& grad_out,
    FlashAttnRoute route) {
  const auto q = query.contiguous();
  const auto k = key.contiguous();
  const auto v = value.contiguous();
  const auto o = out.contiguous();
  const auto l_saved = l.contiguous();
  const auto dO = grad_out.contiguous();

  const int64_t seq = q.size(2);
  const int64_t kv_blocks = kv_blocks_for(route);
  const auto f32_opts = q.options().dtype(torch::kFloat32);

  auto dVec = torch::empty({1, 1, 1, seq}, f32_opts);
  auto dQ = torch::zeros({1, 1, seq, 64}, f32_opts);
  auto dK = torch::zeros({1, 1, seq, 64}, f32_opts);
  auto dV = torch::zeros({1, 1, seq, 64}, f32_opts);

  LeanTensorRef dO_ref(dO);
  LeanTensorRef o_ref(o);
  LeanTensorRef d_ref(dVec);

  auto cfg = launch_config_for(q, route);
  auto prep_cfg = cfg;
  prep_cfg.grid_x = 1;
  prep_cfg.grid_y = static_cast<uint64_t>(seq / 64);
  prep_cfg.block_x = 128;
  prep_cfg.shared_mem = generated_backward_prep_shared_mem();
  auto prep_result = lean_launch_Tyr_GPU_Kernels_tkMhaH100BwdPrep2Block(
      dO_ref.obj, o_ref.obj, d_ref.obj,
      static_cast<uint64_t>(seq), static_cast<uint64_t>(q.size(3)),
      prep_cfg.grid_x, prep_cfg.grid_y, prep_cfg.grid_z,
      prep_cfg.block_x, prep_cfg.block_y, prep_cfg.block_z,
      prep_cfg.shared_mem, prep_cfg.stream);
  auto bwd_cfg = cfg;
  bwd_cfg.grid_x = kv_blocks / 2;
  bwd_cfg.grid_y = 1;
  bwd_cfg.block_x = 384;
  bwd_cfg.shared_mem = generated_backward_sweep_shared_mem();
  throw_on_launcher_error(prep_result, "tkMhaH100BwdPrep2Block");

  LeanTensorRef q_ref(q);
  LeanTensorRef k_ref(k);
  LeanTensorRef v_ref(v);
  LeanTensorRef l_ref(l_saved);
  LeanTensorRef dQ_ref(dQ);
  LeanTensorRef dK_ref(dK);
  LeanTensorRef dV_ref(dV);

  lean_object* bwd_result = nullptr;
  if (route == FlashAttnRoute::TkMhaH1002Block) {
    bwd_result = lean_launch_Tyr_GPU_Kernels_tkMhaH100Bwd2BlockKvSweep(
        q_ref.obj, k_ref.obj, v_ref.obj, dO_ref.obj, l_ref.obj, d_ref.obj,
        dQ_ref.obj, dK_ref.obj, dV_ref.obj,
        static_cast<uint64_t>(seq), static_cast<uint64_t>(q.size(3)),
        bwd_cfg.grid_x, bwd_cfg.grid_y, bwd_cfg.grid_z,
        bwd_cfg.block_x, bwd_cfg.block_y, bwd_cfg.block_z,
        bwd_cfg.shared_mem, bwd_cfg.stream);
    throw_on_launcher_error(bwd_result, "tkMhaH100Bwd2BlockKvSweep");
  } else {
    bwd_result = lean_launch_Tyr_GPU_Kernels_tkMhaH100Bwd12BlockKvSweep(
        q_ref.obj, k_ref.obj, v_ref.obj, dO_ref.obj, l_ref.obj, d_ref.obj,
        dQ_ref.obj, dK_ref.obj, dV_ref.obj,
        static_cast<uint64_t>(seq), static_cast<uint64_t>(q.size(3)),
        bwd_cfg.grid_x, bwd_cfg.grid_y, bwd_cfg.grid_z,
        bwd_cfg.block_x, bwd_cfg.block_y, bwd_cfg.block_z,
        bwd_cfg.shared_mem, bwd_cfg.stream);
    throw_on_launcher_error(bwd_result, "tkMhaH100Bwd12BlockKvSweep");
  }

  return {dQ, dK, dV};
}

static std::pair<torch::Tensor, torch::Tensor> vendored_tk_forward(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    bool is_causal) {
  const auto q = query.contiguous();
  const auto k = key.contiguous();
  const auto v = value.contiguous();
  auto result = tyr_tk_attention_forward_nosync(q, k, v, is_causal);
  TORCH_CHECK(result.size() == 2, "tyr::flash_attn: vendored TK forward returned unexpected tensor count");
  return {result[0], result[1]};
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> vendored_tk_backward(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& out,
    const torch::Tensor& l,
    const torch::Tensor& grad_out,
    bool is_causal) {
  const auto q = query.contiguous();
  const auto k = key.contiguous();
  const auto v = value.contiguous();
  const auto o = out.contiguous();
  const auto l_saved = l.contiguous();
  const auto dO = grad_out.contiguous();
  auto result = tyr_tk_attention_backward_nosync(q, k, v, o, l_saved, dO, is_causal);
  TORCH_CHECK(result.size() == 3, "tyr::flash_attn: vendored TK backward returned unexpected tensor count");
  return {result[0], result[1], result[2]};
}

static torch::Tensor portable_flash_attn(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const c10::optional<torch::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    const c10::optional<double>& scale,
    bool enable_gqa) {
  return bounded_sdpa(query, key, value, attn_mask, AttentionMaskKind::Padding,
      dropout_p, is_causal, scale, enable_gqa);
}

template <FlashAttnRoute Route, FlashAttnImpl Impl>
class NativeFlashAttnFunction
    : public torch::autograd::Function<NativeFlashAttnFunction<Route, Impl>> {
 public:
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      torch::Tensor query,
      torch::Tensor key,
      torch::Tensor value) {
    auto [out, l] =
        Impl == FlashAttnImpl::Generated
            ? generated_forward(query, key, value, Route)
            : vendored_tk_forward(query, key, value, false);
    ctx->save_for_backward({query, key, value, out, l});
    return out;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    if (grad_outputs.empty() || !grad_outputs[0].defined()) {
      return {torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
    auto [dQ, dK, dV] =
        Impl == FlashAttnImpl::Generated
            ? generated_backward(
                  saved[0], saved[1], saved[2], saved[3], saved[4], grad_outputs[0], Route)
            : vendored_tk_backward(
                  saved[0], saved[1], saved[2], saved[3], saved[4], grad_outputs[0], false);
    return {dQ, dK, dV};
  }
};

template <FlashAttnImpl Impl>
static torch::Tensor flash_attn_dispatch_impl(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const c10::optional<torch::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    const c10::optional<double>& scale,
    bool enable_gqa) {
  check_flash_attn_args(query, key, value, attn_mask);

  auto route = select_route(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
  switch (route) {
    case FlashAttnRoute::TkMhaH100Decode:
      return generated_decode_forward(query, key, value);
    case FlashAttnRoute::TkMhaH1002Block:
      return NativeFlashAttnFunction<FlashAttnRoute::TkMhaH1002Block, FlashAttnImpl::Generated>::apply(
          query, key, value);
    case FlashAttnRoute::TkMhaH10012Block:
      return NativeFlashAttnFunction<FlashAttnRoute::TkMhaH10012Block, Impl>::apply(query, key, value);
    case FlashAttnRoute::Portable:
    default:
      return portable_flash_attn(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
  }
}

torch::Tensor flash_attn_dispatch(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const c10::optional<torch::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    const c10::optional<double>& scale,
    bool enable_gqa) {
  return flash_attn_dispatch_impl<FlashAttnImpl::VendoredTk>(
      query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
}

torch::Tensor flash_attn_dispatch_generated(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const c10::optional<torch::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    const c10::optional<double>& scale,
    bool enable_gqa) {
  return flash_attn_dispatch_impl<FlashAttnImpl::Generated>(
      query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
}

torch::Tensor flash_attn_dispatch_vendored_tk(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const c10::optional<torch::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    const c10::optional<double>& scale,
    bool enable_gqa) {
  return flash_attn_dispatch_impl<FlashAttnImpl::VendoredTk>(
      query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
}

} // namespace tyr_ops

TORCH_LIBRARY(tyr, m) {
  m.def("flash_attn(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0.0, bool is_causal=False, float? scale=None, bool enable_gqa=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(tyr, CompositeImplicitAutograd, m) {
  m.impl("flash_attn", TORCH_FN(tyr_ops::flash_attn_dispatch));
}

extern "C" LEAN_EXPORT lean_object* lean_torch_tyr_flash_attn_4d(
    uint64_t /*batch*/,
    uint64_t /*n_head*/,
    uint64_t /*n_kv_head*/,
    uint64_t /*q_seq*/,
    uint64_t /*kv_seq*/,
    uint64_t /*head_dim*/,
    b_lean_obj_arg query,
    b_lean_obj_arg key,
    b_lean_obj_arg value,
    lean_obj_arg attn_mask,
    double dropout_p,
    uint8_t is_causal,
    lean_obj_arg scale,
    uint8_t enable_gqa) {
  return tyr::checkedCall("flashAttention", [&] {
    // Both options are transferred by Lean; keep them owned even if borrowing
    // Q/K/V or parsing the first option throws before the second is inspected.
    std::unique_ptr<lean_object, decltype(&lean_dec)> mask_owner(attn_mask, lean_dec);
    std::unique_ptr<lean_object, decltype(&lean_dec)> scale_owner(scale, lean_dec);
    auto q = borrowTensor(query);
    auto k = borrowTensor(key);
    auto v = borrowTensor(value);

    c10::optional<torch::Tensor> mask_opt = c10::nullopt;
    if (!lean_is_scalar(attn_mask)) {
      mask_opt = borrowTensor(lean_ctor_get(attn_mask, 0));
    }
    mask_owner.reset();

    c10::optional<double> scale_opt = c10::nullopt;
    if (!lean_is_scalar(scale)) {
      scale_opt = static_cast<double>(lean_unbox_float(lean_ctor_get(scale, 0)));
    }
    scale_owner.reset();

    auto result = tyr_ops::flash_attn_dispatch(
        q, k, v, mask_opt, dropout_p, is_causal != 0, scale_opt, enable_gqa != 0);
    return fromTorchTensor(result);
  });
}
