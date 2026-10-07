// Optional CPU inference benchmark; run each case in a fresh process so the
// process-wide peak RSS values are comparable. Not part of normal CI.
// From the repository root (add LibTorch's C++ ABI flag if its package needs one):
//   c++ -std=c++20 -O2 -Icc/include -Iexternal/wheels/torch/include \
//     -Iexternal/wheels/torch/include/torch/csrc/api/include \
//     cc/tools/benchmark_attention.cpp -Lexternal/wheels/torch/lib \
//     -Wl,-rpath,"$PWD/external/wheels/torch/lib" -ltorch -ltorch_cpu -lc10 \
//     -o /tmp/benchmark_attention
//   /tmp/benchmark_attention old window 8192
//   /tmp/benchmark_attention new window 8192
// Replace "window" with "full" for global causal attention. Inputs are seeded
// float32 [B=1,Hq=8,Hkv=2,L,D=64], with window=512 and one Torch CPU thread.
// "old" reproduces the former expanded-KV/dense-window-mask implementation.
// RSS includes the runtime, inputs, outputs and temporary work; it is not CUDA
// allocated memory. Concurrent builds can strongly affect forward timing.

#include <chrono>
#include <iostream>
#include <limits>
#include <string>
#include <sys/resource.h>
#include <torch/version.h>

#include "tyr_attention.h"

int main(int argc, char** argv) {
  if (argc != 4 || (std::string(argv[1]) != "old" && std::string(argv[1]) != "new") ||
      (std::string(argv[2]) != "full" && std::string(argv[2]) != "window")) {
    std::cerr << "Usage: benchmark_attention old|new full|window SEQUENCE_LENGTH\n";
    return 1;
  }
  const int64_t length = std::stoll(argv[3]);
  if (length <= 0) {
    std::cerr << "Sequence length must be positive\n";
    return 1;
  }
  torch::set_num_threads(1);
  torch::manual_seed(714);
  torch::NoGradGuard no_grad;
  const bool old = std::string(argv[1]) == "old";
  const bool window = std::string(argv[2]) == "window";
  const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto q = torch::randn({1, 8, length, 64}, options);
  auto k = torch::randn({1, 2, length, 64}, options);
  auto v = torch::randn_like(k);
  const auto start = std::chrono::steady_clock::now();
  torch::Tensor result;
  if (old) {
    k = k.repeat_interleave(4, 1);
    v = v.repeat_interleave(4, 1);
    std::optional<torch::Tensor> mask;
    if (window) {
      auto rows = torch::arange(length, torch::kLong).unsqueeze(1);
      auto cols = torch::arange(length, torch::kLong).unsqueeze(0);
      auto causal = cols > rows;
      auto past = rows - cols >= 512;
      auto blocked = causal | past;
      mask = torch::where(blocked,
          torch::full({length, length}, -std::numeric_limits<float>::infinity(), options),
          torch::zeros({length, length}, options)).unsqueeze(0).unsqueeze(0);
    }
    result = torch::scaled_dot_product_attention(q, k, v, mask, 0.0, !window);
  } else {
    result = tyr_ops::bounded_sdpa(q, k, v, std::nullopt,
        tyr_ops::AttentionMaskKind::Padding, 0.0, true, std::nullopt, true,
        window ? std::optional<int64_t>(512) : std::nullopt);
  }
  const auto milliseconds = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
  const auto checksum = result.sum(torch::kFloat64).item<double>();
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    std::cerr << "Could not read peak RSS\n";
    return 1;
  }
#ifdef __APPLE__
  const auto peak_bytes = static_cast<uint64_t>(usage.ru_maxrss);
#else
  const auto peak_bytes = static_cast<uint64_t>(usage.ru_maxrss) * 1024;
#endif
  std::cout << "libtorch=" << TORCH_VERSION << " device=cpu dtype=float32 mode=" << argv[1]
      << " attention=" << argv[2] << " sequence=" << length
      << " forward_ms=" << milliseconds << " peak_rss_bytes=" << peak_bytes
      << " checksum=" << checksum << '\n';
}
