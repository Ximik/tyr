#include "generated/tyr_ffi.h"
#include <ATen/CPUGeneratorImpl.h>
#include <torch/torch.h>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

extern "C" void lean_initialize();

namespace {
using Object = std::unique_ptr<lean_object, decltype(&lean_dec)>;
void require(bool valid, const char* message) {
  if (!valid) throw std::runtime_error(message);
}
lean_object* shape(std::initializer_list<uint64_t> dims) {
  auto* result = lean_alloc_array(dims.size(), dims.size());
  size_t i = 0;
  for (auto dim : dims) lean_array_set_core(result, i++, lean_box_uint64(dim));
  return result;
}
Object full(std::initializer_list<uint64_t> dims, double value) {
  return Object(lean_torch_full(shape(dims), value, 0, lean_box(1)), lean_dec);
}
lean_object* someBorrowed(lean_object* value) {
  auto* result = lean_alloc_ctor(1, 1, 0);
  lean_inc(value);
  lean_ctor_set(result, 0, value);
  return result;
}
torch::Tensor state() {
  auto generator = at::detail::getDefaultCPUGenerator();
  std::lock_guard<std::mutex> lock(generator.mutex());
  return generator.get_state().clone();
}
void seed(uint64_t value) {
  auto generator = at::detail::getDefaultCPUGenerator();
  std::lock_guard<std::mutex> lock(generator.mutex());
  generator.set_current_seed(value);
}
std::vector<double> values(lean_object* result, bool io) {
  lean_object* tensor;
  if (io) {
    require(lean_io_result_is_ok(result), "Expected IO.ok attention result");
    tensor = lean_io_result_get_value(result);
  } else {
    require(!lean_is_scalar(result) && !lean_is_external(result) && lean_obj_tag(result) == 1,
      "Expected Except.ok attention result");
    tensor = lean_ctor_get(result, 0);
  }
  require(!lean_is_scalar(tensor) && lean_is_external(tensor), "Attention payload is not a tensor");
  Object observed(lean_torch_tensor_to_float_array_dynamic(tensor), lean_dec);
  require(lean_io_result_is_ok(observed.get()), "Could not inspect attention output");
  auto* array = lean_io_result_get_value(observed.get());
  std::vector<double> output;
  for (size_t i = 0; i < lean_array_size(array); ++i)
    output.push_back(lean_unbox_float(lean_array_get_core(array, i)));
  require(output.size() == 64, "Unexpected attention shape");
  return output;
}

struct Inputs {
  Object q = full({1, 1, 16, 4}, 0);
  Object v = full({1, 1, 16, 4}, 1);
  Object padding = full({1, 16}, 1);
  Object edges = full({1, 16, 16}, 1);
  Object bias = full({1, 1, 16, 16}, 0);

  void referencesUnchanged() const {
    require(q->m_rc == 1 && v->m_rc == 1 && padding->m_rc == 1 &&
            edges->m_rc == 1 && bias->m_rc == 1, "Attention changed borrowed references");
  }
};

enum class Family { Basic, GQA, Padding, QKV, Window, Flash, Gemma, Bias };
const char* name(Family family) {
  switch (family) {
    case Family::Basic: return "attention";
    case Family::GQA: return "attentionGQA";
    case Family::Padding: return "attentionGQAMask";
    case Family::QKV: return "attentionGQAQKV";
    case Family::Window: return "attentionWindow";
    case Family::Flash: return "flashAttention";
    case Family::Gemma: return "gemma4MaskedAttention";
    case Family::Bias: return "branchingAttentionBias";
  }
  throw std::runtime_error("Unknown attention family");
}
Object pureCall(Family family, const Inputs& x) {
  auto* q = x.q.get(); auto* v = x.v.get();
  lean_object* result = nullptr;
  switch (family) {
    case Family::Basic: result = lean_torch_sdpa_4d(q, q, v, 0); break;
    case Family::GQA: result = lean_torch_sdpa_gqa(q, q, v, 0, 0); break;
    case Family::Padding: result = lean_torch_sdpa_gqa_mask(q, q, v, x.padding.get(), 0, 0); break;
    case Family::QKV: result = lean_torch_sdpa_gqa_qkv(q, q, v, 0, 0); break;
    case Family::Window: result = lean_torch_sdpa_gqa_window(q, q, v, 1, 0, 8); break;
    case Family::Flash: result = lean_torch_tyr_flash_attn_4d(q, q, v, someBorrowed(x.padding.get()), 0, lean_box(0), 0); break;
    case Family::Gemma: result = lean_torch_sdpa_gqa_mask_qkv(q, q, v, x.edges.get(), 0); break;
    case Family::Bias: result = lean_torch_sdpa_4d_bias(q, q, v, x.bias.get(), 0); break;
  }
  return Object(result, lean_dec);
}
Object ioCall(Family family, const Inputs& x, double p) {
  auto* q = x.q.get(); auto* v = x.v.get();
  lean_object* result = nullptr;
  switch (family) {
    case Family::Basic: result = lean_torch_sdpa_4d_io(q, q, v, p, 0); break;
    case Family::GQA: result = lean_torch_sdpa_gqa_io(q, q, v, p, 0, 0); break;
    case Family::Padding: result = lean_torch_sdpa_gqa_mask_io(q, q, v, x.padding.get(), p, 0, 0); break;
    case Family::QKV: result = lean_torch_sdpa_gqa_qkv_io(q, q, v, p, 0, 0); break;
    case Family::Window: result = lean_torch_sdpa_gqa_window_io(q, q, v, p, 1, 0, 8); break;
    case Family::Flash: result = lean_torch_tyr_flash_attn_4d_io(q, q, v, someBorrowed(x.padding.get()), p, 0, lean_box(0), 0); break;
    case Family::Gemma: result = lean_torch_sdpa_gqa_mask_qkv_io(q, q, v, x.edges.get(), p, 0); break;
    case Family::Bias: result = lean_torch_sdpa_4d_bias_io(q, q, v, x.bias.get(), p, 0); break;
  }
  return Object(result, lean_dec);
}

void check(Family family) {
  Inputs x;
  seed(1729);
  const auto initial = state();
  auto inference = pureCall(family, x);
  auto expected = values(inference.get(), false);
  require(torch::equal(initial, state()), "Pure attention consumed RNG state");
  auto zero = ioCall(family, x, 0);
  require(values(zero.get(), true) == expected, "Zero dropout differs from pure attention");
  require(torch::equal(initial, state()), "Zero dropout consumed RNG state");

  auto first = ioCall(family, x, 0.5);
  auto firstValues = values(first.get(), true);
  auto firstState = state();
  require(!torch::equal(initial, firstState), "First IO call did not advance RNG");
  auto second = ioCall(family, x, 0.5);
  auto secondValues = values(second.get(), true);
  auto secondState = state();
  require(!torch::equal(firstState, secondState), "Second IO call did not advance RNG");
  seed(1729);
  auto replayFirst = ioCall(family, x, 0.5);
  auto replaySecond = ioCall(family, x, 0.5);
  require(values(replayFirst.get(), true) == firstValues &&
          values(replaySecond.get(), true) == secondValues && torch::equal(state(), secondState),
          "Seed replay failed to reproduce both draws and RNG state");

  auto allDropped = ioCall(family, x, 1);
  for (auto value : values(allDropped.get(), true)) require(value == 0, "p=1 did not drop all attention");
  for (double invalid : {-0.5, 1.5, std::numeric_limits<double>::quiet_NaN()}) {
    const auto beforeError = state();
    auto error = ioCall(family, x, invalid);
    require(!lean_io_result_is_ok(error.get()), "Invalid probability did not return IO.error");
    require(torch::equal(beforeError, state()), "Rejected probability consumed RNG");
    x.referencesUnchanged();
  }
  x.referencesUnchanged();
}
} // namespace

int main() {
  lean_initialize();
  int failures = 0;
  for (auto family : {Family::Basic, Family::GQA, Family::Padding, Family::QKV,
                     Family::Window, Family::Flash, Family::Gemma, Family::Bias}) {
    try { check(family); std::cout << "PASS " << name(family) << " effects\n"; }
    catch (const std::exception& error) {
      ++failures;
      std::cerr << "FAIL " << name(family) << ": " << error.what() << '\n';
    }
  }
  return failures ? 1 : 0;
}
