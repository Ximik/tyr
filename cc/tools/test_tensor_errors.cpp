#include "tyr_raw_tensor_abi.h"
#include "tyr_neural_abi.h"
#include <bit>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

extern "C" void lean_initialize();
// Kept for native ABI compatibility; no current Lean declaration uses rand.
extern "C" lean_object* lean_torch_rand(lean_object*, int, lean_object*);

namespace {
using Object = std::unique_ptr<lean_object, decltype(&lean_dec)>;
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
lean_object* shape(std::initializer_list<uint64_t> dims) {
  auto* result = lean_alloc_array(dims.size(), dims.size());
  size_t i = 0;
  for (auto dim : dims) lean_array_set_core(result, i++, lean_box_uint64(dim));
  return result;
}
lean_object* retain(lean_object* value) { lean_inc(value); return value; }
lean_object* some(lean_object* value) {
  auto* result = lean_alloc_ctor(1, 1, 0);
  lean_ctor_set(result, 0, value);
  return result;
}
Object full(std::initializer_list<uint64_t> dims, double value = 0) {
  return Object(lean_torch_full(shape(dims), value, 0, lean_box(1)), lean_dec);
}
lean_object* exceptValue(lean_object* result) {
  require(!lean_is_scalar(result) && !lean_is_external(result) && lean_obj_tag(result) == 1,
    "Expected Except.ok constructor, not a raw tensor or IO carrier");
  auto* value = lean_ctor_get(result, 0);
  require(!lean_is_scalar(value) && lean_is_external(value), "Except.ok payload is not RawTensor");
  return value;
}
void exceptError(lean_object* result, const char* fragment) {
  require(!lean_is_scalar(result) && !lean_is_external(result) && lean_obj_tag(result) == 0,
    "Expected Except.error constructor, not a raw tensor or IO carrier");
  auto* error = lean_ctor_get(result, 0);
  require(lean_is_string(error), "Except.error payload is not String");
  require(std::string(lean_string_cstr(error)).find(fragment) != std::string::npos,
    "Except.error omitted expected context");
}
void values(lean_object* tensor, size_t count, double expected) {
  Object result(lean_torch_tensor_to_float_array_dynamic(tensor), lean_dec);
  require(lean_io_result_is_ok(result.get()), "Could not inspect result values");
  auto* array = lean_io_result_get_value(result.get());
  require(lean_array_size(array) == count, "Unexpected output size");
  for (size_t i = 0; i < count; ++i)
    require(lean_unbox_float(lean_array_get_core(array, i)) == expected, "Unexpected output value");
}
Object flash(lean_object* q, lean_object* k, lean_object* v, double dropout = 0,
             lean_object* mask = lean_box(0), lean_object* scale = lean_box(0)) {
  return Object(lean_torch_tyr_flash_attn_4d(1, 1, 1, 2, 2, 2,
    q, k, v, mask, dropout, 0, scale, 0), lean_dec);
}
void flashSuccess() {
  auto q = full({1, 1, 2, 2}), v = full({1, 1, 2, 2}, 7);
  auto result = flash(q.get(), q.get(), v.get());
  values(exceptValue(result.get()), 4, 7);
  require(q->m_rc == 1 && v->m_rc == 1, "FlashAttention changed borrowed references");
}
void flashDropout() {
  auto q = full({1, 1, 2, 2});
  for (double dropout : {-0.5, 1.5}) {
    auto result = flash(q.get(), q.get(), q.get(), dropout);
    exceptError(result.get(), "dropout");
  }
}
void flashRank() {
  auto q = full({2, 2});
  auto result = flash(q.get(), q.get(), q.get());
  exceptError(result.get(), "rank-4");
}
void flashDType() {
  Object q(lean_torch_full_int(shape({1, 1, 2, 2}), 1), lean_dec);
  auto result = flash(q.get(), q.get(), q.get());
  exceptError(result.get(), "flashAttention");
}
void flashOptionCleanup(bool invalidCarrier) {
  auto q = full({1, 1, 2, 2}), maskTensor = full({1, 3});
  Object invalid(lean_mk_string("not a tensor"), lean_dec);
  Object mask(some(retain(maskTensor.get())), lean_dec), scale(some(lean_box_float(0.5)), lean_dec);
  auto result = flash(invalidCarrier ? invalid.get() : q.get(), q.get(), q.get(), 0,
    retain(mask.get()), retain(scale.get()));
  exceptError(result.get(), invalidCarrier ? "borrowTensor" : "attn_mask");
  require(mask->m_rc == 1 && scale->m_rc == 1, "FlashAttention leaked owned options on failure");
  require(q->m_rc == 1 && maskTensor->m_rc == 2, "FlashAttention changed borrowed tensor ownership");
}
void randintError(bool invalidDevice) {
  Object dims(shape({4}), lean_dec);
  auto* cuda = lean_alloc_ctor(0, 0, sizeof(uint64_t));
  lean_ctor_set_uint64(cuda, 0, std::numeric_limits<uint64_t>::max());
  Object device(invalidDevice ? cuda : lean_box(1), lean_dec);
  if (!invalidDevice) lean_dec(cuda);
  Object result(lean_torch_randint(3, invalidDevice ? 4 : 3, retain(dims.get()), 0,
    retain(device.get())), lean_dec);
  require(lean_io_result_is_error(result.get()), "randint did not return IO.error");
  require(dims->m_rc == 1, "randint leaked shape on failure");
  if (invalidDevice) require(device->m_rc == 1, "randint leaked device on failure");
}
void randintSigned() {
  Object result(lean_torch_randint(std::bit_cast<uint64_t>(int64_t{-7}),
    std::bit_cast<uint64_t>(int64_t{-6}), shape({4}), 0, lean_box(1)), lean_dec);
  require(lean_io_result_is_ok(result.get()), "Valid signed randint failed");
  values(lean_io_result_get_value(result.get()), 4, -7);
}
void randomFactoryError(bool normal, bool invalidDevice) {
  Object dims(invalidDevice ? shape({4}) : shape({std::numeric_limits<uint64_t>::max()}), lean_dec);
  auto* cuda = lean_alloc_ctor(0, 0, sizeof(uint64_t));
  lean_ctor_set_uint64(cuda, 0, std::numeric_limits<uint64_t>::max());
  Object device(invalidDevice ? cuda : lean_box(1), lean_dec);
  if (!invalidDevice) lean_dec(cuda);
  Object result(normal ? lean_torch_randn(retain(dims.get()), 0, retain(device.get()))
                       : lean_torch_rand(retain(dims.get()), 0, retain(device.get())), lean_dec);
  require(lean_io_result_is_error(result.get()), "Random factory did not return IO.error");
  require(dims->m_rc == 1, "Random factory leaked shape on failure");
  if (invalidDevice) require(device->m_rc == 1, "Random factory leaked device on failure");
}
void randomFactorySuccess(bool normal) {
  Object dims(shape({2, 3}), lean_dec);
  for (bool requiresGrad : {false, true}) {
    Object result(normal ? lean_torch_randn(retain(dims.get()), requiresGrad, lean_box(1))
                         : lean_torch_rand(retain(dims.get()), requiresGrad, lean_box(1)), lean_dec);
    require(lean_io_result_is_ok(result.get()), "Valid random factory failed");
    auto* tensor = lean_io_result_get_value(result.get());
    require(!lean_is_scalar(tensor) && lean_is_external(tensor), "Random factory did not return RawTensor");
    Object actual(lean_torch_get_shape(shape({}), tensor), lean_dec);
    require(lean_array_size(actual.get()) == 2 &&
      lean_unbox_uint64(lean_array_get_core(actual.get(), 0)) == 2 &&
      lean_unbox_uint64(lean_array_get_core(actual.get(), 1)) == 3,
      "Random factory changed requested shape");
    require(lean_torch_requires_grad(shape({2, 3}), tensor) == requiresGrad,
      "Random factory changed requested trainability");
    require(lean_torch_is_leaf(tensor), "Random factory did not return a leaf");
    require(dims->m_rc == 1, "Random factory leaked shape on success");
  }
}
void dropoutError(double probability, bool invalidCarrier = false) {
  Object dims(shape({4}), lean_dec);
  auto tensor = full({4}, 7);
  Object invalid(lean_mk_string("not a tensor"), lean_dec);
  Object result(lean_torch_dropout(retain(dims.get()),
    invalidCarrier ? invalid.get() : tensor.get(), probability, 1), lean_dec);
  require(lean_io_result_is_error(result.get()), "Dropout did not return IO.error");
  require(dims->m_rc == 1, "Dropout leaked owned shape on failure");
  require(tensor->m_rc == 1 && invalid->m_rc == 1, "Dropout changed borrowed argument ownership");
}
void dropoutSuccess() {
  auto tensor = full({4}, 7);
  for (auto probability : {0.0, 1.0}) {
    Object dims(shape({4}), lean_dec);
    Object result(lean_torch_dropout(retain(dims.get()), tensor.get(), probability, 1), lean_dec);
    require(lean_io_result_is_ok(result.get()), "Valid dropout failed");
    values(lean_io_result_get_value(result.get()), 4, probability == 0 ? 7 : 0);
    require(dims->m_rc == 1 && tensor->m_rc == 1, "Dropout ownership imbalance on success");
  }
  Object inference(lean_torch_dropout(shape({4}), tensor.get(), 0.5, 0), lean_dec);
  require(lean_io_result_is_ok(inference.get()), "Inference dropout failed");
  values(lean_io_result_get_value(inference.get()), 4, 7);
}
void transpose(bool invalid, bool huge = false) {
  auto input = full({2, 3}, 4);
  Object dims(shape({2, 3}), lean_dec);
  Object result(lean_torch_transpose(retain(dims.get()), input.get(), 0,
    huge ? std::numeric_limits<uint64_t>::max() : invalid ? 9 : 1), lean_dec);
  if (invalid || huge) exceptError(result.get(), "transpose");
  else {
    auto* tensor = exceptValue(result.get());
    values(tensor, 6, 4);
    Object actual(lean_torch_get_shape(shape({}), tensor), lean_dec);
    require(lean_array_size(actual.get()) == 2 &&
      lean_unbox_uint64(lean_array_get_core(actual.get(), 0)) == 3 &&
      lean_unbox_uint64(lean_array_get_core(actual.get(), 1)) == 2, "Transpose shape changed");
  }
  require(dims->m_rc == 1 && input->m_rc == 1, "Transpose ownership imbalance");
}
void transposeScalar() {
  auto input = full({}, 5);
  Object result(lean_torch_transpose(shape({}), input.get(), 0, 0), lean_dec);
  values(exceptValue(result.get()), 1, 5);
}
} // namespace

int main() {
  lean_initialize();
  int failures = 0;
  auto run = [&](const char* label, const std::function<void()>& action) {
    try { action(); std::cout << "PASS " << label << '\n'; }
    catch (const std::exception& error) {
      ++failures;
      std::cout << "FAIL " << label << ": " << error.what() << '\n';
    }
  };
  run("flash/success-carrier", flashSuccess);
  run("flash/dropout-error", flashDropout);
  run("flash/rank-error", flashRank);
  run("flash/dtype-error", flashDType);
  run("flash/options-early-error", [] { flashOptionCleanup(true); });
  run("flash/options-dispatch-error", [] { flashOptionCleanup(false); });
  run("randint/bounds-error", [] { randintError(false); });
  run("randint/device-error", [] { randintError(true); });
  run("randint/signed-success", randintSigned);
  run("randn/device-error", [] { randomFactoryError(true, true); });
  run("randn/shape-error", [] { randomFactoryError(true, false); });
  run("randn/success", [] { randomFactorySuccess(true); });
  run("rand/device-error", [] { randomFactoryError(false, true); });
  run("rand/shape-error", [] { randomFactoryError(false, false); });
  run("rand/success", [] { randomFactorySuccess(false); });
  run("dropout/negative-probability", [] { dropoutError(-0.5); });
  run("dropout/excessive-probability", [] { dropoutError(1.5); });
  run("dropout/nan-probability", [] { dropoutError(std::numeric_limits<double>::quiet_NaN()); });
  run("dropout/invalid-carrier", [] { dropoutError(0.5, true); });
  run("dropout/success", dropoutSuccess);
  run("transpose/success-carrier", [] { transpose(false); });
  run("transpose/axis-error", [] { transpose(true); });
  run("transpose/wrapped-axis-error", [] { transpose(false, true); });
  run("transpose/scalar-success", transposeScalar);
  std::cout << failures << " native error-carrier failures\n";
  return failures == 0 ? 0 : 1;
}
