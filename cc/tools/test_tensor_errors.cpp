#include "generated/tyr_ffi.h"
#include "tyr_checked_result.h"
#include <bit>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

extern "C" void lean_initialize();

// Tests-first mode applies the checked contract to the previous raw boundary.
// It must fail: native errors escape and successful results lack Except.ok.
#ifdef TYR_CHECKED_BASELINE
#define CHECKED(F) F
#else
#define CHECKED(F) F##_checked
#endif

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
Object flash(lean_object* q, lean_object* k, lean_object* v,
             lean_object* mask = lean_box(0), lean_object* scale = lean_box(0)) {
  return Object(lean_torch_tyr_flash_attn_4d(
    q, k, v, mask, 0, scale, 0), lean_dec);
}
Object flashIO(lean_object* q, lean_object* k, lean_object* v, double dropout,
               lean_object* mask = lean_box(0), lean_object* scale = lean_box(0)) {
  return Object(lean_torch_tyr_flash_attn_4d_io(
    q, k, v, mask, dropout, 0, scale, 0), lean_dec);
}
void checkedCrossEntropy(bool unreduced, int invalid) {
  Object logits(invalid == 1 ? lean_torch_full_int(shape({2, 2}), 0)
                            : lean_torch_full(shape({2, 2}), 0, 0, lean_box(1)), lean_dec);
  Object labels(lean_torch_full_int(shape({2}), invalid == 2 ? 2 : 0), lean_dec);
  Object result(unreduced
    ? CHECKED(lean_torch_cross_entropy_none)(logits.get(), labels.get())
    : CHECKED(lean_torch_cross_entropy_2d)(logits.get(), labels.get()), lean_dec);
  if (invalid) {
    exceptError(result.get(), "crossEntropy");
  } else {
    Object observed(lean_torch_tensor_to_float_array_dynamic(exceptValue(result.get())), lean_dec);
    require(lean_io_result_is_ok(observed.get()), "Could not inspect loss values");
    auto* array = lean_io_result_get_value(observed.get());
    require(lean_array_size(array) == (unreduced ? 2 : 1), "Unexpected loss result shape");
    for (size_t i = 0; i < lean_array_size(array); ++i)
      require(std::abs(lean_unbox_float(lean_array_get_core(array, i)) - std::log(2.0)) < 1e-6,
              "Incorrect cross entropy value");
  }
  require(logits->m_rc == 1 && labels->m_rc == 1, "Cross entropy changed borrowed references");
}

void checkedItem(bool invalid) {
  Object input(lean_torch_full(shape(invalid ? std::initializer_list<uint64_t>{2}
                                          : std::initializer_list<uint64_t>{}),
                             3.25, 0, lean_box(1)), lean_dec);
  Object result(lean_torch_item_checked(input.get()), lean_dec);
  if (invalid) {
    exceptError(result.get(), "item");
  } else {
    require(!lean_is_scalar(result.get()) && !lean_is_external(result.get()) &&
            lean_obj_tag(result.get()) == 1, "Expected Except.ok scalar carrier");
    require(lean_unbox_float(lean_ctor_get(result.get(), 0)) == 3.25,
            "Incorrect boxed Float payload");
  }
  require(input->m_rc == 1, "Item changed borrowed references");
}

void checkedIndexSelect(int scenario) {
  const uint64_t size = scenario == 3 ? 0 : 2;
  const uint64_t count = scenario == 3 ? 0 : 1;
  const uint64_t id = scenario == 1 ? UINT64_MAX : scenario == 2 ? 2 : 0;
  Object input(lean_torch_full(shape({size}), 3.25, 0, lean_box(1)), lean_dec);
  Object indices(lean_torch_full_int(shape({count}), id), lean_dec);
  Object result(lean_torch_index_select_checked(input.get(), 0, indices.get()), lean_dec);
  if (scenario == 1 || scenario == 2) exceptError(result.get(), "indexSelect");
  else values(exceptValue(result.get()), count, 3.25);
  require(input->m_rc == 1 && indices->m_rc == 1,
          "Index selection changed borrowed references");
}

void flashSuccess() {
  auto q = full({1, 1, 2, 2}), v = full({1, 1, 2, 2}, 7);
  auto result = flash(q.get(), q.get(), v.get());
  values(exceptValue(result.get()), 4, 7);
  require(q->m_rc == 1 && v->m_rc == 1, "FlashAttention changed borrowed references");
}
void flashDropout() {
  auto q = full({1, 1, 2, 2});
  for (double dropout : {-0.5, 1.5, std::numeric_limits<double>::quiet_NaN()}) {
    auto result = flashIO(q.get(), q.get(), q.get(), dropout);
    require(lean_io_result_is_error(result.get()), "Invalid FlashAttention dropout did not return IO.error");
  }
  require(q->m_rc == 1, "FlashAttention IO changed borrowed references");
  auto success = flashIO(q.get(), q.get(), q.get(), 1.0);
  require(lean_io_result_is_ok(success.get()), "Valid dropout p=1 failed after errors");
  values(lean_io_result_get_value(success.get()), 4, 0);
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
void flashOptionCleanup(bool invalidCarrier, bool effectful = false) {
  auto q = full({1, 1, 2, 2}), maskTensor = full({1, 3});
  Object invalid(lean_mk_string("not a tensor"), lean_dec);
  Object mask(some(retain(maskTensor.get())), lean_dec), scale(some(lean_box_float(0.5)), lean_dec);
  auto result = effectful
    ? flashIO(invalidCarrier ? invalid.get() : q.get(), q.get(), q.get(), 0,
        retain(mask.get()), retain(scale.get()))
    : flash(invalidCarrier ? invalid.get() : q.get(), q.get(), q.get(),
        retain(mask.get()), retain(scale.get()));
  if (effectful) require(lean_io_result_is_error(result.get()), "Expected IO.error for invalid FlashAttention inputs");
  else exceptError(result.get(), invalidCarrier ? "borrowTensor" : "attn_mask");
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
void randomFactoryError(bool invalidDevice) {
  Object dims(invalidDevice ? shape({4}) : shape({std::numeric_limits<uint64_t>::max()}), lean_dec);
  auto* cuda = lean_alloc_ctor(0, 0, sizeof(uint64_t));
  lean_ctor_set_uint64(cuda, 0, std::numeric_limits<uint64_t>::max());
  Object device(invalidDevice ? cuda : lean_box(1), lean_dec);
  if (!invalidDevice) lean_dec(cuda);
  Object result(lean_torch_randn(retain(dims.get()), 0, retain(device.get())), lean_dec);
  require(lean_io_result_is_error(result.get()), "Random factory did not return IO.error");
  require(dims->m_rc == 1, "Random factory leaked shape on failure");
  if (invalidDevice) require(device->m_rc == 1, "Random factory leaked device on failure");
}
void randomFactorySuccess() {
  Object dims(shape({2, 3}), lean_dec);
  for (bool requiresGrad : {false, true}) {
    Object result(lean_torch_randn(retain(dims.get()), requiresGrad, lean_box(1)), lean_dec);
    require(lean_io_result_is_ok(result.get()), "Valid random factory failed");
    auto* tensor = lean_io_result_get_value(result.get());
    require(!lean_is_scalar(tensor) && lean_is_external(tensor), "Random factory did not return RawTensor");
    Object actual(lean_torch_get_shape(tensor), lean_dec);
    require(lean_array_size(actual.get()) == 2 &&
      lean_unbox_uint64(lean_array_get_core(actual.get(), 0)) == 2 &&
      lean_unbox_uint64(lean_array_get_core(actual.get(), 1)) == 3,
      "Random factory changed requested shape");
    require(lean_torch_requires_grad(tensor) == requiresGrad,
      "Random factory changed requested trainability");
    require(lean_torch_is_leaf(tensor), "Random factory did not return a leaf");
    require(dims->m_rc == 1, "Random factory leaked shape on success");
  }
}
void dropoutError(double probability, bool invalidCarrier = false) {
  auto tensor = full({4}, 7);
  Object invalid(lean_mk_string("not a tensor"), lean_dec);
  Object result(lean_torch_dropout(
    invalidCarrier ? invalid.get() : tensor.get(), probability, 1), lean_dec);
  require(lean_io_result_is_error(result.get()), "Dropout did not return IO.error");
  require(tensor->m_rc == 1 && invalid->m_rc == 1, "Dropout changed borrowed argument ownership");
}
void dropoutSuccess() {
  auto tensor = full({4}, 7);
  for (auto probability : {0.0, 1.0}) {
        Object result(lean_torch_dropout(tensor.get(), probability, 1), lean_dec);
    require(lean_io_result_is_ok(result.get()), "Valid dropout failed");
    values(lean_io_result_get_value(result.get()), 4, probability == 0 ? 7 : 0);
    require(tensor->m_rc == 1, "Dropout ownership imbalance on success");
  }
  Object inference(lean_torch_dropout(tensor.get(), 0.5, 0), lean_dec);
  require(lean_io_result_is_ok(inference.get()), "Inference dropout failed");
  values(lean_io_result_get_value(inference.get()), 4, 7);
}
void transpose(bool invalid, bool huge = false) {
  auto input = full({2, 3}, 4);
    Object result(lean_torch_transpose(input.get(), 0,
    huge ? std::numeric_limits<uint64_t>::max() : invalid ? 9 : 1), lean_dec);
  if (invalid || huge) exceptError(result.get(), "transpose");
  else {
    auto* tensor = exceptValue(result.get());
    values(tensor, 6, 4);
    Object actual(lean_torch_get_shape(tensor), lean_dec);
    require(lean_array_size(actual.get()) == 2 &&
      lean_unbox_uint64(lean_array_get_core(actual.get(), 0)) == 3 &&
      lean_unbox_uint64(lean_array_get_core(actual.get(), 1)) == 2, "Transpose shape changed");
  }
  require(input->m_rc == 1, "Transpose ownership imbalance");
}
void transposeScalar() {
  auto input = full({}, 5);
  Object result(lean_torch_transpose(input.get(), 0, 0), lean_dec);
  values(exceptValue(result.get()), 1, 5);
}

Object cast(lean_object* input, tyr::lean::DType dtype) {
  Object dims(lean_torch_get_shape(input), lean_dec);
  Object result(lean_torch_cast_dtype_checked(retain(dims.get()), input,
                                              static_cast<uint8_t>(dtype)), lean_dec);
  return Object(retain(exceptValue(result.get())), lean_dec);
}

void checkedExpand(bool invalid) {
  auto input = full({2}, 3);
  Object target(shape({invalid ? 3u : 2u}), lean_dec);
  Object result(CHECKED(lean_torch_expand)(input.get(), retain(target.get())), lean_dec);
  if (invalid) exceptError(result.get(), "expand");
  else values(exceptValue(result.get()), 2, 3);
  require(target->m_rc == 1 && input->m_rc == 1,
    "checked expand ownership imbalance");
}

void checkedCat(bool invalid) {
  auto first = full({2, 2}, 3), second = full({invalid ? 3u : 2u, 2}, 3);
  Object tensors(lean_alloc_array(2, 2), lean_dec);
  lean_array_set_core(tensors.get(), 0, retain(first.get()));
  lean_array_set_core(tensors.get(), 1, retain(second.get()));
  Object result(CHECKED(lean_torch_cat_dyn)(tensors.get(), lean_int64_to_int(1)), lean_dec);
  if (invalid) exceptError(result.get(), "catMany");
  else values(exceptValue(result.get()), 8, 3);
  require(tensors->m_rc == 1 && first->m_rc == 2 && second->m_rc == 2,
    "checked cat changed borrowed array ownership");
}

using Reduction = lean_object* (*)(lean_object*, lean_object*, uint8_t);
void checkedReduction(Reduction operation, const char* context, bool invalidAxis, bool integral = false) {
  auto input = integral ? Object(lean_torch_full_int(shape({2}), 3), lean_dec) : full({2}, 3);
  Object axes(some(shape({invalidAxis ? 1u : 0u})), lean_dec);
  Object result(operation(input.get(), retain(axes.get()), 0), lean_dec);
  if (invalidAxis || integral) exceptError(result.get(), context);
  else values(exceptValue(result.get()), 1, std::string(context) == "sum" ? 6 : 3);
  require(axes->m_rc == 1 && input->m_rc == 1,
    "checked reduction ownership imbalance");
}

void checkedMax(bool invalid) {
  auto input = full({invalid ? 0u : 2u}, 3);
  Object result(CHECKED(lean_torch_max_all)(input.get()), lean_dec);
  if (invalid) exceptError(result.get(), "maxAll");
  else values(exceptValue(result.get()), 1, 3);
  require(input->m_rc == 1, "checked max changed borrowed tensor ownership");
}

void checkedRelu(bool invalid) {
  auto original = full({2}, 3);
  auto input = invalid ? cast(original.get(), tyr::lean::DType::Complex64) : Object(retain(original.get()), lean_dec);
    const auto refs = input->m_rc;
  Object result(CHECKED(lean_torch_tensor_relu)(input.get()), lean_dec);
  if (invalid) exceptError(result.get(), "relu");
  else values(exceptValue(result.get()), 2, 3);
  require(input->m_rc == refs, "checked relu ownership imbalance");
}

using Binary = lean_object* (*)(lean_object*, lean_object*);
void checkedBinary(Binary operation, const char* context, bool invalid) {
  auto a = full({2}, 6), b = full({invalid ? 3u : 2u}, 2);
  Object result(operation(a.get(), b.get()), lean_dec);
  if (invalid) exceptError(result.get(), context);
  else values(exceptValue(result.get()), 2,
    std::string(context) == "add" ? 8 : std::string(context) == "sub" ? 4 :
    std::string(context) == "div" ? 3 : 12);
  require(a->m_rc == 1 && b->m_rc == 1, "checked arithmetic ownership imbalance");
}

using Scalar = lean_object* (*)(lean_object*, double);
void checkedScalar(Scalar operation, const char* context, bool invalid) {
  auto original = full({2}, 6);
  auto input = invalid ? cast(original.get(), tyr::lean::DType::Float8E4M3FN) : Object(retain(original.get()), lean_dec);
  const auto refs = input->m_rc;
  Object result(operation(input.get(), 2), lean_dec);
  if (invalid) exceptError(result.get(), context);
  else values(exceptValue(result.get()), 2,
    std::string(context) == "addScalar" ? 8 : std::string(context) == "subScalar" ? 4 :
    std::string(context) == "divScalar" ? 3 : 12);
  require(input->m_rc == refs, "checked scalar arithmetic ownership imbalance");
}

void checkedWhere(bool invalid) {
  auto a = full({2}, 3), b = full({2}, 8);
  auto condition = invalid ? Object(retain(a.get()), lean_dec) : cast(a.get(), tyr::lean::DType::Bool);
  Object result(CHECKED(lean_torch_where)(condition.get(), a.get(), b.get()), lean_dec);
  if (invalid) exceptError(result.get(), "whereSelect");
  else values(exceptValue(result.get()), 2, 3);
}

void checkedMatrices(bool invalid) {
  auto original = full({2, 2}, 2);
  auto input = invalid ? cast(original.get(), tyr::lean::DType::Bool) : Object(retain(original.get()), lean_dec);
  const auto refs = input->m_rc;
  auto check = [&](lean_object* raw, const char* context) {
    Object result(raw, lean_dec);
    if (invalid) exceptError(result.get(), context);
    else values(exceptValue(result.get()), 4, 8);
    require(input->m_rc == refs, "checked matrix operation changed borrowed tensor ownership");
  };
  check(CHECKED(lean_torch_mm)(input.get(), input.get()), "mm");
  check(CHECKED(lean_torch_matmul)(input.get(), input.get()), "matmul");
  check(CHECKED(lean_torch_linear)(input.get(), input.get()), "linear");
  check(CHECKED(lean_torch_linear3d)(input.get(), input.get()), "linear3d");
  check(CHECKED(lean_torch_matmul3d_2d)(input.get(), input.get()), "matmul3d");
  check(CHECKED(lean_torch_bmm4d)(input.get(), input.get()), "bmm4d");
}

void checkedEarlyFailure() {
  Object invalid(lean_mk_string("not a tensor"), lean_dec);
  Object target(shape({2}), lean_dec), axes(some(shape({0})), lean_dec);
  Object expanded(CHECKED(lean_torch_expand)(invalid.get(), retain(target.get())), lean_dec);
  exceptError(expanded.get(), "borrowTensor");
  Object reduced(CHECKED(lean_torch_sum)(invalid.get(), retain(axes.get()), 0), lean_dec);
  exceptError(reduced.get(), "borrowTensor");
  require(invalid->m_rc == 1 && target->m_rc == 1 && axes->m_rc == 1,
    "checked early failure leaked owned parameters or changed borrowed argument");
}

void checkedExceptionKinds() {
  static_assert(noexcept(tyr::checkedCall("test", []() -> lean_object* { return lean_box(0); })));
  Object owned(shape({2}), lean_dec);
  auto check = [&](int kind) {
    Object result(tyr::checkedCall("firewall", [&]() -> lean_object* {
      Object transferred(retain(owned.get()), lean_dec);
      if (kind == 0) TORCH_CHECK(false, "libtorch test error");
      if (kind == 1) throw std::runtime_error("standard test error");
      throw 7;
    }), lean_dec);
    exceptError(result.get(), kind == 0 ? "libtorch test error" :
      kind == 1 ? "standard test error" : "unknown native exception");
    require(owned->m_rc == 1, "Exception firewall failed to unwind native ownership guards");
  };
  for (int kind : {0, 1, 2}) check(kind);
}

void checkedIoExceptionKinds() {
  static_assert(noexcept(tyr::checkedIoCall("test", []() -> lean_object* {
    return lean_io_result_mk_ok(lean_box(0));
  })));
  Object owned(shape({2}), lean_dec);
  for (int kind : {0, 1, 2}) {
    Object result(tyr::checkedIoCall("IO firewall", [&]() -> lean_object* {
      Object transferred(retain(owned.get()), lean_dec);
      if (kind == 0) TORCH_CHECK(false, "libtorch test error");
      if (kind == 1) throw std::runtime_error("standard test error");
      throw 7;
    }), lean_dec);
    require(lean_io_result_is_error(result.get()), "IO firewall did not return IO.error");
    require(owned->m_rc == 1, "IO firewall failed to unwind native ownership guards");
  }
  Object success(tyr::checkedIoCall("IO firewall", [] {
    return lean_io_result_mk_ok(lean_box(23));
  }), lean_dec);
  require(lean_io_result_is_ok(success.get()) &&
    lean_unbox(lean_io_result_get_value(success.get())) == 23,
    "IO firewall changed the successful result carrier");
}

void checkedComplexDivision() {
  auto real = full({2}, 3);
  auto complex = cast(real.get(), tyr::lean::DType::Complex32);
  Object result(CHECKED(lean_torch_div)(complex.get(), complex.get()), lean_dec);
  exceptError(result.get(), "ComplexHalf");
}

void checkedMixedMatrix() {
  auto real = full({2, 2}, 3);
  auto other = cast(real.get(), tyr::lean::DType::Float64);
  Object result(CHECKED(lean_torch_mm)(real.get(), other.get()), lean_dec);
  exceptError(result.get(), "dtype");
}

void checkedAffine(int mismatch) {
  auto input = full({1, 2, 2}, 3), originalWeight = full({2, 2}, 2), originalBias = full({2}, 1);
  auto weight = mismatch == 1 ? cast(originalWeight.get(), tyr::lean::DType::Float64) :
    Object(retain(originalWeight.get()), lean_dec);
  auto bias = mismatch == 2 ? cast(originalBias.get(), tyr::lean::DType::Float64) :
    Object(retain(originalBias.get()), lean_dec);
  const auto weightRefs = weight->m_rc, biasRefs = bias->m_rc;
  Object result(CHECKED(lean_torch_affine3d)(
    input.get(), weight.get(), bias.get()), lean_dec);
  if (mismatch) exceptError(result.get(), "affine3d");
  else values(exceptValue(result.get()), 4, 13);
  require(input->m_rc == 1 && weight->m_rc == weightRefs && bias->m_rc == biasRefs,
    "checked affine changed borrowed tensor ownership");
}

void checkedSingularValueNorm(bool nuclear, bool invalid) {
  auto input = invalid ? Object(lean_torch_full_int(shape({2, 2}), 3), lean_dec) : full({2, 2}, 3);
#ifdef TYR_CHECKED_BASELINE
  Object result(lean_box_float(nuclear ? lean_torch_nuclear_norm(input.get()) :
    lean_torch_spectral_norm(input.get())), lean_dec);
#else
  Object result(nuclear ? lean_torch_nuclear_norm_checked(2, 2, input.get()) :
    lean_torch_spectral_norm_checked(2, 2, input.get()), lean_dec);
#endif
  if (invalid) exceptError(result.get(), nuclear ? "nuclearNorm" : "spectralNorm");
  else {
    require(lean_obj_tag(result.get()) == 1, "Norm did not return Except.ok");
    require(std::abs(lean_unbox_float(lean_ctor_get(result.get(), 0)) - 6) < 1e-5,
      "Norm returned an incorrect value");
  }
  require(input->m_rc == 1, "checked norm changed borrowed tensor ownership");
}

using Unary = lean_object* (*)(lean_object*);
void checkedUnary(Unary operation, const char* context, bool invalid) {
  auto input = full({2}, 1);
  Object bad(lean_mk_string("not a tensor"), lean_dec);
  Object result(operation(invalid ? bad.get() : input.get()), lean_dec);
  if (invalid) exceptError(result.get(), context);
  else {
    Object array(lean_torch_tensor_to_float_array_dynamic(exceptValue(result.get())), lean_dec);
    require(lean_io_result_is_ok(array.get()), "Checked unary success has invalid tensor payload");
    auto* data = lean_io_result_get_value(array.get());
    require(lean_array_size(data) == 2 && std::isfinite(lean_unbox_float(lean_array_get_core(data, 0))),
      "Checked unary returned wrong shape or invalid values for unit input");
  }
  require(input->m_rc == 1 && bad->m_rc == 1, "Checked unary ownership imbalance");
}

void checkedMaximum(bool nanLeft, bool nanRight) {
  auto a = full({2}, nanLeft ? std::numeric_limits<double>::quiet_NaN() : 2);
  auto b = full({2}, nanRight ? std::numeric_limits<double>::quiet_NaN() : 3);
#ifdef TYR_CHECKED_BASELINE
  Object condition(lean_torch_gt(a.get(), b.get()), lean_dec);
  Object result(lean_torch_where(condition.get(), a.get(), b.get()), lean_dec);
  auto* tensor = result.get();
#else
  Object result(lean_torch_maximum_checked(a.get(), b.get()), lean_dec);
  auto* tensor = exceptValue(result.get());
#endif
  Object array(lean_torch_tensor_to_float_array_dynamic(tensor), lean_dec);
  require(lean_io_result_is_ok(array.get()), "Could not inspect maximum result");
  auto* data = lean_io_result_get_value(array.get());
  require(lean_array_size(data) == 2, "maximum changed shape");
  const auto value = lean_unbox_float(lean_array_get_core(data, 0));
  require((nanLeft || nanRight) ? std::isnan(value) : value == 3,
    "maximum must propagate NaN from either operand");
}

void checkedUnaryParameterErrors() {
  auto input = full({2}, 1);
    Object softmax(CHECKED(lean_torch_tensor_softmax)(input.get(), 9), lean_dec);
  exceptError(softmax.get(), "softmax");
  Object logSoftmax(CHECKED(lean_torch_log_softmax)(input.get(), 9), lean_dec);
  exceptError(logSoftmax.get(), "logSoftmax");
  auto complex = cast(input.get(), tyr::lean::DType::Complex64);
  Object gelu(CHECKED(lean_torch_tensor_gelu)(complex.get()), lean_dec);
  exceptError(gelu.get(), "gelu");
  auto fp8 = cast(input.get(), tyr::lean::DType::Float8E4M3FN);
  Object pow(CHECKED(lean_torch_pow)(fp8.get(), 2), lean_dec);
  exceptError(pow.get(), "pow");
  require(input->m_rc == 1, "Checked parameterized unary ownership imbalance");
}

void checkedComparison(Binary operation, const char* context, bool invalid) {
  auto a = full({2}, 6), b = full({invalid ? 3u : 2u}, 2);
  Object result(operation(a.get(), b.get()), lean_dec);
  if (invalid) exceptError(result.get(), context);
  else values(exceptValue(result.get()), 2, std::string(context) == "lt" ? 0 : 1);
  require(a->m_rc == 1 && b->m_rc == 1, "Checked comparison ownership imbalance");
}

void checkedCatPair(bool invalid) {
  auto a = full({2, 2}, 3), b = full({invalid ? 3u : 2u, 2}, 3);
  Object result(CHECKED(lean_torch_cat2)(a.get(), b.get(),
    lean_int64_to_int(1)), lean_dec);
  if (invalid) exceptError(result.get(), "cat");
  else values(exceptValue(result.get()), 8, 3);
  require(a->m_rc == 1 && b->m_rc == 1,
    "Checked pair concatenation ownership imbalance");
}

#ifndef TYR_CHECKED_BASELINE
void checkedRMS(bool invalid) {
  auto input = full({2, 2}, 3), weight = full({invalid ? 3u : 2u}, 2);
  Object result(lean_torch_rms_norm_weighted_checked(
    input.get(), weight.get(), 0), lean_dec);
  if (invalid) exceptError(result.get(), "rmsNormWeighted");
  else values(exceptValue(result.get()), 4, 2);
  require(input->m_rc == 1 && weight->m_rc == 1,
    "checked weighted RMS ownership imbalance");
}

void checkedLinalgFailures() {
  Object input(lean_torch_full_int(shape({2, 2}), 3), lean_dec);
  Object rows(lean_torch_row_norms_checked(input.get()), lean_dec);
  exceptError(rows.get(), "rowNorms");
  Object svd(lean_torch_svd_checked(input.get()), lean_dec);
  exceptError(svd.get(), "svd");
  Object svdvals(lean_torch_svdvals_checked(input.get()), lean_dec);
  exceptError(svdvals.get(), "svdvals");
  Object l2(lean_torch_l2_norm_checked(4, input.get()), lean_dec);
  exceptError(l2.get(), "l2Norm");
  Object frob(lean_torch_frobenius_norm_checked(2, 2, input.get()), lean_dec);
  exceptError(frob.get(), "frobeniusNorm");
  Object max(lean_torch_max_row_norm_checked(2, 2, input.get()), lean_dec);
  exceptError(max.get(), "maxRowNorm");
  auto singular = full({2, 2}, 0);
  Object log(lean_torch_matrix_log_checked(2, singular.get()), lean_dec);
  exceptError(log.get(), "matrixLog");
}
#endif
void checkedArgmax(int mode) {
  auto original = full({2, mode == 2 ? 0u : 3u}, 7);
  auto input = mode == 1 ? cast(original.get(), tyr::lean::DType::Complex64) :
    Object(retain(original.get()), lean_dec);
    const auto refs = input->m_rc;
  Object invalid(lean_mk_string("not a tensor"), lean_dec);
  const uint64_t dim = mode == 3 ? std::numeric_limits<uint64_t>::max() : 1;
  Object result(CHECKED(lean_torch_argmax)(
    mode == 4 ? invalid.get() : input.get(), dim), lean_dec);
  if (mode == 0) values(exceptValue(result.get()), 2, 0);
  else exceptError(result.get(), "argmax");
  require(input->m_rc == refs && invalid->m_rc == 1,
    "checked argmax changed borrowed refs or leaked shape");
}

void checkedSelectionPair(bool topK, int mode) {
  auto original = topK ? full({2, 3}, 7) : full({1, 2, mode == 2 ? 0u : 3u}, 7);
  auto input = mode == 1 ? cast(original.get(), tyr::lean::DType::Complex64) :
    Object(retain(original.get()), lean_dec);
  const auto refs = input->m_rc;
  const uint64_t k = mode == 2 ? 4 : mode == 3 ? std::numeric_limits<uint64_t>::max() : 2;
  Object result(topK ? CHECKED(lean_torch_topk_2d)(input.get(), k, 1) :
    CHECKED(lean_torch_max_dim_3d)(input.get(), 2), lean_dec);
  if (mode != 0) exceptError(result.get(), topK ? "topKLast2d" : "maxLast3d");
  else {
    require(!lean_is_scalar(result.get()) && !lean_is_external(result.get()) &&
      lean_obj_tag(result.get()) == 1, "Expected Except.ok selection pair");
    auto* pair = lean_ctor_get(result.get(), 0);
    require(!lean_is_scalar(pair) && !lean_is_external(pair) && lean_obj_tag(pair) == 0,
      "Expected selection value/index pair");
    values(lean_ctor_get(pair, 0), topK ? 4 : 2, 7);
    require(lean_torch_get_dtype_enum(lean_ctor_get(pair, 1)) ==
              static_cast<uint8_t>(tyr::lean::DType::Int64), "Selection index dtype changed");
  }
  require(input->m_rc == refs, "checked selection changed borrowed references");
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
  run("checked/item-success", [] { checkedItem(false); });
  run("checked/item-shape-error", [] { checkedItem(true); });
  run("checked/indexSelect-success", [] { checkedIndexSelect(0); });
  run("checked/indexSelect-negative-error", [] { checkedIndexSelect(1); });
  run("checked/indexSelect-bounds-error", [] { checkedIndexSelect(2); });
  run("checked/indexSelect-empty", [] { checkedIndexSelect(3); });
  for (bool unreduced : {false, true}) {
    run(unreduced ? "checked/crossEntropyNone-success" : "checked/crossEntropy-success",
        [&] { checkedCrossEntropy(unreduced, 0); });
    run(unreduced ? "checked/crossEntropyNone-dtype-error" : "checked/crossEntropy-dtype-error",
        [&] { checkedCrossEntropy(unreduced, 1); });
    run(unreduced ? "checked/crossEntropyNone-label-error" : "checked/crossEntropy-label-error",
        [&] { checkedCrossEntropy(unreduced, 2); });
  }
  run("flash/dropout-error", flashDropout);
  run("flash/rank-error", flashRank);
  run("flash/dtype-error", flashDType);
  run("flash/options-early-error", [] { flashOptionCleanup(true); });
  run("flash/options-dispatch-error", [] { flashOptionCleanup(false); });
  run("flash/io-options-early-error", [] { flashOptionCleanup(true, true); });
  run("flash/io-options-dispatch-error", [] { flashOptionCleanup(false, true); });
  run("randint/bounds-error", [] { randintError(false); });
  run("randint/device-error", [] { randintError(true); });
  run("randint/signed-success", randintSigned);
  run("randn/device-error", [] { randomFactoryError(true); });
  run("randn/shape-error", [] { randomFactoryError(false); });
  run("randn/success", [] { randomFactorySuccess(); });
  run("dropout/negative-probability", [] { dropoutError(-0.5); });
  run("dropout/excessive-probability", [] { dropoutError(1.5); });
  run("dropout/nan-probability", [] { dropoutError(std::numeric_limits<double>::quiet_NaN()); });
  run("dropout/invalid-carrier", [] { dropoutError(0.5, true); });
  run("dropout/success", dropoutSuccess);
  run("transpose/success-carrier", [] { transpose(false); });
  run("transpose/axis-error", [] { transpose(true); });
  run("transpose/wrapped-axis-error", [] { transpose(false, true); });
  run("transpose/scalar-success", transposeScalar);
  run("checked/argmax-success", [] { checkedArgmax(0); });
  run("checked/argmax-complex-error", [] { checkedArgmax(1); });
  run("checked/argmax-empty-error", [] { checkedArgmax(2); });
  run("checked/argmax-wrapped-axis-error", [] { checkedArgmax(3); });
  run("checked/argmax-owned-shape-early-error", [] { checkedArgmax(4); });
  run("checked/maxLast3d-success", [] { checkedSelectionPair(false, 0); });
  run("checked/maxLast3d-complex-error", [] { checkedSelectionPair(false, 1); });
  run("checked/maxLast3d-empty-error", [] { checkedSelectionPair(false, 2); });
  run("checked/topKLast2d-success", [] { checkedSelectionPair(true, 0); });
  run("checked/topKLast2d-complex-error", [] { checkedSelectionPair(true, 1); });
  run("checked/topKLast2d-range-error", [] { checkedSelectionPair(true, 2); });
  run("checked/topKLast2d-wrapped-k-error", [] { checkedSelectionPair(true, 3); });
  run("checked/expand-error", [] { checkedExpand(true); });
  run("checked/expand-success", [] { checkedExpand(false); });
  run("checked/cat-error", [] { checkedCat(true); });
  run("checked/cat-success", [] { checkedCat(false); });
  run("checked/sum-axis-error", [] { checkedReduction(CHECKED(lean_torch_sum), "sum", true); });
  run("checked/sum-success", [] { checkedReduction(CHECKED(lean_torch_sum), "sum", false); });
  run("checked/mean-axis-error", [] { checkedReduction(CHECKED(lean_torch_mean), "mean", true); });
  run("checked/mean-integral-error", [] { checkedReduction(CHECKED(lean_torch_mean), "mean", false, true); });
  run("checked/mean-success", [] { checkedReduction(CHECKED(lean_torch_mean), "mean", false); });
  run("checked/max-empty-error", [] { checkedMax(true); });
  run("checked/max-success", [] { checkedMax(false); });
  run("checked/relu-complex-error", [] { checkedRelu(true); });
  run("checked/relu-success", [] { checkedRelu(false); });
  for (bool invalid : {false, true}) {
    run(invalid ? "checked/add-error" : "checked/add-success", [&] { checkedBinary(CHECKED(lean_torch_tensor_add), "add", invalid); });
    run(invalid ? "checked/sub-error" : "checked/sub-success", [&] { checkedBinary(CHECKED(lean_torch_tensor_sub), "sub", invalid); });
    run(invalid ? "checked/mul-error" : "checked/mul-success", [&] { checkedBinary(CHECKED(lean_torch_tensor_mul), "mul", invalid); });
    run(invalid ? "checked/div-error" : "checked/div-success", [&] { checkedBinary(CHECKED(lean_torch_div), "div", invalid); });
    run(invalid ? "checked/mulTensorScalar-error" : "checked/mulTensorScalar-success", [&] { checkedBinary(CHECKED(lean_torch_mul_tensor_scalar), "mulTensorScalar", invalid); });
    run(invalid ? "checked/addScalar-error" : "checked/addScalar-success", [&] { checkedScalar(CHECKED(lean_torch_add_scalar), "addScalar", invalid); });
    run(invalid ? "checked/subScalar-error" : "checked/subScalar-success", [&] { checkedScalar(CHECKED(lean_torch_sub_scalar), "subScalar", invalid); });
    run(invalid ? "checked/mulScalar-error" : "checked/mulScalar-success", [&] { checkedScalar(CHECKED(lean_torch_mul_scalar), "mulScalar", invalid); });
    run(invalid ? "checked/divScalar-error" : "checked/divScalar-success", [&] { checkedScalar(CHECKED(lean_torch_div_scalar), "divScalar", invalid); });
    run(invalid ? "checked/where-error" : "checked/where-success", [&] { checkedWhere(invalid); });
    run(invalid ? "checked/matrices-kernel-error" : "checked/matrices-success", [&] { checkedMatrices(invalid); });
  }
  run("checked/owned-arguments-early-error", checkedEarlyFailure);
  run("checked/exception-kinds", checkedExceptionKinds);
  run("checked/IO-exception-kinds", checkedIoExceptionKinds);
  run("checked/Complex32-division-kernel-error", checkedComplexDivision);
  run("checked/mm-mixed-dtype-error", checkedMixedMatrix);
  run("checked/affine-success", [] { checkedAffine(0); });
  run("checked/affine-weight-dtype-error", [] { checkedAffine(1); });
  run("checked/affine-bias-dtype-error", [] { checkedAffine(2); });
  run("checked/spectral-success", [] { checkedSingularValueNorm(false, false); });
  run("checked/spectral-domain-error", [] { checkedSingularValueNorm(false, true); });
  run("checked/nuclear-success", [] { checkedSingularValueNorm(true, false); });
  run("checked/nuclear-domain-error", [] { checkedSingularValueNorm(true, true); });
  for (bool invalid : {false, true}) {
    run(invalid ? "checked/exp-error" : "checked/exp-success", [&] { checkedUnary(CHECKED(lean_torch_tensor_exp), "exp", invalid); });
    run(invalid ? "checked/sqrt-error" : "checked/sqrt-success", [&] { checkedUnary(CHECKED(lean_torch_sqrt), "sqrt", invalid); });
    run(invalid ? "checked/rsqrt-error" : "checked/rsqrt-success", [&] { checkedUnary(CHECKED(lean_torch_rsqrt), "rsqrt", invalid); });
    run(invalid ? "checked/sigmoid-error" : "checked/sigmoid-success", [&] { checkedUnary(CHECKED(lean_torch_tensor_sigmoid), "sigmoid", invalid); });
    run(invalid ? "checked/tanh-error" : "checked/tanh-success", [&] { checkedUnary(CHECKED(lean_torch_tensor_tanh), "tanh", invalid); });
    run(invalid ? "checked/gelu-error" : "checked/gelu-success", [&] { checkedUnary(CHECKED(lean_torch_tensor_gelu), "gelu", invalid); });
    run(invalid ? "checked/silu-error" : "checked/silu-success", [&] { checkedUnary(CHECKED(lean_torch_tensor_silu), "silu", invalid); });
    run(invalid ? "checked/log-error" : "checked/log-success", [&] { checkedUnary(CHECKED(lean_torch_tensor_log), "log", invalid); });
    run(invalid ? "checked/log10-error" : "checked/log10-success", [&] { checkedUnary(CHECKED(lean_torch_tensor_log10), "log10", invalid); });
    run(invalid ? "checked/sin-error" : "checked/sin-success", [&] { checkedUnary(CHECKED(lean_torch_tensor_sin), "sin", invalid); });
    run(invalid ? "checked/cos-error" : "checked/cos-success", [&] { checkedUnary(CHECKED(lean_torch_tensor_cos), "cos", invalid); });
    run(invalid ? "checked/atan-error" : "checked/atan-success", [&] { checkedUnary(CHECKED(lean_torch_tensor_atan), "atan", invalid); });
    run(invalid ? "checked/elu-error" : "checked/elu-success", [&] { checkedUnary(CHECKED(lean_torch_tensor_elu), "elu", invalid); });
  }
  run("checked/maximum-success", [] { checkedMaximum(false, false); });
  run("checked/maximum-left-nan", [] { checkedMaximum(true, false); });
  run("checked/maximum-right-nan", [] { checkedMaximum(false, true); });
  run("checked/unary-parameter-errors", checkedUnaryParameterErrors);
  for (bool invalid : {false, true}) {
    run(invalid ? "checked/gt-error" : "checked/gt-success", [&] { checkedComparison(CHECKED(lean_torch_gt), "gt", invalid); });
    run(invalid ? "checked/lt-error" : "checked/lt-success", [&] { checkedComparison(CHECKED(lean_torch_lt), "lt", invalid); });
    run(invalid ? "checked/ge-error" : "checked/ge-success", [&] { checkedComparison(CHECKED(lean_torch_ge), "ge", invalid); });
    run(invalid ? "checked/logicalAnd-error" : "checked/logicalAnd-success", [&] { checkedComparison(CHECKED(lean_torch_logical_and), "logicalAnd", invalid); });
    run(invalid ? "checked/cat-pair-error" : "checked/cat-pair-success", [&] { checkedCatPair(invalid); });
  }
#ifndef TYR_CHECKED_BASELINE
  run("checked/rms-success", [] { checkedRMS(false); });
  run("checked/rms-shape-error", [] { checkedRMS(true); });
  run("checked/linalg-errors", checkedLinalgFailures);
#endif
  std::cout << failures << " native error-carrier failures\n";
  return failures == 0 ? 0 : 1;
}
