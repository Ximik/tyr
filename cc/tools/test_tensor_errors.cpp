#include "tensor_test_support.h"
#include <bit>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>

// Tests-first mode applies the checked contract to the previous raw boundary.
// It must fail: native errors escape and successful results lack Except.ok.
#ifdef TYR_CHECKED_BASELINE
#define CHECKED(F) F
#else
#define CHECKED(F) F##_checked
#endif

namespace {
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
void crossEntropy(bool unreduced) {
  Object logits(lean_torch_full(shape({2, 2}), 0, 0, lean_box(1)), lean_dec);
  Object labels(lean_torch_full_int(shape({2}), 0), lean_dec);
  Object result(unreduced ? lean_torch_cross_entropy_none(logits.get(), labels.get())
                          : lean_torch_cross_entropy_2d(logits.get(), labels.get()), lean_dec);
  Object observed(lean_torch_tensor_to_float_array_dynamic(result.get()), lean_dec);
  require(lean_io_result_is_ok(observed.get()), "Could not inspect loss values");
  auto* array = lean_io_result_get_value(observed.get());
  require(lean_array_size(array) == (unreduced ? 2 : 1), "Unexpected loss result shape");
  for (size_t i = 0; i < lean_array_size(array); ++i)
    require(std::abs(lean_unbox_float(lean_array_get_core(array, i)) - std::log(2.0)) < 1e-6,
            "Incorrect cross entropy value");
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

void indexSelect(bool empty) {
  const uint64_t count = empty ? 0 : 1;
  Object input(lean_torch_full(shape({empty ? uint64_t{0} : uint64_t{2}}), 3.25, 0, lean_box(1)), lean_dec);
  Object indices(lean_torch_full_int(shape({count}), 0), lean_dec);
  Object result(lean_torch_index_select(input.get(), Nat(0), indices.get()), lean_dec);
  values(result.get(), count, 3.25);
  Object checked(lean_torch_index_select_checked(input.get(), Nat(0), indices.get()), lean_dec);
  values(exceptValue(checked.get()), count, 3.25);
  require(input->m_rc == 1 && indices->m_rc == 1,
          "Index selection changed borrowed references");
}

void checkedIndexSelectError(uint64_t id) {
  Object input(lean_torch_full(shape({2}), 3.25, 0, lean_box(1)), lean_dec);
  Object indices(lean_torch_full_int(shape({1}), id), lean_dec);
  Object result(lean_torch_index_select_checked(input.get(), Nat(0), indices.get()), lean_dec);
  exceptError(result.get(), "indexSelect");
  require(input->m_rc == 1 && indices->m_rc == 1,
          "Index selection changed borrowed references");
}

void flashSuccess() {
  auto q = full({1, 1, 2, 2}), v = full({1, 1, 2, 2}, 7);
  auto result = flash(q.get(), q.get(), v.get());
  values(result.get(), 4, 7);
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
void flashOptionCleanup(bool invalidCarrier, bool effectful = false) {
  auto q = full({1, 1, 2, 2}), maskTensor = full({1, 3});
  Object invalid(lean_mk_string("not a tensor"), lean_dec);
  Object mask(some(retain(maskTensor.get())), lean_dec), scale(some(lean_box_float(0.5)), lean_dec);
  auto* carrier = invalidCarrier ? invalid.get() : q.get();
  if (effectful) {
    auto result = flashIO(carrier, q.get(), q.get(), 0, retain(mask.get()), retain(scale.get()));
    require(lean_io_result_is_error(result.get()), "Expected IO.error for invalid FlashAttention inputs");
  } else {
    expectThrow([&] { return lean_torch_tyr_flash_attn_4d(
      carrier, q.get(), q.get(), retain(mask.get()), 0, retain(scale.get()), 0); });
  }
  require(mask->m_rc == 1 && scale->m_rc == 1, "FlashAttention leaked owned options on failure");
  require(q->m_rc == 1 && maskTensor->m_rc == 2, "FlashAttention changed borrowed tensor ownership");
}
void randintError(bool invalidDevice) {
  Object dims(shape({4}), lean_dec);
  auto* cuda = lean_alloc_ctor(0, 1, 0);
  lean_ctor_set(cuda, 0, lean_uint64_to_nat(std::numeric_limits<uint64_t>::max()));
  Object device(invalidDevice ? cuda : lean_box(1), lean_dec);
  if (!invalidDevice) lean_dec(cuda);
  Object result(lean_torch_randint(3, invalidDevice ? 4 : 3, retain(dims.get()),
    retain(device.get())), lean_dec);
  require(lean_io_result_is_error(result.get()), "randint did not return IO.error");
  require(dims->m_rc == 1, "randint leaked shape on failure");
  if (invalidDevice) require(device->m_rc == 1, "randint leaked device on failure");
}
void randintSigned() {
  Object result(lean_torch_randint(std::bit_cast<uint64_t>(int64_t{-7}),
    std::bit_cast<uint64_t>(int64_t{-6}), shape({4}), lean_box(1)), lean_dec);
  require(lean_io_result_is_ok(result.get()), "Valid signed randint failed");
  values(lean_io_result_get_value(result.get()), 4, -7);
}
void randomFactoryError(bool invalidDevice) {
  Object dims(invalidDevice ? shape({4}) : shape({std::numeric_limits<uint64_t>::max()}), lean_dec);
  auto* cuda = lean_alloc_ctor(0, 1, 0);
  lean_ctor_set(cuda, 0, lean_uint64_to_nat(std::numeric_limits<uint64_t>::max()));
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
      lean_unbox(lean_array_get_core(actual.get(), 0)) == 2 &&
      lean_unbox(lean_array_get_core(actual.get(), 1)) == 3,
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
    Object result(lean_torch_transpose(input.get(), Nat(0),
    Nat(huge ? std::numeric_limits<uint64_t>::max() : invalid ? 9 : 1)), lean_dec);
  if (invalid || huge) exceptError(result.get(), "transpose");
  else {
    auto* tensor = exceptValue(result.get());
    values(tensor, 6, 4);
    Object actual(lean_torch_get_shape(tensor), lean_dec);
    require(lean_array_size(actual.get()) == 2 &&
      lean_unbox(lean_array_get_core(actual.get(), 0)) == 3 &&
      lean_unbox(lean_array_get_core(actual.get(), 1)) == 2, "Transpose shape changed");
  }
  require(input->m_rc == 1, "Transpose ownership imbalance");
}
void transposeScalar() {
  auto input = full({}, 5);
  Object result(lean_torch_transpose(input.get(), Nat(0), Nat(0)), lean_dec);
  values(exceptValue(result.get()), 1, 5);
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
void reduction(Reduction operation, double expected) {
  auto input = full({2}, 3);
  Object axes(some(shape({0})), lean_dec);
  Object result(operation(input.get(), retain(axes.get()), 0), lean_dec);
  values(result.get(), 1, expected);
  require(axes->m_rc == 1 && input->m_rc == 1, "reduction ownership imbalance");
}

void maxAll() {
  auto input = full({2}, 3);
  Object result(lean_torch_max_all(input.get()), lean_dec);
  values(result.get(), 1, 3);
  require(input->m_rc == 1, "max changed borrowed tensor ownership");
}

void relu() {
  auto input = full({2}, 3);
  Object result(lean_torch_tensor_relu(input.get()), lean_dec);
  values(result.get(), 2, 3);
  require(input->m_rc == 1, "relu ownership imbalance");
}

void binary(Binary operation, double expected) {
  auto a = full({2}, 6), b = full({2}, 2);
  Object result(operation(a.get(), b.get()), lean_dec);
  values(result.get(), 2, expected);
  require(a->m_rc == 1 && b->m_rc == 1, "arithmetic ownership imbalance");
}

using Scalar = lean_object* (*)(lean_object*, double);
void scalar(Scalar operation, double expected) {
  auto input = full({2}, 6);
  Object result(operation(input.get(), 2), lean_dec);
  values(result.get(), 2, expected);
  require(input->m_rc == 1, "scalar arithmetic ownership imbalance");
}

void where() {
  auto a = full({2}, 3), b = full({2}, 8);
  auto condition = cast(a.get(), tyr::lean::DType::Bool);
  Object result(lean_torch_where(condition.get(), a.get(), b.get()), lean_dec);
  values(result.get(), 2, 3);
}

void matrix(Binary operation) {
  auto input = full({2, 2}, 2);
  Object result(operation(input.get(), input.get()), lean_dec);
  values(result.get(), 4, 8);
  require(input->m_rc == 1, "matrix operation changed borrowed tensor ownership");
}

void checkedEarlyFailure() {
  Object invalid(lean_mk_string("not a tensor"), lean_dec);
  Object target(shape({2}), lean_dec);
  Object expanded(CHECKED(lean_torch_expand)(invalid.get(), retain(target.get())), lean_dec);
  exceptError(expanded.get(), "borrowTensor");
  require(invalid->m_rc == 1 && target->m_rc == 1,
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

void affine() {
  auto input = full({1, 2, 2}, 3), weight = full({2, 2}, 2), bias = full({2}, 1);
  Object result(lean_torch_affine3d(input.get(), weight.get(), bias.get()), lean_dec);
  values(result.get(), 4, 13);
  require(input->m_rc == 1 && weight->m_rc == 1 && bias->m_rc == 1,
    "affine changed borrowed tensor ownership");
}

void singularValueNorm(bool nuclear) {
  auto input = full({2, 2}, 3);
  const double norm = nuclear ? lean_torch_nuclear_norm(input.get()) : lean_torch_spectral_norm(input.get());
  require(std::abs(norm - 6) < 1e-5, "Norm returned an incorrect value");
  require(input->m_rc == 1, "norm changed borrowed tensor ownership");
}

void unary(Unary operation) {
  auto input = full({2}, 1);
  Object result(operation(input.get()), lean_dec);
  Object array(lean_torch_tensor_to_float_array_dynamic(result.get()), lean_dec);
  require(lean_io_result_is_ok(array.get()), "Unary result has an invalid tensor payload");
  auto* data = lean_io_result_get_value(array.get());
  require(lean_array_size(data) == 2 && std::isfinite(lean_unbox_float(lean_array_get_core(data, 0))),
    "Unary returned wrong shape or invalid values for unit input");
  require(input->m_rc == 1, "Unary ownership imbalance");
}

void maximum(bool nanLeft, bool nanRight) {
  auto a = full({2}, nanLeft ? std::numeric_limits<double>::quiet_NaN() : 2);
  auto b = full({2}, nanRight ? std::numeric_limits<double>::quiet_NaN() : 3);
  Object result(lean_torch_maximum(a.get(), b.get()), lean_dec);
  Object array(lean_torch_tensor_to_float_array_dynamic(result.get()), lean_dec);
  require(lean_io_result_is_ok(array.get()), "Could not inspect maximum result");
  auto* data = lean_io_result_get_value(array.get());
  require(lean_array_size(data) == 2, "maximum changed shape");
  const auto value = lean_unbox_float(lean_array_get_core(data, 0));
  require((nanLeft || nanRight) ? std::isnan(value) : value == 3,
    "maximum must propagate NaN from either operand");
  require(a->m_rc == 1 && b->m_rc == 1, "maximum changed borrowed references");
}

void comparison(Binary operation, double expected) {
  auto a = full({2}, 6), b = full({2}, 2);
  Object result(operation(a.get(), b.get()), lean_dec);
  values(result.get(), 2, expected);
  require(a->m_rc == 1 && b->m_rc == 1, "comparison ownership imbalance");
}

void logicalAnd() {
  auto a = full({2}, 6), b = full({2}, 2);
  Object result(lean_torch_logical_and(a.get(), b.get()), lean_dec);
  values(result.get(), 2, 1);
  require(a->m_rc == 1 && b->m_rc == 1, "logicalAnd ownership imbalance");
}

void catPair() {
  auto a = full({2, 2}, 3), b = full({2, 2}, 3);
  Object result(lean_torch_cat2(a.get(), b.get(), Nat(1)), lean_dec);
  values(result.get(), 8, 3);
  require(a->m_rc == 1 && b->m_rc == 1, "Pair concatenation ownership imbalance");
}

void rms() {
  auto input = full({2, 2}, 3), weight = full({2}, 2);
  Object result(lean_torch_rms_norm_weighted(input.get(), weight.get(), 0), lean_dec);
  values(result.get(), 4, 2);
  require(input->m_rc == 1 && weight->m_rc == 1, "weighted RMS ownership imbalance");
}

#ifndef TYR_CHECKED_BASELINE
void checkedLinalgFailures() {
  Object input(lean_torch_full_int(shape({2, 2}), 3), lean_dec);
  Object svd(lean_torch_svd_checked(input.get()), lean_dec);
  exceptError(svd.get(), "svd");
  auto singular = full({2, 2}, 0);
  Object log(lean_torch_matrix_log_checked(singular.get()), lean_dec);
  exceptError(log.get(), "matrixLog");
}
#endif

void argmax() {
  auto input = full({2, 3}, 7);
  Object result(lean_torch_argmax(input.get(), Nat(1)), lean_dec);
  values(result.get(), 2, 0);
  require(input->m_rc == 1, "argmax changed borrowed refs");
}

void selectionPair(bool topK) {
  auto input = topK ? full({2, 3}, 7) : full({1, 2, 3}, 7);
  Object pair(topK ? lean_torch_topk_2d(input.get(), Nat(2), Nat(1)) :
    lean_torch_max_dim_3d(input.get(), Nat(2)), lean_dec);
  require(!lean_is_scalar(pair.get()) && !lean_is_external(pair.get()) && lean_obj_tag(pair.get()) == 0,
    "Expected selection value/index pair");
  values(lean_ctor_get(pair.get(), 0), topK ? 4 : 2, 7);
  require(lean_torch_get_dtype_enum(lean_ctor_get(pair.get(), 1)) ==
            static_cast<uint8_t>(tyr::lean::DType::Int64), "Selection index dtype changed");
  require(input->m_rc == 1, "selection changed borrowed references");
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
  run("indexSelect/success", [] { indexSelect(false); });
  run("indexSelect/empty", [] { indexSelect(true); });
  run("checked/indexSelect-negative-error", [] { checkedIndexSelectError(UINT64_MAX); });
  run("checked/indexSelect-bounds-error", [] { checkedIndexSelectError(2); });
  run("flash/dropout-error", flashDropout);
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
  run("checked/expand-error", [] { checkedExpand(true); });
  run("checked/expand-success", [] { checkedExpand(false); });
  run("checked/cat-error", [] { checkedCat(true); });
  run("checked/cat-success", [] { checkedCat(false); });
  run("checked/owned-arguments-early-error", checkedEarlyFailure);
  run("checked/exception-kinds", checkedExceptionKinds);
  run("checked/IO-exception-kinds", checkedIoExceptionKinds);
  run("logicalAnd/success", logicalAnd);
  run("cat-pair/success", catPair);
#ifndef TYR_CHECKED_BASELINE
  run("checked/linalg-errors", checkedLinalgFailures);
#endif
  run("crossEntropy-success", [] { crossEntropy(false); });
  run("crossEntropyNone-success", [] { crossEntropy(true); });
  run("argmax-success", argmax);
  run("maxLast3d-success", [] { selectionPair(false); });
  run("topKLast2d-success", [] { selectionPair(true); });
  run("sum-success", [] { reduction(lean_torch_sum, 6); });
  run("mean-success", [] { reduction(lean_torch_mean, 3); });
  run("maxAll-success", maxAll);
  run("relu-success", relu);
  run("add-success", [] { binary(lean_torch_tensor_add, 8); });
  run("sub-success", [] { binary(lean_torch_tensor_sub, 4); });
  run("mul-success", [] { binary(lean_torch_tensor_mul, 12); });
  run("div-success", [] { binary(lean_torch_div, 3); });
  run("mulTensorScalar-success", [] { binary(lean_torch_mul_tensor_scalar, 12); });
  run("addScalar-success", [] { scalar(lean_torch_add_scalar, 8); });
  run("subScalar-success", [] { scalar(lean_torch_sub_scalar, 4); });
  run("mulScalar-success", [] { scalar(lean_torch_mul_scalar, 12); });
  run("divScalar-success", [] { scalar(lean_torch_div_scalar, 3); });
  run("where-success", where);
  for (const auto& [name, operation] : matrixOps)
    run((std::string(name) + "-success").c_str(), [operation = operation] { matrix(operation); });
  run("affine-success", affine);
  run("spectralNorm-success", [] { singularValueNorm(false); });
  run("nuclearNorm-success", [] { singularValueNorm(true); });
  for (const auto& [name, operation] : unaryOps)
    run((std::string(name) + "-success").c_str(), [operation = operation] { unary(operation); });
  run("maximum-success", [] { maximum(false, false); });
  run("maximum-left-nan", [] { maximum(true, false); });
  run("maximum-right-nan", [] { maximum(false, true); });
  run("gt-success", [] { comparison(lean_torch_gt, 1); });
  run("lt-success", [] { comparison(lean_torch_lt, 0); });
  run("ge-success", [] { comparison(lean_torch_ge, 1); });
  run("rmsNormWeighted-success", rms);
  std::cout << failures << " native error-carrier failures\n";
  return failures == 0 ? 0 : 1;
}
