#pragma once

#include "generated/tyr_ffi.h"
#include "tyr_try_result.h"
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

extern "C" void lean_initialize();

namespace {
using Object = std::unique_ptr<lean_object, decltype(&lean_dec)>;
// Temporary Lean Nat for a borrowed argument; values above the small-Nat range
// become bignums, as Lean would pass them.
struct Nat {
  lean_object* object;
  Nat(uint64_t value) : object(lean_uint64_to_nat(value)) {}
  Nat(const Nat&) = delete;
  Nat& operator=(const Nat&) = delete;
  ~Nat() { lean_dec(object); }
  operator lean_object*() const { return object; }
};
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
lean_object* shape(std::initializer_list<uint64_t> dims) {
  auto* result = lean_alloc_array(dims.size(), dims.size());
  size_t i = 0;
  for (auto dim : dims) lean_array_set_core(result, i++, lean_box(dim));
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
// Unguarded exports throw in this harness; from Lean the same failure aborts.
template <typename Action>
void expectThrow(Action action) {
  try {
    Object unexpected(action(), lean_dec);
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error("Expected native rejection");
}
Object cast(lean_object* input, tyr::lean::DType dtype) {
  return Object(lean_torch_cast_dtype(input, static_cast<uint8_t>(dtype)), lean_dec);
}

using Binary = lean_object* (*)(lean_object*, lean_object*);
using Unary = lean_object* (*)(lean_object*);
const std::vector<std::pair<const char*, Binary>> matrixOps = {
  {"mm", lean_torch_mm}, {"matmul", lean_torch_matmul}, {"linear", lean_torch_linear},
  {"linear3d", lean_torch_linear3d}, {"matmul3d", lean_torch_matmul3d_2d}, {"bmm4d", lean_torch_bmm4d}};
const std::vector<std::pair<const char*, Unary>> unaryOps = {
  {"exp", lean_torch_tensor_exp}, {"sqrt", lean_torch_sqrt}, {"rsqrt", lean_torch_rsqrt},
  {"sigmoid", lean_torch_tensor_sigmoid}, {"tanh", lean_torch_tensor_tanh}, {"gelu", lean_torch_tensor_gelu},
  {"silu", lean_torch_tensor_silu}, {"log", lean_torch_tensor_log}, {"log10", lean_torch_tensor_log10},
  {"sin", lean_torch_tensor_sin}, {"cos", lean_torch_tensor_cos}, {"atan", lean_torch_tensor_atan},
  {"elu", lean_torch_tensor_elu}};
}  // namespace
