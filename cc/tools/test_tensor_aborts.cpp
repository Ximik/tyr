// Invalid calls into unguarded exports. Each must abort through Tyr's terminate
// handler; scripts/test_native_tensor_errors.sh runs every case in its own process.
// Without arguments, prints `NAME<TAB>FRAGMENT` per case; with `--abort-case NAME`,
// runs that case.
#include "tensor_test_support.h"
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>

namespace {
struct AbortCase {
  std::string name;
  std::string fragment;
  std::function<void()> call;
};

Object notATensor() { return Object(lean_mk_string("not a tensor"), lean_dec); }

void crossEntropy(bool unreduced, bool integralLogits) {
  Object logits(integralLogits ? lean_torch_full_int(shape({2, 2}), 0)
                               : lean_torch_full(shape({2, 2}), 0, 0, lean_box(1)), lean_dec);
  Object labels(lean_torch_full_int(shape({2}), integralLogits ? 0 : 2), lean_dec);
  Object result(unreduced ? lean_torch_cross_entropy_none(logits.get(), labels.get())
                          : lean_torch_cross_entropy_2d(logits.get(), labels.get()), lean_dec);
}

void argmax(lean_object* input, uint64_t dim) {
  Object result(lean_torch_argmax(input, Nat(dim)), lean_dec);
}

void maxLast3d(Object input) {
  Object result(lean_torch_max_dim_3d(input.get(), Nat(2)), lean_dec);
}

void topK(Object input, uint64_t k) {
  Object result(lean_torch_topk_2d(input.get(), Nat(k), Nat(1)), lean_dec);
}

void reduce(lean_object* (*operation)(lean_object*, lean_object*, uint8_t), Object input, uint64_t axis) {
  Object result(operation(input.get(), some(shape({axis})), 0), lean_dec);
}

void binaryShapes(Binary operation) {
  auto a = full({2}, 6), b = full({3}, 2);
  Object result(operation(a.get(), b.get()), lean_dec);
}

void scalarFloat8(lean_object* (*operation)(lean_object*, double)) {
  auto input = cast(full({2}, 6).get(), tyr::lean::DType::Float8E4M3FN);
  Object result(operation(input.get(), 2), lean_dec);
}

void indexSelect(Object indices) {
  auto input = full({2}, 3.25);
  Object result(lean_torch_index_select(input.get(), Nat(0), indices.get()), lean_dec);
}

void flash(Object q) {
  Object result(lean_torch_tyr_flash_attn_4d(q.get(), q.get(), q.get(), lean_box(0), 0, lean_box(0), 0), lean_dec);
}

void affineDType(bool weight) {
  auto input = full({1, 2, 2}, 3);
  auto w = weight ? cast(full({2, 2}, 2).get(), tyr::lean::DType::Float64) : full({2, 2}, 2);
  auto b = weight ? full({2}, 1) : cast(full({2}, 1).get(), tyr::lean::DType::Float64);
  Object result(lean_torch_affine3d(input.get(), w.get(), b.get()), lean_dec);
}

std::vector<AbortCase> cases() {
  std::vector<AbortCase> all;
  auto add = [&](std::string name, std::string fragment, std::function<void()> call) {
    all.push_back({std::move(name), std::move(fragment), std::move(call)});
  };
  for (bool unreduced : {false, true}) {
    const std::string loss = unreduced ? "crossEntropyNone" : "crossEntropy";
    add(loss + "-dtype", "real floating logits", [=] { crossEntropy(unreduced, true); });
    add(loss + "-label", "outside the class range", [=] { crossEntropy(unreduced, false); });
  }
  add("argmax-complex", "does not support complex", [] { argmax(cast(full({2, 3}, 7).get(), tyr::lean::DType::Complex64).get(), 1); });
  add("argmax-empty", "non-zero size", [] { argmax(full({2, 0}, 7).get(), 1); });
  add("argmax-wrapped-axis", "dim exceeds", [] { argmax(full({2, 3}, 7).get(), std::numeric_limits<uint64_t>::max()); });
  add("argmax-not-a-tensor", "borrowTensor", [] { argmax(notATensor().get(), 1); });
  add("maxLast3d-complex", "does not support complex", [] { maxLast3d(cast(full({1, 2, 3}, 7).get(), tyr::lean::DType::Complex64)); });
  add("maxLast3d-empty", "non-zero size", [] { maxLast3d(full({1, 2, 0}, 7)); });
  add("topKLast2d-complex", "does not support complex", [] { topK(cast(full({2, 3}, 7).get(), tyr::lean::DType::Complex64), 2); });
  add("topKLast2d-range", "k out of range", [] { topK(full({2, 3}, 7), 4); });
  add("topKLast2d-wrapped-k", "k exceeds", [] { topK(full({2, 3}, 7), std::numeric_limits<uint64_t>::max()); });
  add("sum-axis", "Dimension out of range", [] { reduce(lean_torch_sum, full({2}, 3), 1); });
  add("mean-axis", "Dimension out of range", [] { reduce(lean_torch_mean, full({2}, 3), 1); });
  add("mean-integral", "floating point or complex", [] { reduce(lean_torch_mean, Object(lean_torch_full_int(shape({2}), 3), lean_dec), 0); });
  add("sum-not-a-tensor", "borrowTensor", [] { reduce(lean_torch_sum, notATensor(), 0); });
  add("maxAll-empty", "numel() == 0", [] { auto input = full({0}, 3); Object r(lean_torch_max_all(input.get()), lean_dec); });
  add("relu-complex", "not supported for complex", [] {
    auto input = cast(full({2}, 3).get(), tyr::lean::DType::Complex64);
    Object r(lean_torch_tensor_relu(input.get()), lean_dec);
  });
  add("add-shape", "must match the size", [] { binaryShapes(lean_torch_tensor_add); });
  add("sub-shape", "must match the size", [] { binaryShapes(lean_torch_tensor_sub); });
  add("mul-shape", "must match the size", [] { binaryShapes(lean_torch_tensor_mul); });
  add("div-shape", "must match the size", [] { binaryShapes(lean_torch_div); });
  add("mulTensorScalar-shape", "must match the size", [] { binaryShapes(lean_torch_mul_tensor_scalar); });
  add("addScalar-float8", "not implemented for 'Float8_e4m3fn'", [] { scalarFloat8(lean_torch_add_scalar); });
  add("subScalar-float8", "not implemented for 'Float8_e4m3fn'", [] { scalarFloat8(lean_torch_sub_scalar); });
  add("mulScalar-float8", "not implemented for 'Float8_e4m3fn'", [] { scalarFloat8(lean_torch_mul_scalar); });
  add("divScalar-float8", "not implemented for 'Float8_e4m3fn'", [] { scalarFloat8(lean_torch_div_scalar); });
  add("where-float-condition", "boolean tensor", [] {
    auto a = full({2}, 3), b = full({2}, 8);
    Object r(lean_torch_where(a.get(), a.get(), b.get()), lean_dec);
  });
  for (const auto& [name, operation] : matrixOps)
    add(std::string(name) + "-bool", "not implemented for 'Bool'", [operation = operation] {
      auto input = cast(full({2, 2}, 2).get(), tyr::lean::DType::Bool);
      Object r(operation(input.get(), input.get()), lean_dec);
    });
  add("div-complex32-kernel", "ComplexHalf", [] {
    auto input = cast(full({2}, 3).get(), tyr::lean::DType::Complex32);
    Object r(lean_torch_div(input.get(), input.get()), lean_dec);
  });
  add("mm-mixed-dtype", "dtype", [] {
    auto real = full({2, 2}, 3);
    auto other = cast(real.get(), tyr::lean::DType::Float64);
    Object r(lean_torch_mm(real.get(), other.get()), lean_dec);
  });
  add("indexSelect-negative", "indexSelect index is out of range",
      [] { indexSelect(Object(lean_torch_full_int(shape({1}), std::numeric_limits<uint64_t>::max()), lean_dec)); });
  add("indexSelect-bounds", "indexSelect index is out of range",
      [] { indexSelect(Object(lean_torch_full_int(shape({1}), 2), lean_dec)); });
  add("indexSelect-float-indices", "requires Int64 or Int32 indices", [] { indexSelect(full({1}, 0)); });
  add("logicalAnd-shape", "must match the size", [] {
    auto a = full({2}, 6), b = full({3}, 2);
    Object r(lean_torch_logical_and(a.get(), b.get()), lean_dec);
  });
  add("cat-shape", "Sizes of tensors must match", [] {
    auto a = full({2, 2}, 3), b = full({3, 2}, 3);
    Object r(lean_torch_cat2(a.get(), b.get(), Nat(1)), lean_dec);
  });
  add("transpose-axis", "dimension out of range", [] {
    auto input = full({2, 3}, 4);
    Object r(lean_torch_transpose(input.get(), Nat(0), Nat(9)), lean_dec);
  });
  add("transpose-wrapped-axis", "dim1 exceeds", [] {
    auto input = full({2, 3}, 4);
    Object r(lean_torch_transpose(input.get(), Nat(0), Nat(std::numeric_limits<uint64_t>::max())), lean_dec);
  });
  add("svd-try-integral", "floating point or complex", [] {
    Object input(lean_torch_full_int(shape({2, 2}), 3), lean_dec);
    Object r(lean_torch_svd_try(input.get()), lean_dec);
  });
  add("svd-try-missing-kernel", "not implemented for", [] {
    auto input = cast(full({2, 2}, 1).get(), tyr::lean::DType::Complex32);
    Object r(lean_torch_svd_try(input.get()), lean_dec);
  });
  add("flashAttention-rank", "rank-4", [] { flash(full({2, 2}, 1)); });
  add("flashAttention-dtype", "expected scalar type Float but found Long", [] { flash(Object(lean_torch_full_int(shape({1, 1, 2, 2}), 1), lean_dec)); });
  add("affine-weight-dtype", "must have the same dtype", [] { affineDType(true); });
  add("affine-bias-dtype", "must have the same dtype", [] { affineDType(false); });
  add("spectralNorm-integral", "floating point or complex", [] {
    Object input(lean_torch_full_int(shape({2, 2}), 3), lean_dec);
    lean_torch_spectral_norm(input.get());
  });
  add("nuclearNorm-integral", "floating point or complex", [] {
    Object input(lean_torch_full_int(shape({2, 2}), 3), lean_dec);
    lean_torch_nuclear_norm(input.get());
  });
  for (const auto& [name, operation] : unaryOps)
    add(std::string(name) + "-not-a-tensor", "borrowTensor", [operation = operation] {
      Object r(operation(notATensor().get()), lean_dec);
    });
  add("softmax-axis", "dimension out of range", [] { auto input = full({2}, 1); Object r(lean_torch_tensor_softmax(input.get(), Nat(9)), lean_dec); });
  add("logSoftmax-axis", "dimension out of range", [] { auto input = full({2}, 1); Object r(lean_torch_log_softmax(input.get(), Nat(9)), lean_dec); });
  add("gelu-complex", "not implemented for 'ComplexFloat'", [] {
    auto input = cast(full({2}, 1).get(), tyr::lean::DType::Complex64);
    Object r(lean_torch_tensor_gelu(input.get()), lean_dec);
  });
  add("pow-float8", "not implemented for 'Float8_e4m3fn'", [] {
    auto input = cast(full({2}, 1).get(), tyr::lean::DType::Float8E4M3FN);
    Object r(lean_torch_pow(input.get(), 2), lean_dec);
  });
  add("gt-shape", "must match the size", [] { binaryShapes(lean_torch_gt); });
  add("lt-shape", "must match the size", [] { binaryShapes(lean_torch_lt); });
  add("ge-shape", "must match the size", [] { binaryShapes(lean_torch_ge); });
  add("rmsNormWeighted-shape", "must match the size", [] {
    auto input = full({2, 2}, 3), weight = full({3}, 2);
    Object r(lean_torch_rms_norm_weighted(input.get(), weight.get(), 0), lean_dec);
  });
  auto integral = [] { return Object(lean_torch_full_int(shape({2, 2}), 3), lean_dec); };
  add("rowNorms-integral", "floating point or complex", [=] { auto i = integral(); Object r(lean_torch_row_norms(i.get()), lean_dec); });
  add("svdvals-integral", "floating point or complex", [=] { auto i = integral(); Object r(lean_torch_svdvals(i.get()), lean_dec); });
  add("l2Norm-integral", "floating point or complex", [=] { auto i = integral(); lean_torch_l2_norm(i.get()); });
  add("frobeniusNorm-integral", "floating point or complex", [=] { auto i = integral(); lean_torch_frobenius_norm(i.get()); });
  add("maxRowNorm-integral", "floating point or complex", [=] { auto i = integral(); lean_torch_max_row_norm(i.get()); });
  return all;
}
}  // namespace

int main(int argc, char** argv) {
  lean_initialize();
  const auto all = cases();
  if (argc == 1) {
    for (const auto& c : all) std::cout << c.name << '\t' << c.fragment << '\n';
    return 0;
  }
  if (argc == 3 && std::string(argv[1]) == "--abort-case") {
    for (const auto& c : all)
      if (c.name == argv[2]) { c.call(); return 0; }
    std::cerr << "unknown abort case: " << argv[2] << '\n';
    return 2;
  }
  std::cerr << "usage: " << argv[0] << " [--abort-case NAME]\n";
  return 2;
}
