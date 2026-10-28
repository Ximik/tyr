#include "generated/tyr_ffi.h"

#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

extern "C" void lean_initialize();

namespace {
using LeanObject = std::unique_ptr<lean_object, decltype(&lean_dec)>;
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

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

lean_object* shape(std::initializer_list<uint64_t> dimensions) {
  auto* result = lean_alloc_array(dimensions.size(), dimensions.size());
  size_t index = 0;
  for (auto dimension : dimensions) {
    lean_array_set_core(result, index++, lean_box(dimension));
  }
  return result;
}

lean_object* some(lean_object* value) {
  auto* result = lean_alloc_ctor(1, 1, 0);
  lean_ctor_set(result, 0, value);
  return result;
}

lean_object* retain(lean_object* value) {
  lean_inc(value);
  return value;
}

// One reference remains with the observer; transfer() gives a distinct owned
// reference to the FFI. Fresh objects avoid persistent-constant RC exemptions.
class ObservedArg {
  LeanObject object;
public:
  explicit ObservedArg(lean_object* value) : object(value, lean_dec) {
    require(!lean_is_scalar(value) && value->m_rc == 1, "Expected fresh owned object");
  }
  lean_object* get() const { return object.get(); }
  lean_object* transfer() const { return retain(object.get()); }
  void consumed(const std::string& label) const {
    require(object->m_rc == 1, label + ": transferred reference was not released");
  }
};

template <typename Action>
void expectThrow(Action action) {
  try {
    LeanObject unexpected(action(), lean_dec);
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error("Expected native rejection");
}


void checkBorrowedTensor(lean_object* tensor) {
  require(tensor->m_rc == 1, "Borrowed tensor reference count changed");
  require(lean_torch_get_dtype_enum(tensor) == static_cast<uint8_t>(tyr::lean::DType::Float32),
          "Borrowed tensor is invalid");
  LeanObject values(lean_torch_get_values(tensor, Nat(4)), lean_dec);
  require(lean_sarray_size(values.get()) == 4, "Borrowed tensor shape changed");
  auto* data = reinterpret_cast<const double*>(lean_sarray_cptr(values.get()));
  for (size_t i = 0; i < 4; ++i) require(data[i] == 1.0, "Borrowed tensor contents changed");
}

void testShapesAndMetadata(lean_object* tensor) {
  using Unary = lean_object* (*)(lean_object*);
  const std::pair<const char*, Unary> calls[] = {
    {"toString", lean_torch_to_string}, {"runtimeShape", lean_torch_get_shape},
    {"device", lean_torch_get_device_enum},
    {"deviceString", lean_torch_get_device}, {"stats", lean_torch_get_stats},
    {"zerosLike", lean_torch_zeros_like}, {"onesLike", lean_torch_ones_like},
    {"abs", lean_torch_abs}, {"sqrt", lean_torch_sqrt}, {"rsqrt", lean_torch_rsqrt},
    {"relu", lean_torch_tensor_relu}, {"exp", lean_torch_tensor_exp},
    {"detach", lean_torch_detach}, {"toFloat", lean_torch_to_float},
    {"toBFloat16", lean_torch_to_bfloat16}, {"toLong", lean_torch_to_long}
  };
  for (const auto& entry : calls) {
    LeanObject result(entry.second(tensor), lean_dec);
  }
  (void)lean_torch_get_dtype_enum(tensor);
  lean_dec(lean_torch_get_values(tensor, Nat(4)));
  lean_dec(lean_torch_add_scalar(tensor, 2));
  lean_dec(lean_torch_sub_scalar(tensor, 2));
  lean_dec(lean_torch_mul_scalar(tensor, 2));
  lean_dec(lean_torch_div_scalar(tensor, 2));
  lean_dec(lean_torch_tensor_add(tensor, tensor));
  lean_dec(lean_torch_tensor_softmax(tensor, 1));
  lean_dec(lean_torch_copy_(retain(tensor), tensor));
  lean_dec(lean_torch_copy_inplace(retain(tensor), tensor));
  lean_dec(lean_torch_transpose(tensor, Nat(0), Nat(1)));
  checkBorrowedTensor(tensor);
}

void testOwnedArraysAndOptions(lean_object* tensor) {
  for (bool invalid : {false, true}) {
    ObservedArg permutation(shape(invalid ? std::initializer_list<uint64_t>{0, 0}
                                         : std::initializer_list<uint64_t>{1, 0}));
    auto call = [&] { return lean_torch_permute(tensor, permutation.transfer()); };
    if (invalid) expectThrow(call); else lean_dec(call());
    permutation.consumed("permute permutation");
  }
  for (auto reduction : {lean_torch_sum, lean_torch_mean}) {
    for (bool invalid : {false, true}) {
      ObservedArg dims(some(shape({invalid ? uint64_t{99} : uint64_t{1}})));
      auto call = [&] { return reduction(tensor, dims.transfer(), 0); };
      if (invalid) expectThrow(call); else lean_dec(call());
      dims.consumed("reduction option");
    }
  }
  lean_dec(lean_torch_matmul(tensor, tensor));
  ObservedArg target(shape({2, 2, 2}));
  lean_dec(lean_torch_expand(tensor, target.transfer()));
  target.consumed("expand target");

  auto* elements = lean_alloc_array(2, 2);
  lean_array_set_core(elements, 0, retain(tensor));
  lean_array_set_core(elements, 1, retain(tensor));
  {
    ObservedArg array(elements);
    lean_dec(lean_torch_stack_1d(array.transfer(), 0));
    array.consumed("stack array");
  }
  checkBorrowedTensor(tensor);
}

void testNativeFailureCleanup(lean_object* tensor) {
  LeanObject incompatible(lean_torch_ones(shape({3, 3}), 0, lean_box(1)), lean_dec);
  expectThrow([&] { return lean_torch_tensor_add(tensor, incompatible.get()); });

  auto* cuda = lean_alloc_ctor(0, 0, sizeof(uint64_t));
  lean_ctor_set_uint64(cuda, 0, std::numeric_limits<uint64_t>::max());
  ObservedArg device(cuda), factoryShape(shape({2, 2}));
  expectThrow([&] { return lean_torch_ones(factoryShape.transfer(), 0, device.transfer()); });
  factoryShape.consumed("factory shape before device rejection");
  device.consumed("factory device rejection");

  for (auto operation : {lean_torch_squeeze, lean_torch_unsqueeze, lean_torch_unbind}) {
    ObservedArg axis(lean_uint64_to_nat(std::numeric_limits<uint64_t>::max() - 100));
    expectThrow([&] { return operation(tensor, axis.transfer()); });
    axis.consumed("boxed Nat dimension");
  }

  // These deliberately invalid handles exercise failure before any option or
  // array is parsed. The RawTensor carrier normally prevents such callers;
  // cleanup must still be correct if validation fails at the native boundary.
  LeanObject invalid(lean_mk_string("not an external tensor"), lean_dec);
  ObservedArg weight(some(retain(tensor))), bias(some(retain(tensor)));
  expectThrow([&] { return lean_torch_group_norm(invalid.get(), Nat(1),
                                                weight.transfer(), bias.transfer(), 1e-5); });
  weight.consumed("groupNorm early failure weight");
  bias.consumed("groupNorm early failure bias");

  for (bool effectful : {false, true}) {
    ObservedArg mask(some(retain(tensor))), scale(some(lean_box_float(0.5)));
    auto* result = effectful
      ? lean_torch_tyr_flash_attn_4d_io(
          invalid.get(), tensor, tensor, mask.transfer(), 0, 0, scale.transfer(), 0)
      : lean_torch_tyr_flash_attn_4d(
          invalid.get(), tensor, tensor, mask.transfer(), 0, scale.transfer(), 0);
    if (effectful) require(lean_io_result_is_error(result), "Expected FlashAttention IO.error");
    else require(!lean_is_scalar(result) && !lean_is_external(result) && lean_obj_tag(result) == 0,
                 "Expected FlashAttention Except.error");
    lean_dec(result);
    mask.consumed("flash attention early failure mask");
    scale.consumed("flash attention early failure scale");
  }
}

void testModelArguments(lean_object* tensor) {
  LeanObject input(lean_torch_ones(shape({1, 1, 3, 3}), 0, lean_box(1)), lean_dec);
  LeanObject weight(lean_torch_ones(shape({1, 1, 1, 1}), 0, lean_box(1)), lean_dec);
  LeanObject bias(lean_torch_zeros(shape({1}), 0, lean_box(1)), lean_dec);
  ObservedArg stride(shape({1, 1})), padding(shape({0, 0})), dilation(shape({1, 1}));
  lean_dec(lean_torch_conv2d_bias(
    input.get(), weight.get(), bias.get(), stride.transfer(), padding.transfer(), dilation.transfer()));
  stride.consumed("conv2d stride"); padding.consumed("conv2d padding"); dilation.consumed("conv2d dilation");

  ObservedArg size(shape({6, 6})), mode(lean_mk_string("nearest"));
  lean_dec(lean_torch_interpolate(input.get(), size.transfer(), mode.transfer(), 0));
  size.consumed("interpolate size"); mode.consumed("interpolate mode");

  ObservedArg optionalWeight(some(retain(tensor))), reduction(lean_mk_string("mean"));
  lean_dec(lean_torch_binary_cross_entropy(tensor, tensor,
    optionalWeight.transfer(), reduction.transfer()));
  optionalWeight.consumed("BCE optional weight"); reduction.consumed("BCE reduction");
}
} // namespace

int main() {
  lean_initialize();
  try {
    LeanObject tensor(lean_torch_ones(shape({2, 2}), 0, lean_box(1)), lean_dec);
    testShapesAndMetadata(tensor.get());
    testOwnedArraysAndOptions(tensor.get());
    testNativeFailureCleanup(tensor.get());
    testModelArguments(tensor.get());
    checkBorrowedTensor(tensor.get());
    std::cout << "Native tensor ownership checks passed (success, exception, and borrowed arguments).\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Native tensor ownership check failed: " << error.what() << '\n';
    return 1;
  }
}
