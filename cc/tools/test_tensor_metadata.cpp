#include "tyr_tensor_metadata.h"
#include "tyr_neural_abi.h"
#include <ATen/Context.h>
#include <lean/lean.h>
#include <torch/torch.h>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
void lean_initialize();
lean_object* lean_torch_get_dtype(lean_object*, lean_object*);
lean_object* lean_torch_get_device_enum(lean_object*, lean_object*);
lean_object* lean_torch_convert_tensor(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_mps_is_available(lean_object*);
}

namespace {

using LeanObject = std::unique_ptr<lean_object, decltype(&lean_dec)>;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Action>
void requireError(const std::string& fragment, Action action) {
  try {
    action();
  } catch (const c10::Error& error) {
    require(std::string(error.what()).find(fragment) != std::string::npos,
            "Unexpected native error: " + std::string(error.what()));
    return;
  }
  throw std::runtime_error("Expected native rejection containing: " + fragment);
}

void testDeviceConversion() {
  using tyr::metadata::deviceFromLeanTag;
  using tyr::metadata::leanDeviceTag;
  const auto max_index = static_cast<uint64_t>(std::numeric_limits<c10::DeviceIndex>::max());
  require(deviceFromLeanTag(1).is_cpu(), "CPU must decode to CPU");
  require(deviceFromLeanTag(0, 0).index() == 0, "CUDA zero must retain its index");
  require(deviceFromLeanTag(0, max_index).index() == max_index,
          "Largest representable CUDA index must remain unchanged");
  for (const auto index : {max_index + 1, uint64_t{255}, uint64_t{256},
                          std::numeric_limits<uint64_t>::max()}) {
    requireError("CUDA device index", [=] { (void)deviceFromLeanTag(0, index); });
  }
  requireError("Invalid Lean Device constructor tag", [] { (void)deviceFromLeanTag(3); });
#ifdef __APPLE__
  require(deviceFromLeanTag(2).is_mps(), "Apple MPS must decode to MPS without CPU fallback");
#else
  requireError("MPS devices are unsupported", [] { (void)deviceFromLeanTag(2); });
#endif
  require(leanDeviceTag(c10::Device(c10::DeviceType::CPU)) == 1,
          "Native CPU must encode as CPU");
  require(leanDeviceTag(c10::Device(c10::DeviceType::CUDA, 0)) == 0,
          "Native CUDA must encode as CUDA");
  require(leanDeviceTag(c10::Device(c10::DeviceType::MPS)) == 2,
          "Native MPS must encode as MPS");
  requireError("Unsupported native tensor device", [] {
    (void)leanDeviceTag(c10::Device(c10::DeviceType::Meta));
  });
  requireError("no concrete index", [] {
    (void)leanDeviceTag(c10::Device(c10::DeviceType::CUDA));
  });
}

// Exercise actual RawTensor load and metadata exports, not just the helper.
// CPU complex tensors need no accelerator and must retain distinct unknown
// dtype tokens when Lean's DType parser does not recognize their scalar types.
torch::Tensor scalarFixture(c10::ScalarType dtype) {
  const auto options = torch::TensorOptions().dtype(dtype);
  if (c10::isComplexType(dtype)) {
    return torch::full({}, c10::Scalar(c10::complex<double>(1.25, -2.5)), options);
  }
  return torch::full({}, 1.25, options);
}

LeanObject loadThroughRawABI(const std::filesystem::path& directory,
                            c10::ScalarType dtype, const std::string& name) {
  const auto path = directory / (name + ".pt");
  torch::save(std::vector<torch::Tensor>{scalarFixture(dtype)}, path.string());
  LeanObject path_object(lean_mk_string(path.string().c_str()), lean_dec);
  LeanObject result(lean_torch_load_tensor_exact(lean_alloc_array(0, 0), path_object.get()), lean_dec);
  require(lean_io_result_is_ok(result.get()), "RawTensor load failed for " + name);
  auto* raw = lean_io_result_get_value(result.get());
  lean_inc(raw);
  return LeanObject(raw, lean_dec);
}

std::string dtypeThroughRawABI(const std::filesystem::path& directory,
                              c10::ScalarType dtype, const std::string& name) {
  auto tensor = loadThroughRawABI(directory, dtype, name);
  auto* raw = tensor.get();
  LeanObject dtype_name(lean_torch_get_dtype(lean_alloc_array(0, 0), raw), lean_dec);
  LeanObject device(lean_torch_get_device_enum(lean_alloc_array(0, 0), raw), lean_dec);
  require(lean_is_scalar(device.get()) && lean_unbox(device.get()) == 1,
          "RawTensor CPU metadata must report CPU");
  return lean_string_cstr(dtype_name.get());
}

LeanObject dtypeOption(const char* tag) {
  if (tag == nullptr) return LeanObject(lean_box(0), lean_dec);
  auto* option = lean_alloc_ctor(1, 1, 0);
  lean_ctor_set(option, 0, lean_mk_string(tag));
  return LeanObject(option, lean_dec);
}

LeanObject convertThroughRawABI(lean_object* source, lean_object* device,
                               const char* dtype, const std::string& context) {
  auto option = dtypeOption(dtype);
  LeanObject result(lean_torch_convert_tensor(lean_alloc_array(0, 0), source,
                                            device, option.get()), lean_dec);
  require(lean_io_result_is_ok(result.get()), context + " returned an IO error");
  auto* raw = lean_io_result_get_value(result.get());
  lean_inc(raw);
  return LeanObject(raw, lean_dec);
}

void requireConverted(const std::filesystem::path& directory, lean_object* raw, c10::ScalarType dtype,
                      c10::DeviceType device, const std::string& context) {
  // Inspect placement before serialization: loading onto CPU below is only a
  // way to inspect full scalar values without reaching into a RawTensor handle.
  LeanObject native_device(lean_torch_get_device_enum(lean_alloc_array(0, 0), raw), lean_dec);
  const auto expected_device = tyr::metadata::leanDeviceTag(c10::Device(device));
  require(lean_is_scalar(native_device.get()) && lean_unbox(native_device.get()) == expected_device,
          context + " returned the wrong device");
  LeanObject native_dtype(lean_torch_get_dtype(lean_alloc_array(0, 0), raw), lean_dec);
  const std::string expected_dtype = dtype == c10::ScalarType::Float ? "Float32" :
      dtype == c10::ScalarType::Double ? "Float64" : c10::toString(dtype);
  require(std::string(lean_string_cstr(native_dtype.get())) == expected_dtype,
          context + " reported the wrong dtype");

  const auto path = directory / "converted_scalar.pt";
  LeanObject path_object(lean_mk_string(path.string().c_str()), lean_dec);
  LeanObject saved(lean_torch_save_tensor(lean_alloc_array(0, 0), raw, path_object.get()), lean_dec);
  require(lean_io_result_is_ok(saved.get()), context + " failed to save for value inspection");
  std::vector<torch::Tensor> tensors;
  torch::load(tensors, path.string(), c10::Device(c10::DeviceType::CPU));
  require(tensors.size() == 1, context + " saved the wrong tensor count");
  const auto& tensor = tensors[0];
  require(tensor.scalar_type() == dtype, context + " returned the wrong dtype");
  require(tensor.dim() == 0, context + " changed the scalar shape");
  require(torch::equal(tensor, scalarFixture(dtype)), context + " changed the scalar value");
}

void testCombinedConversion(const std::filesystem::path& directory) {
  auto input = loadThroughRawABI(directory, c10::ScalarType::Float, "convert_float32");
  auto converted = convertThroughRawABI(input.get(), lean_box(1), "F64", "CPU F32 to F64");
  requireConverted(directory, converted.get(), c10::ScalarType::Double, c10::DeviceType::CPU,
                   "CPU F32 to F64");
  requireConverted(directory, input.get(), c10::ScalarType::Float, c10::DeviceType::CPU,
                   "CPU conversion source");

  // None retains the actual native dtype, even when Lean represents it only
  // through an exact Unknown token rather than a supported cast tag.
  for (const auto dtype : {c10::ScalarType::Float, c10::ScalarType::Double,
                           c10::ScalarType::ComplexFloat, c10::ScalarType::ComplexDouble}) {
    const auto name = std::string("retain_") + c10::toString(dtype);
    auto original = loadThroughRawABI(directory, dtype, name);
    auto retained = convertThroughRawABI(original.get(), lean_box(1), nullptr, name);
    requireConverted(directory, retained.get(), dtype, c10::DeviceType::CPU, name);
  }

  for (const char* tag : {"not-a-dtype", "ComplexDouble"}) {
    auto option = dtypeOption(tag);
    LeanObject result(lean_torch_convert_tensor(lean_alloc_array(0, 0), input.get(),
                                              lean_box(1), option.get()), lean_dec);
    require(lean_io_result_is_error(result.get()),
            std::string("Unsupported dtype tag must return IO.error: ") + tag);
  }
  for (const uint64_t index : {uint64_t{128}, uint64_t{255}, uint64_t{256},
                               std::numeric_limits<uint64_t>::max()}) {
    LeanObject device(lean_alloc_ctor(0, 0, 8), lean_dec);
    lean_ctor_set_uint64(device.get(), 0, index);
    LeanObject result(lean_torch_convert_tensor(lean_alloc_array(0, 0), input.get(),
                                              device.get(), lean_box(0)), lean_dec);
    require(lean_io_result_is_error(result.get()),
            "Invalid CUDA index must return IO.error before attempting allocation");
  }

  LeanObject availability(lean_torch_mps_is_available(lean_box(0)), lean_dec);
  require(lean_io_result_is_ok(availability.get()), "MPS availability query failed");
  if (!lean_unbox(lean_io_result_get_value(availability.get()))) {
    std::cout << "SKIP combined MPS conversion checks: MPS is unavailable.\n";
    return;
  }
  require(at::hasMPS(), "Native availability and ATen MPS capability disagree");
  auto mps_input = convertThroughRawABI(input.get(), lean_box(2), nullptr, "CPU F32 to MPS");
  requireConverted(directory, mps_input.get(), c10::ScalarType::Float, c10::DeviceType::MPS,
                   "CPU F32 to MPS");
  auto cpu_double = convertThroughRawABI(mps_input.get(), lean_box(1), "F64", "MPS F32 to CPU F64");
  requireConverted(directory, cpu_double.get(), c10::ScalarType::Double, c10::DeviceType::CPU,
                   "MPS F32 to CPU F64");
  auto mps_float = convertThroughRawABI(cpu_double.get(), lean_box(2), "F32", "CPU F64 to MPS F32");
  requireConverted(directory, mps_float.get(), c10::ScalarType::Float, c10::DeviceType::MPS,
                   "CPU F64 to MPS F32");
  std::cout << "Combined MPS F32/CPU F64 conversion checks passed in both directions.\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "Usage: test_tensor_metadata TEMP_DIRECTORY");
    lean_initialize();
    testDeviceConversion();
    const auto f32 = dtypeThroughRawABI(argv[1], c10::ScalarType::Float, "float32");
    const auto c64 = dtypeThroughRawABI(argv[1], c10::ScalarType::ComplexFloat, "complex64");
    const auto c128 = dtypeThroughRawABI(argv[1], c10::ScalarType::ComplexDouble, "complex128");
    require(f32 == "Float32", "Known dtype canonical spelling changed: " + f32);
    require(c64 == "ComplexFloat", "ComplexFloat metadata was erased: " + c64);
    require(c128 == "ComplexDouble", "ComplexDouble metadata was erased: " + c128);
    require(c64 != c128, "Different unsupported dtypes must remain distinguishable");
    testCombinedConversion(argv[1]);
    std::cout << "Native tensor metadata checks passed (device bounds/platform, Meta rejection, RawTensor dtype and conversion ABI).\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Native tensor metadata check failed: " << error.what() << '\n';
    return 1;
  }
}
