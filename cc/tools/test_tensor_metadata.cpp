#include "tyr_tensor_metadata.h"
#include "generated/tyr_ffi.h"
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
void lean_initialize_runtime_module();
lean_object* initialize_Init(uint8_t);
lean_object* lean_torch_get_dtype(lean_object*, lean_object*);
lean_object* lean_torch_get_device_enum(lean_object*, lean_object*);
lean_object* lean_torch_convert_tensor(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_cast_dtype_checked(lean_object*, lean_object*, lean_object*);
}

namespace {

using LeanObject = std::unique_ptr<lean_object, decltype(&lean_dec)>;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void phase(const std::string& label) {
  std::cout << "[metadata] " << label << std::endl;
}

void requireIoSuccess(lean_object* result, const std::string& context) {
  if (lean_io_result_is_ok(result)) return;
  std::cerr << "Native IO error during " << context << ":\n";
  lean_io_result_show_error(result);
  throw std::runtime_error(context + " returned an IO error");
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
  constexpr auto cuda = static_cast<unsigned>(tyr::lean::DeviceKind::CUDA);
  constexpr auto cpu = static_cast<unsigned>(tyr::lean::DeviceKind::CPU);
  constexpr auto mps = static_cast<unsigned>(tyr::lean::DeviceKind::MPS);
  const auto max_index = static_cast<uint64_t>(std::numeric_limits<c10::DeviceIndex>::max());
  require(deviceFromLeanTag(cpu).is_cpu(), "CPU must decode to CPU");
  require(deviceFromLeanTag(cuda, 0).index() == 0, "CUDA zero must retain its index");
  require(deviceFromLeanTag(cuda, max_index).index() == max_index,
          "Largest representable CUDA index must remain unchanged");
  for (const auto index : {max_index + 1, uint64_t{255}, uint64_t{256},
                          std::numeric_limits<uint64_t>::max()}) {
    requireError("CUDA device index", [=] { (void)deviceFromLeanTag(cuda, index); });
  }
  requireError("Invalid Lean Device constructor tag", [] { (void)deviceFromLeanTag(mps + 1); });
#ifdef __APPLE__
  require(deviceFromLeanTag(mps).is_mps(), "Apple MPS must decode to MPS without CPU fallback");
#else
  requireError("MPS devices are unsupported", [] { (void)deviceFromLeanTag(mps); });
#endif
  require(leanDeviceTag(c10::Device(c10::DeviceType::CPU)) == cpu,
          "Native CPU must encode as CPU");
  require(leanDeviceTag(c10::Device(c10::DeviceType::CUDA, 0)) == cuda,
          "Native CUDA must encode as CUDA");
  require(leanDeviceTag(c10::Device(c10::DeviceType::MPS)) == mps,
          "Native MPS must encode as MPS");
  requireError("Unsupported native tensor device", [] {
    (void)leanDeviceTag(c10::Device(c10::DeviceType::Meta));
  });
  requireError("no concrete index", [] {
    (void)leanDeviceTag(c10::Device(c10::DeviceType::CUDA));
  });
}

void testDtypeRegistry() {
  using namespace tyr::metadata;
  constexpr auto count = static_cast<size_t>(c10::ScalarType::Undefined);
  static_assert(dtypeInfos.size() == count, "Update the complete PyTorch dtype inventory");
  std::array<bool, count> seen{};
  for (unsigned tag = 0; tag < dtypeInfos.size(); ++tag) {
    const auto& info = dtypeInfos[tag];
    const auto ordinal = static_cast<size_t>(info.scalar_type);
    require(ordinal < count && !seen[ordinal], "Duplicate or invalid native dtype mapping");
    seen[ordinal] = true;
    require(scalarTypeFromLeanDTypeTag(tag) == info.scalar_type, "Lean dtype tag decode mismatch");
    require(leanDTypeTag(info.scalar_type) == tag, "Lean dtype tag encode mismatch");
    require(std::string(c10::toString(info.scalar_type)) == info.native_name,
            "Native dtype alias mismatch");
    for (const auto& name : {std::string(info.canonical_name), std::string(info.native_name),
                             std::string(info.torch_name), "torch." + std::string(info.torch_name)}) {
      require(parseDtype(name) == info.scalar_type, "Dtype parse mismatch: " + name);
    }
    const auto native_names = c10::getDtypeNames(info.scalar_type);
    for (const auto& name : {native_names.first, native_names.second}) {
      if (name.empty()) continue;
      require(parseDtype(name) == info.scalar_type &&
              parseDtype("torch." + name) == info.scalar_type,
              "Runtime-reported dtype alias mismatch: " + name);
    }
    require(parseDtype(dtypeName(info.scalar_type)) == info.scalar_type,
            "Canonical native dtype metadata must roundtrip");
  }
  for (bool present : seen) require(present, "Missing native ScalarType entry");
  require(!scalarTypeFromLeanDTypeTag(static_cast<unsigned>(tyr::lean::DType::Unknown)), "Unknown payload is not a concrete dtype tag");
  require(!parseDtype("Undefined") && !parseDtype("NumOptions") && !parseDtype("dynamic"),
          "Sentinels and wildcard tokens must not request a native dtype");
  require(parseDtype("chalf") == c10::ScalarType::ComplexHalf &&
          parseDtype("cfloat") == c10::ScalarType::ComplexFloat &&
          parseDtype("cdouble") == c10::ScalarType::ComplexDouble,
          "Python complex aliases must retain their component widths");
  require(c10::elementSize(*parseDtype("Complex32")) == 4 &&
          c10::elementSize(*parseDtype("Complex64")) == 8 &&
          c10::elementSize(*parseDtype("Complex128")) == 16,
          "Complex names count the total bits of both components");
}

// Exercise actual RawTensor load and metadata exports, not just the helper.
// CPU complex tensors need no accelerator and retain exact canonical widths.
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
  requireIoSuccess(result.get(), "RawTensor load " + name);
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
  require(lean_is_scalar(device.get()) &&
              lean_unbox(device.get()) == static_cast<unsigned>(tyr::lean::DeviceKind::CPU),
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
  phase("conversion: " + context);
  auto option = dtypeOption(dtype);
  LeanObject result(lean_torch_convert_tensor(lean_alloc_array(0, 0), source,
                                            device, option.get()), lean_dec);
  requireIoSuccess(result.get(), context);
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
  const std::string expected_dtype = tyr::metadata::dtypeName(dtype);
  require(std::string(lean_string_cstr(native_dtype.get())) == expected_dtype,
          context + " reported the wrong dtype");

  const auto path = directory / "converted_scalar.pt";
  LeanObject path_object(lean_mk_string(path.string().c_str()), lean_dec);
  LeanObject saved(lean_torch_save_tensor(lean_alloc_array(0, 0), raw, path_object.get()), lean_dec);
  requireIoSuccess(saved.get(), context + " save for value inspection");
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

  for (const char* tag : {"QInt8", "Bits1x8", "not-a-dtype"}) {
    LeanObject target(lean_mk_string(tag), lean_dec);
    LeanObject result(lean_torch_cast_dtype_checked(lean_alloc_array(0, 0), input.get(),
                                                   target.get()), lean_dec);
    require(lean_obj_tag(result.get()) == 0,
            std::string("Unsupported checked cast must return Except.error: ") + tag);
    require(std::string(lean_string_cstr(lean_ctor_get(result.get(), 0))).find("castDType:") == 0,
            "Checked cast errors must retain operation context");
    auto option = dtypeOption(tag);
    LeanObject moved(lean_torch_convert_tensor(lean_alloc_array(0, 0), input.get(),
                                             lean_box(1), option.get()), lean_dec);
    require(lean_io_result_is_error(moved.get()),
            std::string("Unsupported combined cast must return IO.error: ") + tag);
  }

  // None retains the actual native dtype, including both complex widths.
  for (const auto dtype : {c10::ScalarType::Float, c10::ScalarType::Double,
                           c10::ScalarType::ComplexFloat, c10::ScalarType::ComplexDouble}) {
    const auto name = std::string("retain_") + c10::toString(dtype);
    auto original = loadThroughRawABI(directory, dtype, name);
    auto retained = convertThroughRawABI(original.get(), lean_box(1), nullptr, name);
    requireConverted(directory, retained.get(), dtype, c10::DeviceType::CPU, name);
  }

  auto complex_input = loadThroughRawABI(directory, c10::ScalarType::ComplexFloat, "cast_complex64");
  auto complex_output = convertThroughRawABI(complex_input.get(), lean_box(1),
                                            "torch.complex128", "Complex64 to Complex128");
  requireConverted(directory, complex_output.get(), c10::ScalarType::ComplexDouble,
                   c10::DeviceType::CPU, "Complex64 to Complex128");
  LeanObject complex_tag(lean_mk_string("Complex128"), lean_dec);
  LeanObject checked_complex(lean_torch_cast_dtype_checked(lean_alloc_array(0, 0),
                              complex_input.get(), complex_tag.get()), lean_dec);
  require(lean_obj_tag(checked_complex.get()) == 1, "Complex checked cast must return Except.ok");
  requireConverted(directory, lean_ctor_get(checked_complex.get(), 0),
                   c10::ScalarType::ComplexDouble, c10::DeviceType::CPU,
                   "Checked Complex64 to Complex128");

  auto* wrong_shape = lean_alloc_array(1, 1);
  lean_array_set_core(wrong_shape, 0, lean_box_uint64(2));
  LeanObject shape_error(lean_torch_cast_dtype_checked(wrong_shape, complex_input.get(),
                          complex_tag.get()), lean_dec);
  require(lean_obj_tag(shape_error.get()) == 0 &&
          std::string(lean_string_cstr(lean_ctor_get(shape_error.get(), 0)))
              .find("source tensor shape mismatch") != std::string::npos,
          "Checked dtype conversion must validate the supplied source shape");

  for (const char* tag : {"not-a-dtype", "ComplexFloat_typo"}) {
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

  LeanObject availability(lean_torch_mps_is_available(), lean_dec);
  requireIoSuccess(availability.get(), "MPS availability query");
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
    // Match Lean 4.29's generated runtime-only entry point. These ABI tests
    // need Init's exported IO helpers, but do not use Std or the Lean frontend.
    phase("initialize Lean runtime");
    lean_initialize_runtime_module();
    phase("initialize Init");
    LeanObject initialized(initialize_Init(1), lean_dec);
    requireIoSuccess(initialized.get(), "Init initialization");
    lean_io_mark_end_initialization();
    phase("device conversion checks");
    testDeviceConversion();
    phase("dtype registry checks");
    testDtypeRegistry();
    phase("RawTensor metadata roundtrips");
    const auto f32 = dtypeThroughRawABI(argv[1], c10::ScalarType::Float, "float32");
    const auto c64 = dtypeThroughRawABI(argv[1], c10::ScalarType::ComplexFloat, "complex64");
    const auto c128 = dtypeThroughRawABI(argv[1], c10::ScalarType::ComplexDouble, "complex128");
    require(f32 == "Float32", "Known dtype canonical spelling changed: " + f32);
    require(c64 == "Complex64", "ComplexFloat metadata must report Complex64: " + c64);
    require(c128 == "Complex128", "ComplexDouble metadata must report Complex128: " + c128);
    require(c64 != c128, "Different complex widths must remain distinguishable");
    phase("checked dtype and device conversions");
    testCombinedConversion(argv[1]);
    std::cout << "Native tensor metadata checks passed (device bounds/platform, Meta rejection, RawTensor dtype and conversion ABI).\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Native tensor metadata check failed: " << error.what() << '\n';
    return 1;
  }
}
