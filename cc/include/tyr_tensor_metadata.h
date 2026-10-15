#pragma once

#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <c10/util/Exception.h>
#include <cstdint>
#include <limits>

namespace tyr::metadata {

// This registry follows Lean DType constructor order, not native enum order.
// Keep existing concrete tags 0..11 stable; Unknown is the payload constructor
// at tag 46 and intentionally has no native ScalarType mapping.
struct DTypeInfo {
  c10::ScalarType scalar_type;
  const char* canonical_name;
  const char* native_name;
  const char* torch_name;
};

inline constexpr std::array<DTypeInfo, 46> dtypeInfos{{
    {c10::ScalarType::Byte, "UInt8", "Byte", "uint8"},
    {c10::ScalarType::Char, "Int8", "Char", "int8"},
    {c10::ScalarType::Short, "Int16", "Short", "int16"},
    {c10::ScalarType::Int, "Int32", "Int", "int32"},
    {c10::ScalarType::Long, "Int64", "Long", "int64"},
    {c10::ScalarType::Half, "Float16", "Half", "float16"},
    {c10::ScalarType::BFloat16, "BFloat16", "BFloat16", "bfloat16"},
    {c10::ScalarType::Float, "Float32", "Float", "float32"},
    {c10::ScalarType::Double, "Float64", "Double", "float64"},
    {c10::ScalarType::Bool, "Bool", "Bool", "bool"},
    {c10::ScalarType::Float8_e4m3fn, "Float8E4M3FN", "Float8_e4m3fn", "float8_e4m3fn"},
    {c10::ScalarType::Float8_e5m2, "Float8E5M2", "Float8_e5m2", "float8_e5m2"},
    {c10::ScalarType::ComplexHalf, "Complex32", "ComplexHalf", "complex32"},
    {c10::ScalarType::ComplexFloat, "Complex64", "ComplexFloat", "complex64"},
    {c10::ScalarType::ComplexDouble, "Complex128", "ComplexDouble", "complex128"},
    {c10::ScalarType::QInt8, "QInt8", "QInt8", "qint8"},
    {c10::ScalarType::QUInt8, "QUInt8", "QUInt8", "quint8"},
    {c10::ScalarType::QInt32, "QInt32", "QInt32", "qint32"},
    {c10::ScalarType::QUInt4x2, "QUInt4x2", "QUInt4x2", "quint4x2"},
    {c10::ScalarType::QUInt2x4, "QUInt2x4", "QUInt2x4", "quint2x4"},
    {c10::ScalarType::Bits1x8, "Bits1x8", "Bits1x8", "bits1x8"},
    {c10::ScalarType::Bits2x4, "Bits2x4", "Bits2x4", "bits2x4"},
    {c10::ScalarType::Bits4x2, "Bits4x2", "Bits4x2", "bits4x2"},
    {c10::ScalarType::Bits8, "Bits8", "Bits8", "bits8"},
    {c10::ScalarType::Bits16, "Bits16", "Bits16", "bits16"},
    {c10::ScalarType::Float8_e5m2fnuz, "Float8E5M2FNUZ", "Float8_e5m2fnuz", "float8_e5m2fnuz"},
    {c10::ScalarType::Float8_e4m3fnuz, "Float8E4M3FNUZ", "Float8_e4m3fnuz", "float8_e4m3fnuz"},
    {c10::ScalarType::UInt16, "UInt16", "UInt16", "uint16"},
    {c10::ScalarType::UInt32, "UInt32", "UInt32", "uint32"},
    {c10::ScalarType::UInt64, "UInt64", "UInt64", "uint64"},
    {c10::ScalarType::UInt1, "UInt1", "UInt1", "uint1"},
    {c10::ScalarType::UInt2, "UInt2", "UInt2", "uint2"},
    {c10::ScalarType::UInt3, "UInt3", "UInt3", "uint3"},
    {c10::ScalarType::UInt4, "UInt4", "UInt4", "uint4"},
    {c10::ScalarType::UInt5, "UInt5", "UInt5", "uint5"},
    {c10::ScalarType::UInt6, "UInt6", "UInt6", "uint6"},
    {c10::ScalarType::UInt7, "UInt7", "UInt7", "uint7"},
    {c10::ScalarType::Int1, "Int1", "Int1", "int1"},
    {c10::ScalarType::Int2, "Int2", "Int2", "int2"},
    {c10::ScalarType::Int3, "Int3", "Int3", "int3"},
    {c10::ScalarType::Int4, "Int4", "Int4", "int4"},
    {c10::ScalarType::Int5, "Int5", "Int5", "int5"},
    {c10::ScalarType::Int6, "Int6", "Int6", "int6"},
    {c10::ScalarType::Int7, "Int7", "Int7", "int7"},
    {c10::ScalarType::Float8_e8m0fnu, "Float8E8M0FNU", "Float8_e8m0fnu", "float8_e8m0fnu"},
    {c10::ScalarType::Float4_e2m1fn_x2, "Float4E2M1FNX2", "Float4_e2m1fn_x2", "float4_e2m1fn_x2"},
}};

inline std::optional<c10::ScalarType> scalarTypeFromLeanDTypeTag(unsigned tag) {
  if (tag >= dtypeInfos.size()) return std::nullopt;
  return dtypeInfos[tag].scalar_type;
}

inline unsigned leanDTypeTag(c10::ScalarType dtype) {
  for (unsigned tag = 0; tag < dtypeInfos.size(); ++tag) {
    if (dtypeInfos[tag].scalar_type == dtype) return tag;
  }
  TORCH_CHECK(false, "Unsupported native ScalarType sentinel: ", static_cast<int>(dtype));
}

inline const char* dtypeName(c10::ScalarType dtype) {
  return dtypeInfos[leanDTypeTag(dtype)].canonical_name;
}

inline std::string normalizeDtypeName(std::string_view name) {
  std::string token(name);
  for (char& c : token) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  if (token.compare(0, 6, "torch.") == 0) token.erase(0, 6);
  return token;
}

// Parsing recognizes dtypes, not supported operations on those dtypes.
// SafeTensors must continue to use its narrower storage-format parser.
inline std::optional<c10::ScalarType> parseDtype(std::string_view name) {
  const auto token = normalizeDtypeName(name);
  for (const auto& info : dtypeInfos) {
    if (token == normalizeDtypeName(info.canonical_name) ||
        token == normalizeDtypeName(info.native_name) || token == info.torch_name)
      return info.scalar_type;
  }
  if (token == "u8") return c10::ScalarType::Byte;
  if (token == "i8") return c10::ScalarType::Char;
  if (token == "i16") return c10::ScalarType::Short;
  if (token == "i32") return c10::ScalarType::Int;
  if (token == "i64") return c10::ScalarType::Long;
  if (token == "f16") return c10::ScalarType::Half;
  if (token == "bf16") return c10::ScalarType::BFloat16;
  if (token == "f32") return c10::ScalarType::Float;
  if (token == "f64") return c10::ScalarType::Double;
  if (token == "f8_e4m3") return c10::ScalarType::Float8_e4m3fn;
  if (token == "f8_e5m2") return c10::ScalarType::Float8_e5m2;
  if (token == "chalf") return c10::ScalarType::ComplexHalf;
  if (token == "cfloat") return c10::ScalarType::ComplexFloat;
  if (token == "cdouble") return c10::ScalarType::ComplexDouble;
  if (token == "u16") return c10::ScalarType::UInt16;
  if (token == "u32") return c10::ScalarType::UInt32;
  if (token == "u64") return c10::ScalarType::UInt64;
  if (token == "bit") return c10::ScalarType::UInt1;
  return std::nullopt;
}

// Lean Device constructors: CUDA = 0, CPU = 1, MPS = 2. Validate the
// full-width CUDA index before narrowing to LibTorch's signed DeviceIndex.
inline c10::Device deviceFromLeanTag(unsigned tag, uint64_t cuda_index = 0) {
  switch (tag) {
    case 0:
      TORCH_CHECK(
          cuda_index <= static_cast<uint64_t>(std::numeric_limits<c10::DeviceIndex>::max()),
          "CUDA device index ", cuda_index, " exceeds the supported nonnegative range [0, ",
          static_cast<int>(std::numeric_limits<c10::DeviceIndex>::max()), "]");
      return c10::Device(c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(cuda_index));
    case 1:
      return c10::Device(c10::DeviceType::CPU);
    case 2:
#ifdef __APPLE__
      return c10::Device(c10::DeviceType::MPS);
#else
      TORCH_CHECK(false, "MPS devices are unsupported on non-Apple platforms");
#endif
    default:
      TORCH_CHECK(false, "Invalid Lean Device constructor tag: ", tag);
  }
}

// Native backends without a Lean Device constructor must never masquerade
// as another placement when validating an exact device policy.
inline unsigned leanDeviceTag(const c10::Device& device) {
  if (device.is_cuda()) {
    TORCH_CHECK(device.index() >= 0, "Native CUDA device has no concrete index: ", device);
    return 0;
  }
  if (device.is_cpu()) return 1;
  if (device.is_mps()) return 2;
  TORCH_CHECK(false, "Unsupported native tensor device: ", device);
}

} // namespace tyr::metadata
