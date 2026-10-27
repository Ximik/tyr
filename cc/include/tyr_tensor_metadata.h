#pragma once

#include "../src/generated/tyr_ffi.h"
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <array>
#include <optional>
#include <c10/util/Exception.h>
#include <cstdint>
#include <limits>

namespace tyr::metadata {

// Indexed by Lean DType constructor tag (generated `lean::DType`), not native
// enum order.
struct DTypeInfo {
  lean::DType tag;
  c10::ScalarType scalar_type;
};

inline constexpr std::array<DTypeInfo, lean::DTypeCount> dtypeInfos{{
    {lean::DType::UInt8, c10::ScalarType::Byte},
    {lean::DType::Int8, c10::ScalarType::Char},
    {lean::DType::Int16, c10::ScalarType::Short},
    {lean::DType::Int32, c10::ScalarType::Int},
    {lean::DType::Int64, c10::ScalarType::Long},
    {lean::DType::Float16, c10::ScalarType::Half},
    {lean::DType::BFloat16, c10::ScalarType::BFloat16},
    {lean::DType::Float32, c10::ScalarType::Float},
    {lean::DType::Float64, c10::ScalarType::Double},
    {lean::DType::Bool, c10::ScalarType::Bool},
    {lean::DType::Float8E4M3FN, c10::ScalarType::Float8_e4m3fn},
    {lean::DType::Float8E5M2, c10::ScalarType::Float8_e5m2},
    {lean::DType::Complex32, c10::ScalarType::ComplexHalf},
    {lean::DType::Complex64, c10::ScalarType::ComplexFloat},
    {lean::DType::Complex128, c10::ScalarType::ComplexDouble},
    {lean::DType::QInt8, c10::ScalarType::QInt8},
    {lean::DType::QUInt8, c10::ScalarType::QUInt8},
    {lean::DType::QInt32, c10::ScalarType::QInt32},
    {lean::DType::QUInt4x2, c10::ScalarType::QUInt4x2},
    {lean::DType::QUInt2x4, c10::ScalarType::QUInt2x4},
    {lean::DType::Bits1x8, c10::ScalarType::Bits1x8},
    {lean::DType::Bits2x4, c10::ScalarType::Bits2x4},
    {lean::DType::Bits4x2, c10::ScalarType::Bits4x2},
    {lean::DType::Bits8, c10::ScalarType::Bits8},
    {lean::DType::Bits16, c10::ScalarType::Bits16},
    {lean::DType::Float8E5M2FNUZ, c10::ScalarType::Float8_e5m2fnuz},
    {lean::DType::Float8E4M3FNUZ, c10::ScalarType::Float8_e4m3fnuz},
    {lean::DType::UInt16, c10::ScalarType::UInt16},
    {lean::DType::UInt32, c10::ScalarType::UInt32},
    {lean::DType::UInt64, c10::ScalarType::UInt64},
    {lean::DType::UInt1, c10::ScalarType::UInt1},
    {lean::DType::UInt2, c10::ScalarType::UInt2},
    {lean::DType::UInt3, c10::ScalarType::UInt3},
    {lean::DType::UInt4, c10::ScalarType::UInt4},
    {lean::DType::UInt5, c10::ScalarType::UInt5},
    {lean::DType::UInt6, c10::ScalarType::UInt6},
    {lean::DType::UInt7, c10::ScalarType::UInt7},
    {lean::DType::Int1, c10::ScalarType::Int1},
    {lean::DType::Int2, c10::ScalarType::Int2},
    {lean::DType::Int3, c10::ScalarType::Int3},
    {lean::DType::Int4, c10::ScalarType::Int4},
    {lean::DType::Int5, c10::ScalarType::Int5},
    {lean::DType::Int6, c10::ScalarType::Int6},
    {lean::DType::Int7, c10::ScalarType::Int7},
    {lean::DType::Float8E8M0FNU, c10::ScalarType::Float8_e8m0fnu},
    {lean::DType::Float4E2M1FNX2, c10::ScalarType::Float4_e2m1fn_x2},
}};

constexpr bool dtypeInfosFollowLeanTags() {
  for (unsigned i = 0; i < dtypeInfos.size(); ++i) {
    if (static_cast<unsigned>(dtypeInfos[i].tag) != i) return false;
  }
  return true;
}
static_assert(dtypeInfosFollowLeanTags(),
              "dtypeInfos must have one entry per Lean DType constructor, in order");

constexpr bool dtypeInfosHaveDistinctScalarTypes() {
  for (unsigned i = 0; i < dtypeInfos.size(); ++i) {
    for (unsigned j = i + 1; j < dtypeInfos.size(); ++j) {
      if (dtypeInfos[i].scalar_type == dtypeInfos[j].scalar_type) return false;
    }
  }
  return true;
}
// leanDTypeTag returns the first match, so a repeated ScalarType would map
// two Lean dtypes to one.
static_assert(dtypeInfosHaveDistinctScalarTypes(),
              "dtypeInfos must map each Lean DType to a distinct ScalarType");

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

// Validate the full-width CUDA index before narrowing to LibTorch's signed
// DeviceIndex.
inline c10::Device deviceFromLeanTag(unsigned tag, uint64_t cuda_index = 0) {
  switch (static_cast<lean::DeviceKind>(tag)) {
    case lean::DeviceKind::CUDA:
      TORCH_CHECK(
          cuda_index <= static_cast<uint64_t>(std::numeric_limits<c10::DeviceIndex>::max()),
          "CUDA device index ", cuda_index, " exceeds the supported nonnegative range [0, ",
          static_cast<int>(std::numeric_limits<c10::DeviceIndex>::max()), "]");
      return c10::Device(c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(cuda_index));
    case lean::DeviceKind::CPU:
      return c10::Device(c10::DeviceType::CPU);
    case lean::DeviceKind::MPS:
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
    return static_cast<unsigned>(lean::DeviceKind::CUDA);
  }
  if (device.is_cpu()) return static_cast<unsigned>(lean::DeviceKind::CPU);
  if (device.is_mps()) return static_cast<unsigned>(lean::DeviceKind::MPS);
  TORCH_CHECK(false, "Unsupported native tensor device: ", device);
}

} // namespace tyr::metadata
