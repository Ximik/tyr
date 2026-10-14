#pragma once

#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <cstdint>
#include <limits>

namespace tyr::metadata {

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
