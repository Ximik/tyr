#include "tyr_safetensors_abi.h"
#include "tyr_flux_abi.h"
#include "tyr_vae_abi.h"
#include "tyr_gemma4_abi.h"
#include "tyr_branching_abi.h"
#include "tyr_laguna_abi.h"
#include "tyr_kittentts_abi.h"
#include "tyr_qwen35_abi.h"
/*
 * tyr.cpp - Lean 4 ↔ LibTorch FFI Bindings
 *
 * ============================================================================
 * REFERENCE COUNTING CONVENTIONS
 * ============================================================================
 *
 * This file bridges Lean's reference counting with LibTorch tensors.
 * Understanding these conventions is critical for memory safety.
 *
 * ## Parameter Types
 *
 * - `lean_obj_arg` (owned): Callee receives ownership. MUST call `lean_dec()`
 *   after extracting data. Used for temporary parameters like shape arrays.
 *   Example:
 *     lean_object* lean_torch_randn(lean_obj_arg s, ...) {
 *       auto shape = getShape(s);
 *       lean_dec(s);  // REQUIRED: we own s, must release it
 *       ...
 *     }
 *
 * - `b_lean_obj_arg` (borrowed): Callee does NOT own the object. Must NOT
 *   call `lean_dec()`. Use `borrowTensor()` to get a C++ tensor with shared
 *   ownership that auto-decrements when it goes out of scope.
 *   Example:
 *     lean_object* lean_torch_add(b_lean_obj_arg x, b_lean_obj_arg y) {
 *       auto x_ = borrowTensor(x);  // Shared ownership, no manual cleanup
 *       auto y_ = borrowTensor(y);  // Same
 *       return giveTensor(x_ + y_); // Transfer new tensor to Lean
 *     }
 *
 * ## Key Functions
 *
 * - `borrowTensor(b_lean_obj_arg)`: Get C++ tensor with shared ownership.
 *   When the returned torch::Tensor goes out of scope, refcount is
 *   automatically decremented. No manual cleanup needed.
 *
 * - `giveTensor(torch::Tensor)`: Transfer ownership to Lean by storing an
 *   owning intrusive ref (`TensorImpl*`) in an external object.
 *   Increments g_live_lean_tensors counter.
 *
 * ## Memory Lifecycle
 *
 * 1. Tensor created in C++ → giveTensor() → Lean owns one intrusive ref
 * 2. Lean passes tensor to C++ → borrowTensor() incref + reclaim to `Tensor`
 * 3. Lean GC finalizes tensor → decref finalizer releases Lean's owned ref
 *
 * ## Debugging
 *
 * g_live_lean_tensors tracks outstanding tensors. If this grows unboundedly,
 * there's a memory leak. Query via lean_torch_get_live_tensors().
 *
 * ============================================================================
 */

#include <bit>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <chrono>
#include <thread>
#include <lean/lean.h>
#include "tyr_ffi_abi.h"
#include "tyr_raw_tensor_abi.h"
#include "tyr_owned_kv_abi.h"
#include "tyr_neural_abi.h"
#include <torch/torch.h>
#include <ATen/ATen.h>
#include "tyr_attention.h"
#include "tyr_tensor_metadata.h"
#include "tyr_checked_result.h"

#include <ATen/Generator.h>
#include <ATen/Context.h>
#include <c10/core/CPUAllocator.h>
#if defined(__has_include)
#if __has_include(<ATen/cuda/CUDAGeneratorImpl.h>)
#include <ATen/cuda/CUDAGeneratorImpl.h>
#define TYR_HAS_CUDA_GENERATOR 1
#else
#define TYR_HAS_CUDA_GENERATOR 0
#endif
#else
#define TYR_HAS_CUDA_GENERATOR 0
#endif

namespace tyr_ops {
torch::Tensor flash_attn_dispatch(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const c10::optional<torch::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    const c10::optional<double>& scale,
    bool enable_gqa);
}

#if defined(__has_include)
#if __has_include(<soxr.h>)
#define TYR_HAS_SOXR 1
#include <soxr.h>
#else
#define TYR_HAS_SOXR 0
#endif
#else
#define TYR_HAS_SOXR 0
#endif

#if defined(__has_include)
#if __has_include(<c10/cuda/CUDAStream.h>) && __has_include(<c10/cuda/CUDAFunctions.h>) && __has_include(<cuda_runtime_api.h>)
#define TYR_HAS_CUDA_API 1
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAFunctions.h>
#include <cuda_runtime_api.h>
#else
#define TYR_HAS_CUDA_API 0
#endif

#if defined(__has_include)
#if __has_include(<torch/mps.h>)
#define TYR_HAS_TORCH_MPS_API 1
#include <torch/mps.h>
#else
#define TYR_HAS_TORCH_MPS_API 0
#endif
#else
#define TYR_HAS_TORCH_MPS_API 0
#endif
#else
#define TYR_HAS_CUDA_API 0
#endif

#ifdef __APPLE__
extern "C" bool tyr_apple_mps_is_available(int debug_flag);
#endif

// Own transferred Lean arguments before parsing or calling throwable native code.
using OwnedLeanObject = std::unique_ptr<lean_object, decltype(&lean_dec)>;

// Global atomic counter for live tensors handed to Lean
std::atomic<int64_t> g_live_lean_tensors(0);

void trivialForeach(void *p, b_lean_obj_arg a) { return; }

void deleteTorchTensorFinalize(void *p) {
  auto* impl = static_cast<c10::TensorImpl*>(p);
  if (impl != nullptr) {
    c10::raw::intrusive_ptr::decref(impl);
  }
  g_live_lean_tensors--; // Decrement when a tensor is finalized
}

static lean_external_class* getTorchTensorClass() {
    static lean_external_class* c =
        lean_register_external_class(&deleteTorchTensorFinalize, &trivialForeach);
    return c;
}

// Borrow a tensor from Lean with proper reference counting.
// Returns an owning tensor handle reconstructed from an owning intrusive ref.
torch::Tensor borrowTensor(b_lean_obj_arg o) {
    auto* obj = const_cast<lean_object*>(o);
    if (!lean_is_external(obj)) {
      throw std::runtime_error("borrowTensor: expected external object");
    }

#ifndef NDEBUG
    if (lean_get_external_class(obj) != getTorchTensorClass()) {
      throw std::runtime_error("borrowTensor: unexpected external class");
    }
#endif

    auto* impl = static_cast<c10::TensorImpl*>(lean_get_external_data(obj));
    if (impl == nullptr) {
      return torch::Tensor();
    }

    // Reconstruct an owning intrusive_ptr without using unsafe non-owning reclaim.
    c10::raw::intrusive_ptr::incref(impl);
    auto p = c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>::reclaim(impl);
    return torch::Tensor(std::move(p));
}

// Consume a Lean reference only after acquiring an owning native handle. This
// lets the final Lean reference disappear before native uniqueness is tested.
static torch::Tensor takeTensor(lean_obj_arg owned) {
  OwnedLeanObject owner(owned, lean_dec);
  auto tensor = borrowTensor(owned);
  owner.reset();
  return tensor;
}

// Reuse is deliberately limited to ordinary, allocator-owned CPU storage.
// Accelerator streams and external buffers may have readers that neither
// Lean's RC nor TensorImpl's RC can observe, so those updates always clone.
static bool canReuseTensorValue(const torch::Tensor& tensor) {
  if (!tensor.defined() || !tensor.device().is_cpu() ||
      tensor.layout() != torch::kStrided || tensor.is_quantized() ||
      tensor.is_nested() || tensor.is_conj() || tensor.is_neg() ||
      !tensor.is_contiguous() || tensor.storage_offset() != 0) return false;
  const auto& impl = tensor.getIntrusivePtr();
  if (!impl.is_uniquely_owned() || impl->autograd_meta() != nullptr ||
      impl->is_inference() || !impl->unique_version() || !impl->has_storage()) return false;
  const auto& storage = tensor.storage();
  if (!storage.resizable() || storage.allocator() != c10::GetCPUAllocator() ||
      storage.nbytes() != tensor.nbytes()) return false;
  // Observe the combined strong/weak counts atomically, without adding a ref.
  // Separate reads can miss a weak observer that locks and drops its weak ref
  // between them. Storage pins the lifetime of this non-owning handle; release
  // it before destruction so observation never decrements Storage's reference.
  struct StorageOwnershipObserver {
    c10::intrusive_ptr<c10::StorageImpl> pointer;
    ~StorageOwnershipObserver() { pointer.release(); }
  } observer{c10::intrusive_ptr<c10::StorageImpl>(
      storage.unsafeGetStorageImpl(), c10::raw::DontIncreaseRefcount{})};
  return observer.pointer.is_uniquely_owned();
}

static torch::Tensor tensorForValueUpdate(torch::Tensor tensor) {
  if (canReuseTensorValue(tensor)) return tensor;
  return tensor.clone();
}

// Used under NoGradGuard for cache/optimizer value updates. Cloning must not
// silently turn a trainable parameter into a tensor with requires_grad=false.
static torch::Tensor tensorForNoGradValueUpdate(torch::Tensor tensor) {
  if (canReuseTensorValue(tensor)) return tensor;
  auto result = tensor.clone();
  result.requires_grad_(tensor.requires_grad());
  return result;
}

static void checkCopyMetadata(const torch::Tensor& destination, const torch::Tensor& source) {
  TORCH_CHECK(destination.sizes() == source.sizes(), "copy: tensor shape mismatch");
  TORCH_CHECK(destination.scalar_type() == source.scalar_type(), "copy: tensor dtype mismatch");
  // copy_ transfers across devices while retaining destination placement.
}

static int64_t checkSliceCopy(const torch::Tensor& destination, const torch::Tensor& source,
                              uint64_t dim, uint64_t start) {
  TORCH_CHECK(dim < static_cast<uint64_t>(destination.dim()), "slice copy dimension out of range");
  TORCH_CHECK(source.dim() == destination.dim(), "slice copy rank mismatch");
  const auto axis = static_cast<int64_t>(dim);
  TORCH_CHECK(start <= static_cast<uint64_t>(destination.size(axis)) &&
    static_cast<uint64_t>(source.size(axis)) <= static_cast<uint64_t>(destination.size(axis)) - start,
    "slice copy exceeds capacity");
  for (int64_t i = 0; i < destination.dim(); ++i) {
    TORCH_CHECK(i == axis || destination.size(i) == source.size(i), "slice copy shape mismatch");
  }
  TORCH_CHECK(destination.device() == source.device() && destination.scalar_type() == source.scalar_type(),
    "slice copy device/dtype mismatch");
  return axis;
}

// Transfer ownership of a new tensor to Lean.
// The tensor object is moved to a heap allocation owned by Lean.
lean_object *giveTensor(torch::Tensor t) {
  auto* impl = t.unsafeGetTensorImpl();
  if (impl != nullptr) {
    c10::raw::intrusive_ptr::incref(impl);
  }
  g_live_lean_tensors++; // Increment when a new tensor is given to Lean
  return lean_alloc_external(getTorchTensorClass(), impl);
}

// Alias for backward compatibility
lean_object *fromTorchTensor(torch::Tensor t) {
  return giveTensor(t);
}

std::vector<int64_t> getShape(b_lean_obj_arg s) {
  std::vector<int64_t> shape;  
  for (size_t i = 0; i<lean_array_size(s); i++) {
    shape.push_back(lean_unbox_uint64(lean_array_get_core(s, i)));
  }
  return shape;
}

torch::Device getDevice(b_lean_obj_arg device) {
  const auto tag = lean_obj_tag(device);
  TORCH_CHECK(tag != 0 || !lean_is_scalar(device), "Malformed Lean CUDA device constructor");
  const auto cuda_index = tag == 0 ? lean_ctor_get_uint64(device, 0) : 0;
  const auto result = tyr::metadata::deviceFromLeanTag(tag, cuda_index);
#ifdef __APPLE__
  if (result.is_mps()) {
    static std::once_flag g_mps_runtime_warmup_once;
    std::call_once(g_mps_runtime_warmup_once, []() {
      const int debug_flag = std::getenv("TYR_DEBUG_MPS") != nullptr ? 1 : 0;
      (void)tyr_apple_mps_is_available(debug_flag);
    });
  }
#endif
  return result;
}

static lean_object* mkIoUserError(const std::string& msg) {
  return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(msg.c_str())));
}

static lean_object* mkC10IoError(const char* context, const c10::Error& e) {
  return mkIoUserError(std::string(context) + ": " + std::string(e.what()));
}

static lean_object* mkStdIoError(const char* context, const std::exception& e) {
  return mkIoUserError(std::string(context) + ": " + std::string(e.what()));
}

// ----------------------------------------------------------------------------
// Fail intelligibly on uncaught C++ exceptions.
//
// Most tensor ops are pure on the Lean side (`T s → T s`), so there is no IO
// channel to report a libtorch error through, and we deliberately do not
// guard each entry point. Raw T shape annotations are not enforced by Lean;
// the Tensor facade checks selected operations and raw-entry contracts.
// An invalid raw pure call still terminates the process. This handler makes
// that termination readable: print the pending exception's message
// — for `c10::Error` just the user-facing text, not the multi-page dispatch
// table and C++ backtrace the default handler dumps — and then abort.
// Set TYR_VERBOSE_ERRORS=1 to get the full libtorch report instead.
//
// Caveat: on toolchains where libtorch's exception cannot unwind into our
// runtime (see the libc++abi/libstdc++ note in `lean_torch_manual_seed`
// below), the foreign runtime may invoke its own terminate handler instead
// of this one.
namespace {

std::terminate_handler g_prev_terminate_handler = nullptr;

[[noreturn]] void tyrTerminateHandler() {
  if (std::exception_ptr eptr = std::current_exception()) {
    try {
      std::rethrow_exception(eptr);
    } catch (const c10::Error& e) {
      const bool verbose = std::getenv("TYR_VERBOSE_ERRORS") != nullptr;
      std::fprintf(stderr,
                   "\n[tyr] fatal: uncaught libtorch error at the Lean FFI boundary:\n  %s\n",
                   verbose ? e.what() : e.what_without_backtrace());
      if (!verbose) {
        std::fprintf(stderr,
                     "[tyr] set TYR_VERBOSE_ERRORS=1 for the full libtorch report.\n");
      }
    } catch (const std::exception& e) {
      std::fprintf(stderr,
                   "\n[tyr] fatal: uncaught C++ exception at the Lean FFI boundary:\n  %s\n",
                   e.what());
    } catch (...) {
      std::fprintf(stderr,
                   "\n[tyr] fatal: uncaught non-standard C++ exception at the Lean FFI boundary\n");
    }
    std::fflush(stderr);
    std::abort();
  }
  // terminate() without an active exception: defer to the previous handler.
  if (g_prev_terminate_handler != nullptr) {
    g_prev_terminate_handler();
  }
  std::abort();
}

const bool g_tyr_terminate_handler_installed = [] {
  g_prev_terminate_handler = std::set_terminate(&tyrTerminateHandler);
  return true;
}();

}  // namespace

extern "C" {

lean_object* lean_torch_get_live_tensors(lean_object* /* w */) {
    return lean_io_result_mk_ok(lean_box_uint64(static_cast<uint64_t>(g_live_lean_tensors.load())));
}

// Background on the workarounds below:
//
// The Lean toolchain statically embeds libc++abi + llvm-libunwind into
// every executable. libtorch is built against libstdc++. The two C++
// runtimes tag exceptions with different runtime classes
// ("CLNGC++" vs "GNUCC++") that the personality functions in libtorch's
// frames don't reconcile, so any `c10::Error` thrown by libtorch escapes
// all `catch` blocks in this translation unit and trips
// `std::terminate()` → SIGABRT.
//
// We can't fix the toolchain mismatch from within Tyr, but we can avoid
// the throwing paths in the two FFI sites that hit it:
//   - `torch::manual_seed` iterates every CUDA generator and calls
//     `set_current_seed` → `assertNotCapturing` → `c10_cuda_check` →
//     throw. Replaced with a CPU-only seed via the default CPU
//     generator (caller cares about fixture determinism, which the CPU
//     seed alone covers).
//   - `c10::cuda::device_synchronize()` wraps `cudaDeviceSynchronize`
//     and translates errors via `c10_cuda_check` → throw. Replaced
//     with a direct `cudaDeviceSynchronize` call that returns
//     `cudaError_t` and never throws.

lean_object* lean_torch_manual_seed(uint64_t seed, lean_object* /*w*/) {
  try {
    auto cpu_gen = at::detail::getDefaultCPUGenerator();
    {
      std::lock_guard<std::mutex> lock(cpu_gen.mutex());
      cpu_gen.set_current_seed(seed);
    }
    // Deliberately skip CUDA generator seeding: the libtorch path
    // throws across the libc++abi/libstdc++ ABI boundary on this
    // toolchain. Fixture-regeneration callers (`Examples/GPU/Parity.lean`
    // `seedFixtures`) only need the CPU seed.
    return lean_io_result_mk_ok(lean_box(0));
  } catch (const std::exception& e) {
    return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string((std::string("Failed to set manual seed: ") + e.what()).c_str())));
  } catch (...) {
    return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string("Failed to set manual seed: unknown exception")));
  }
}

// ============================================================================
// Tensor metadata extraction for visualization widgets
// ============================================================================

// Get runtime shape as Lean Array UInt64
lean_object* lean_torch_get_shape(lean_obj_arg s, b_lean_obj_arg t) {
  lean_dec(s);
    auto tensor = borrowTensor(t);
    auto sizes = tensor.sizes();

    lean_object* arr = lean_alloc_array(sizes.size(), sizes.size());
    for (size_t i = 0; i < sizes.size(); i++) {
        lean_array_set_core(arr, i, lean_box_uint64(static_cast<uint64_t>(sizes[i])));
    }
    return arr;
}

// Get dtype as string
lean_object* lean_torch_get_dtype(lean_obj_arg s, b_lean_obj_arg t) {
  lean_dec(s);
    auto tensor = borrowTensor(t);
    return lean_mk_string(tyr::metadata::dtypeName(tensor.scalar_type()));
}

// Get device as string
lean_object* lean_torch_get_device(lean_obj_arg s, b_lean_obj_arg t) {
  lean_dec(s);
    auto tensor = borrowTensor(t);
    auto device = tensor.device();

    std::ostringstream stream;
    stream << device;
    return lean_mk_string(stream.str().c_str());
}

// Get device as Device enum (CUDA=0, CPU=1, MPS=2)
lean_object* lean_torch_get_device_enum(lean_obj_arg s, b_lean_obj_arg t) {
  lean_dec(s);
    auto tensor = borrowTensor(t);
    auto device = tensor.device();

    const auto tag = tyr::metadata::leanDeviceTag(device);
    if (tag == 0) {
        // CUDA tag=0, constructor with 1 field (device index)
        lean_object* obj = lean_alloc_ctor(0, 0, 8);
        lean_ctor_set_uint64(obj, 0, device.index());
        return obj;
    } else if (tag == 1) {
        // CPU tag=1, scalar constructor
        return lean_box(1);
    } else { // leanDeviceTag rejects every unsupported backend.
        // MPS tag=2, scalar constructor
        return lean_box(2);
    }
}

// Get tensor values as flat Array Float (up to max_elements)
lean_object* lean_torch_get_values(lean_obj_arg s, b_lean_obj_arg t, uint64_t max_elements) {
  lean_dec(s);
    auto tensor = borrowTensor(t);

    // Move to CPU if needed and convert to float
    auto cpu_tensor = tensor.cpu().to(torch::kFloat32).contiguous();
    auto flat = cpu_tensor.flatten();

    size_t num_elements = std::min(static_cast<size_t>(flat.numel()),
                                   static_cast<size_t>(max_elements));

    // Use FloatArray for efficiency
    lean_object* arr = lean_alloc_sarray(sizeof(double), num_elements, num_elements);
    double* data = reinterpret_cast<double*>(lean_sarray_cptr(arr));

    auto accessor = flat.accessor<float, 1>();
    for (size_t i = 0; i < num_elements; i++) {
        data[i] = static_cast<double>(accessor[i]);
    }

    return arr;
}

// Get tensor statistics as JSON string
lean_object* lean_torch_get_stats(lean_obj_arg s, b_lean_obj_arg t) {
  lean_dec(s);
    auto tensor = borrowTensor(t);
    auto cpu_tensor = tensor.cpu().to(torch::kFloat32);

    double min_val = cpu_tensor.min().item<double>();
    double max_val = cpu_tensor.max().item<double>();
    double mean_val = cpu_tensor.mean().item<double>();
    double std_val = cpu_tensor.std().item<double>();

    std::ostringstream stream;
    stream << "{\"min\":" << min_val
           << ",\"max\":" << max_val
           << ",\"mean\":" << mean_val
           << ",\"std\":" << std_val << "}";

    return lean_mk_string(stream.str().c_str());
}


// --
// tensor creation api
lean_object* lean_torch_randn(lean_obj_arg s, uint8_t requires_grad, lean_obj_arg device) {
  return tyr::checkedIoCall("randn", [&] {
  OwnedLeanObject s_owner(s, lean_dec);
  OwnedLeanObject device_owner(device, lean_dec);
  try {
    auto device_ = getDevice(device);
    device_owner.reset();
    // CPU-then-transfer to avoid the GB10 CUDA-RNG throw path; see comment
    // on `lean_torch_rand`.
    // Enable gradients only after transfer so the result is a leaf on every device.
    auto cpu_t = torch::randn(getShape(s), torch::TensorOptions().requires_grad(false));
    auto t = device_.is_cpu() ? cpu_t : cpu_t.to(device_);
    t.requires_grad_(requires_grad != 0);
    s_owner.reset();
    return lean_io_result_mk_ok(fromTorchTensor(t));
  } catch (const c10::Error& e) {
    return mkC10IoError("randn", e);
  } catch (const std::exception& e) {
    return mkIoUserError(std::string("randn: ") + e.what());
  }
  });
}
// Generate on CPU then transfer to the target device. The libtorch CUDA
// `Philox` generator hits `c10_cuda_check_implementation` on GB10/sm_120
// which throws `c10::Error` across the libstdc++ → libc++abi boundary
// (see `lean_torch_manual_seed` block comment). Generating on CPU and
// `.to`-transferring avoids the throw entirely and is numerically
// equivalent for fixture-regeneration use cases.
lean_object* lean_torch_rand(lean_obj_arg s, int requires_grad, lean_obj_arg device) {
  OwnedLeanObject s_owner(s, lean_dec);
  OwnedLeanObject device_owner(device, lean_dec);
  try {
    auto device_ = getDevice(device);
    device_owner.reset();
    auto cpu_t = torch::rand(getShape(s), torch::TensorOptions().requires_grad(false));
    auto t = device_.is_cpu() ? cpu_t : cpu_t.to(device_);
    t.requires_grad_(requires_grad != 0);
    s_owner.reset();
    return lean_io_result_mk_ok(fromTorchTensor(t));
  } catch (const c10::Error& e) {
    return mkC10IoError("rand", e);
  } catch (const std::exception& e) {
    return mkIoUserError(std::string("rand: ") + e.what());
  }
}

// Initialize on CPU, then transfer, matching the GB10-safe random factory path.
// Unlike a Float32 sample followed by a cast, this samples in the reference dtype.
lean_object* lean_torch_uniform_like(lean_obj_arg shape, b_lean_obj_arg reference,
                                     double lo, double hi) {
  return tyr::checkedIoCall("uniformLike", [&] {
  OwnedLeanObject shape_owner(shape, lean_dec);
  auto dims = getShape(shape);
  shape_owner.reset();
  if (!std::isfinite(lo) || !std::isfinite(hi) || lo > hi) {
    return mkIoUserError("uniformLike requires finite bounds with lo <= hi");
  }
  try {
    auto ref = borrowTensor(reference);
    const auto dtype = ref.scalar_type();
    if (dtype != torch::kFloat16 && dtype != torch::kBFloat16 &&
        dtype != torch::kFloat32 && dtype != torch::kFloat64) {
      return mkIoUserError("uniformLike supports Float16, BFloat16, Float32 and Float64");
    }
    auto sample = torch::empty(dims, torch::TensorOptions().dtype(dtype)).uniform_(lo, hi);
    if (!ref.device().is_cpu()) sample = sample.to(ref.device());
    return lean_io_result_mk_ok(fromTorchTensor(sample));
  } catch (const c10::Error& e) {
    return mkC10IoError("uniformLike", e);
  } catch (const std::exception& e) {
    return mkIoUserError(std::string("uniformLike: ") + e.what());
  }
  });
}

lean_object* lean_torch_randint(uint64_t lowBits, uint64_t highBits, lean_obj_arg s, uint8_t /*requires_grad*/, lean_obj_arg device) {
  return tyr::checkedIoCall("randint", [&] {
  OwnedLeanObject s_owner(s, lean_dec);
  OwnedLeanObject device_owner(device, lean_dec);
  try {
    const auto low = std::bit_cast<int64_t>(lowBits);
    const auto high = std::bit_cast<int64_t>(highBits);
    auto device_ = getDevice(device);
    device_owner.reset();
    // randint always returns Long (int64) dtype - integral types don't support requires_grad.
    // CPU-then-transfer for the same reason as `lean_torch_rand`.
    auto cpu_t = torch::randint(low, high, getShape(s), torch::TensorOptions().dtype(torch::kLong));
    auto t = device_.is_cpu() ? cpu_t : cpu_t.to(device_);
    s_owner.reset();
    return lean_io_result_mk_ok(fromTorchTensor(t));
  } catch (const c10::Error& e) {
    return mkC10IoError("randint", e);
  } catch (const std::exception& e) {
    return mkIoUserError(std::string("randint: ") + e.what());
  }
  });
}

lean_object* lean_torch_full(lean_obj_arg s, double value, uint8_t requires_grad, lean_obj_arg device) {
  OwnedLeanObject s_owner(s, lean_dec);
  OwnedLeanObject device_owner(device, lean_dec);
  auto device_ = getDevice(device);
  device_owner.reset();
  auto t = torch::full(getShape(s), value, torch::TensorOptions().requires_grad(requires_grad).device(device_));
  s_owner.reset();
  return fromTorchTensor(t);
}

lean_object* lean_torch_zeros(lean_obj_arg s, uint8_t requires_grad, lean_obj_arg device) {
  OwnedLeanObject s_owner(s, lean_dec);
  OwnedLeanObject device_owner(device, lean_dec);
  auto device_ = getDevice(device);
  device_owner.reset();
  auto t = torch::zeros(getShape(s), torch::TensorOptions().requires_grad(requires_grad).device(device_));
  s_owner.reset();
  return fromTorchTensor(t);
}

lean_object* lean_torch_ones(lean_obj_arg s, uint8_t requires_grad, lean_obj_arg device) {
  OwnedLeanObject s_owner(s, lean_dec);
  OwnedLeanObject device_owner(device, lean_dec);
  auto device_ = getDevice(device);
  device_owner.reset();
  auto t = torch::ones(getShape(s), torch::TensorOptions().requires_grad(requires_grad).device(device_));
  s_owner.reset();
  return fromTorchTensor(t);
}

lean_object* lean_torch_zeros_checked(lean_obj_arg shape, lean_obj_arg device) {
  return tyr::checkedCall("zeros", [&] {
    return lean_torch_zeros(shape, 0, device);
  });
}

lean_object* lean_torch_ones_checked(lean_obj_arg shape, lean_obj_arg device) {
  return tyr::checkedCall("ones", [&] {
    return lean_torch_ones(shape, 0, device);
  });
}

lean_object* lean_torch_eye_checked(uint64_t n) {
  return tyr::checkedCall("eye", [&] {
    return lean_torch_eye(n, 0);
  });
}

lean_object* lean_torch_zeros_like(lean_obj_arg s, b_lean_obj_arg self) {
  lean_dec(s);
  auto self_ = borrowTensor(self);
  auto t = torch::zeros_like(self_);
  return fromTorchTensor(t);
}

lean_object* lean_torch_ones_like(lean_obj_arg s, b_lean_obj_arg self) {
  lean_dec(s);
  auto self_ = borrowTensor(self);
  auto t = torch::ones_like(self_);
  return fromTorchTensor(t);
}

lean_object* lean_torch_copy_(lean_obj_arg s, lean_obj_arg dst,
                              b_lean_obj_arg src) {
  return tyr::checkedIoCall("copyInto", [&] {
  lean_dec(s);
  OwnedLeanObject destination_owner(dst, lean_dec);
  try {
    // Borrow first: src may be the very same Lean object as consumed dst.
    auto source = borrowTensor(src);
    auto destination = takeTensor(destination_owner.release());
    checkCopyMetadata(destination, source);
    auto result = tensorForValueUpdate(std::move(destination));
    result.copy_(source);
    return lean_io_result_mk_ok(fromTorchTensor(std::move(result)));
  } catch (const c10::Error& e) {
    return mkC10IoError("copy_ failed", e);
  } catch (const std::exception& e) {
    return mkStdIoError("copy_ failed", e);
  }
  });
}

lean_object* lean_torch_arange(uint64_t start, uint64_t stop, uint64_t step) {
  constexpr auto max_index = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  TORCH_CHECK(start <= max_index && stop <= max_index && step <= max_index,
              "arange: arguments must fit in signed 64-bit tensor indices");
  auto t = torch::arange(static_cast<int64_t>(start), static_cast<int64_t>(stop),
                         static_cast<int64_t>(step), torch::kLong);
  return fromTorchTensor(t);
}

lean_object* lean_torch_eye(uint64_t n, uint8_t requires_grad) {
  TORCH_CHECK(n <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
              "eye: dimension must fit in signed 64 bits");
  auto t = torch::eye(n, torch::TensorOptions().requires_grad(requires_grad));
  return fromTorchTensor(t);
}

lean_object* lean_torch_linspace(double start, double stop, uint64_t steps, uint8_t requires_grad) {
  TORCH_CHECK(steps <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
              "linspace: steps must fit in signed 64 bits");
  auto t = torch::linspace(start, stop, steps, torch::TensorOptions().requires_grad(requires_grad));
  return fromTorchTensor(t);
}

lean_object* lean_torch_logspace(double start, double stop, uint64_t steps, double base, uint8_t requires_grad) {
  TORCH_CHECK(steps <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
              "logspace: steps must fit in signed 64 bits");
  auto t = torch::logspace(start, stop, steps, base, torch::TensorOptions().requires_grad(requires_grad));
  return fromTorchTensor(t);
}


uint8_t lean_torch_requires_grad(lean_obj_arg shape, b_lean_obj_arg x) {
  lean_dec(shape);
    auto x_ = borrowTensor(x);
    auto res_ = x_.requires_grad();
    return res_;
}

//

lean_object* lean_torch_reshape(lean_obj_arg s, b_lean_obj_arg self, b_lean_obj_arg shape) {
  lean_dec(s);
  auto shape_ = getShape(shape);
  // In Lean code, reshape #[ ] is used as a type-erasure cast to dynamic shape.
  // Treat empty shape as a no-op instead of attempting scalar reshape.
  if (shape_.empty()) {
    // Return the original Lean object directly to avoid creating alias wrappers
    // from a borrowed tensor handle.
    lean_inc(self);
    return const_cast<lean_object*>(self);
  }
  auto self_ = borrowTensor(self);
  auto res = torch::reshape(self_, shape_);
  return fromTorchTensor(res);
}

lean_object* lean_torch_reshape_exact(lean_obj_arg s, b_lean_obj_arg self,
                                      b_lean_obj_arg shape) {
  lean_dec(s);
  return fromTorchTensor(borrowTensor(self).reshape(getShape(shape)));
}

lean_object* lean_torch_permute(lean_obj_arg s, b_lean_obj_arg self, lean_obj_arg permutation) {
  lean_dec(s);
  OwnedLeanObject permutation_owner(permutation, lean_dec);
  auto permutation_ = getShape(permutation);
  auto self_ = borrowTensor(self);
  auto res = torch::permute(self_, permutation_);
  return fromTorchTensor(res);
}

lean_object* lean_torch_get(lean_obj_arg s, b_lean_obj_arg self, lean_obj_arg idx) {
  OwnedLeanObject idx_owner(idx, lean_dec);
  lean_dec(s);
  const auto index = static_cast<int64_t>(lean_int64_of_int(idx));
  auto* roundtrip = lean_int64_to_int(index);
  const bool fits = lean_int_dec_eq(idx, roundtrip);
  lean_dec(roundtrip);
  idx_owner.reset();
  TORCH_CHECK(fits, "getOp: index must fit in signed 64 bits");
  auto self_ = borrowTensor(self);
  auto res = self_.index({index});
  return fromTorchTensor(res);
}

lean_object* lean_torch_to(lean_obj_arg s, b_lean_obj_arg self, lean_obj_arg device) {
  lean_dec(s);
  OwnedLeanObject device_owner(device, lean_dec);
  auto self_ = borrowTensor(self);
  auto device_ = getDevice(device);
  device_owner.reset();
  auto res = self_.to(device_);
  return fromTorchTensor(res);
}

lean_object* lean_torch_slice(lean_obj_arg /*s*/,
  b_lean_obj_arg self,
  b_lean_obj_arg dim,
  int64_t start,
  int64_t stop,
  int64_t step) {

  auto dim_ = lean_uint64_of_nat(dim);
  auto self_ = borrowTensor(self);
  auto res = self_.slice(dim_, start, stop, step);
  return fromTorchTensor(res);
}

#define BINOP_FUN(F) \
lean_object* lean_torch_tensor_##F(lean_obj_arg s, b_lean_obj_arg a, b_lean_obj_arg b) { \
  lean_dec(s); \
  auto a_ = borrowTensor(a); \
  auto b_ = borrowTensor(b); \
  auto c_ = torch::F(a_, b_); \
  return fromTorchTensor(c_); \
}

BINOP_FUN(add)
BINOP_FUN(sub)
BINOP_FUN(mul)
extern "C" LEAN_EXPORT lean_object* lean_torch_tensor_mul_diff(lean_object* /*s1*/, lean_object* s2, lean_object* a, lean_object* b) {
    return lean_torch_tensor_mul(s2, a, b);
}
#undef BINOP_FUN

#define UNOP_FUN(F) \
lean_object* lean_torch_tensor_##F(lean_obj_arg s, b_lean_obj_arg a) { \
  lean_dec(s); \
  auto a_ = borrowTensor(a); \
  auto c_ = torch::F(a_); \
  return fromTorchTensor(c_); \
}

UNOP_FUN(celu)
UNOP_FUN(elu)
UNOP_FUN(gelu)
UNOP_FUN(hardtanh)
UNOP_FUN(leaky_relu)
UNOP_FUN(relu)
UNOP_FUN(relu6)
UNOP_FUN(rrelu)
UNOP_FUN(selu)
UNOP_FUN(silu)
UNOP_FUN(sigmoid)
UNOP_FUN(sin)
UNOP_FUN(cos)
UNOP_FUN(atan)
UNOP_FUN(floor)
UNOP_FUN(tanh)
UNOP_FUN(exp)
UNOP_FUN(log)
UNOP_FUN(log10)
UNOP_FUN(softplus)
#undef UNOP_FUN

lean_object* lean_torch_atan2(lean_obj_arg s, b_lean_obj_arg y, b_lean_obj_arg x) {
  lean_dec(s);
  auto y_ = borrowTensor(y);
  auto x_ = borrowTensor(x);
  auto out_ = torch::atan2(y_, x_);
  return fromTorchTensor(out_);
}

lean_object* lean_torch_tensor_grad(lean_obj_arg s, lean_obj_arg s_out, b_lean_obj_arg output, b_lean_obj_arg input, b_lean_obj_arg grad_output) {
  lean_dec(s);
  lean_dec(s_out);
  auto output_ = borrowTensor(output);
  auto input_ = borrowTensor(input);
  auto grad_output_ = borrowTensor(grad_output);

  torch::autograd::variable_list out_v({output_});
  torch::autograd::variable_list in_v({input_});
  torch::autograd::variable_list grad_out_v({grad_output_});    
  // Keep the graph for callers that request multiple grads from the same output
  // (e.g. structured VJP over many leaves).
  auto grad_in_v = torch::autograd::grad(
      out_v, in_v, grad_out_v,
      /*retain_graph=*/true,
      /*create_graph=*/false,
      /*allow_unused=*/true);

  auto grad_in = grad_in_v[0];
  if (!grad_in.defined()) {
    // Lean callers expect a concrete tensor result even when the input is unused.
    grad_in = torch::zeros_like(input_);
  }
  return fromTorchTensor(grad_in);
}

lean_object* lean_torch_backward(lean_obj_arg /* shape */, b_lean_obj_arg output, b_lean_obj_arg grad_output) {
  auto out_ = borrowTensor(output);
  auto grad_output_ = borrowTensor(grad_output);
  out_.backward(grad_output_);

  lean_inc(output);
  return output;
}

uint8_t lean_torch_allclose(b_lean_obj_arg a, b_lean_obj_arg b, double rtol, double atol) {
  auto a_ = borrowTensor(a);
  auto b_ = borrowTensor(b);
  return torch::allclose(a_, b_, rtol, atol);
}

lean_object* lean_torch_grad_of(lean_obj_arg shape, b_lean_obj_arg x) {
  return tyr::checkedIoCall("gradOf", [&] {
  lean_dec(shape);
  try {
    torch::NoGradGuard guard;
    auto out_ = borrowTensor(x);
    auto grad_out_ = out_.grad();
    // Gradient state is mutable. Return an independent, detached snapshot.
    auto snapshot = grad_out_.defined() ? grad_out_.detach().clone() : torch::zeros_like(out_);
    return lean_io_result_mk_ok(fromTorchTensor(std::move(snapshot)));
  } catch (const c10::Error& e) {
    return mkC10IoError("gradOf", e);
  } catch (const std::exception& e) {
    return mkStdIoError("gradOf", e);
  }
  });
}

// Zero gradients
lean_object* lean_torch_zero_grad(lean_obj_arg shape, b_lean_obj_arg x) {
  return tyr::checkedIoCall("zeroGrad", [&] {
  lean_dec(shape);
  try {
    torch::NoGradGuard guard;
    auto x_ = borrowTensor(x);
    if (x_.grad().defined()) {
        x_.grad().zero_();
    }
    return lean_io_result_mk_ok(lean_box(0));
  } catch (const c10::Error& e) {
    return mkC10IoError("zeroGrad", e);
  } catch (const std::exception& e) {
    return mkStdIoError("zeroGrad", e);
  }
  });
}

// Generated Lean frames release owned references only on normal return. Native
// exceptions from pure callback operations cannot safely become a recoverable
// functionalGrad error after unwinding those frames.
static lean_object* applyPureAutogradCallback(lean_obj_arg callback,
                                             lean_obj_arg arguments) noexcept {
  try {
    return lean_apply_1(callback, arguments);
  } catch (const c10::Error& error) {
    std::fprintf(stderr, "functionalGrad: unrecoverable native exception in pure Lean callback; "
                         "Lean reference cleanup cannot unwind: %s\n", error.what_without_backtrace());
  } catch (const std::exception& error) {
    std::fprintf(stderr, "functionalGrad: unrecoverable native exception in pure Lean callback; "
                         "Lean reference cleanup cannot unwind: %s\n", error.what());
  } catch (...) {
    std::fprintf(stderr, "functionalGrad: unrecoverable unknown exception in pure Lean callback; "
                         "Lean reference cleanup cannot unwind\n");
  }
  std::fflush(stderr);
  std::abort();
}

// A pure differentiation boundary. Temporary leaf identities cannot escape:
// the callback returns only a scalar loss, and results are detached gradients.
lean_object* lean_torch_functional_grad(b_lean_obj_arg inputs, lean_obj_arg f) {
  return tyr::checkedCall("functionalGrad", [&] {
    OwnedLeanObject callback(f, lean_dec);
    torch::AutoGradMode enable_grad(true);
    std::vector<torch::Tensor> leaves;
    const size_t size = lean_array_size(inputs);
    leaves.reserve(size);
    OwnedLeanObject arguments(lean_mk_empty_array(), lean_dec);
    for (size_t i = 0; i < size; ++i) {
      auto leaf = borrowTensor(lean_array_get_core(inputs, i)).detach().clone();
      leaf.requires_grad_(true);
      leaves.push_back(leaf);
      arguments.reset(lean_array_push(arguments.release(), fromTorchTensor(leaf)));
    }
    OwnedLeanObject loss_object(applyPureAutogradCallback(callback.release(), arguments.release()), lean_dec);
    auto loss = borrowTensor(loss_object.get());
    TORCH_CHECK(loss.numel() == 1, "functionalGrad: expected scalar output");
    std::vector<torch::Tensor> gradients(size);
    if (size != 0 && loss.requires_grad()) {
      gradients = torch::autograd::grad({loss}, leaves, {}, /*retain_graph=*/true,
                                        /*create_graph=*/false, /*allow_unused=*/true);
    }
    OwnedLeanObject result(lean_mk_empty_array(), lean_dec);
    for (size_t i = 0; i < size; ++i) {
      auto gradient = gradients[i].defined() ? gradients[i].detach().clone()
                                            : torch::zeros_like(leaves[i]);
      result.reset(lean_array_push(result.release(), fromTorchTensor(std::move(gradient))));
    }
    return result.release();
  });
}

// Parameter identity is effectful. Consume the Lean reference before the native
// uniqueness check, and carry all failures through IO.
lean_object* lean_torch_set_requires_grad(lean_obj_arg shape, lean_obj_arg x, uint8_t requires_grad) {
  return tyr::checkedIoCall("setRequiresGrad", [&] {
  lean_dec(shape);
  auto tensor = takeTensor(x);
  try {
    const bool enabled = requires_grad != 0;
    if (tensor.requires_grad() == enabled)
      return lean_io_result_mk_ok(fromTorchTensor(std::move(tensor)));
    TORCH_CHECK(tensor.is_leaf(), "you can only change requires_grad flags of leaf variables");
    if (!canReuseTensorValue(tensor)) {
      torch::NoGradGuard guard;
      auto result = tensor.clone();
      if (tensor.grad().defined()) result.mutable_grad() = tensor.grad().detach().clone();
      tensor = std::move(result);
    }
    tensor.requires_grad_(enabled);
    return lean_io_result_mk_ok(fromTorchTensor(std::move(tensor)));
  } catch (const c10::Error& e) {
    return mkC10IoError("setRequiresGrad", e);
  } catch (const std::exception& e) {
    return mkStdIoError("setRequiresGrad", e);
  }
  });
}

// Detach tensor from computation graph
lean_object* lean_torch_detach(lean_obj_arg shape, b_lean_obj_arg x) {
  lean_dec(shape);
    auto x_ = borrowTensor(x);
    auto result_ = x_.detach();
    return fromTorchTensor(result_);
}

// Clone tensor with gradient
lean_object* lean_torch_clone(lean_obj_arg /* shape */, b_lean_obj_arg x) {
    auto x_ = borrowTensor(x);
    auto result_ = x_.clone();
    return fromTorchTensor(result_);
}

// Retain grad (for non-leaf tensors)
lean_object* lean_torch_retain_grad(lean_obj_arg /* shape */, b_lean_obj_arg x) {
    auto x_ = borrowTensor(x);
    x_.retain_grad();
    lean_inc(x);
    return x;
}

// Check if tensor is leaf (for autograd)
uint8_t lean_torch_is_leaf(b_lean_obj_arg x) {
    auto x_ = borrowTensor(x);
    auto result = x_.is_leaf();
    return result;
}

// Get gradient function (for debugging)
uint8_t lean_torch_has_grad_fn(lean_obj_arg shape, b_lean_obj_arg x) {
  lean_dec(shape);
    auto x_ = borrowTensor(x);
    auto result = x_.grad_fn() != nullptr;
    return result;
}

// Accumulate gradients
lean_object* lean_torch_accumulate_grad(lean_obj_arg /* shape */, b_lean_obj_arg x, b_lean_obj_arg grad) {
    auto x_ = borrowTensor(x);
    auto grad_ = borrowTensor(grad);
    
    if (x_.grad().defined()) {
        x_.mutable_grad() += grad_;
    } else {
        x_.mutable_grad() = grad_.clone();
    }
    
    lean_inc(x);
    return x;
}

// Manual gradient setting
lean_object* lean_torch_set_grad(lean_obj_arg /* shape */, b_lean_obj_arg x, b_lean_obj_arg grad) {
    auto x_ = borrowTensor(x);
    auto grad_ = borrowTensor(grad);
    
    x_.mutable_grad() = grad_.clone();
    
    lean_inc(x);
    return x;
}

// Enable/disable gradient computation globally
lean_object* lean_torch_set_grad_enabled(uint8_t enabled) {
    torch::autograd::GradMode::set_enabled(enabled);
    return lean_io_result_mk_ok(lean_box(0));
}

lean_object* lean_torch_is_grad_enabled() {
    return lean_io_result_mk_ok(lean_box(torch::autograd::GradMode::is_enabled()));
}

// No-grad context functions
lean_object* lean_torch_with_no_grad(lean_obj_arg /* shape */, lean_obj_arg func) {
    torch::NoGradGuard no_grad_guard;
    // Note: This would need special handling in Lean to call the function
    // For now, this is a placeholder for the infrastructure
    return func;
}

// Higher-order gradients
lean_object* lean_torch_grad_grad(lean_obj_arg /* sx */, lean_obj_arg /* sy */, lean_obj_arg /* sz */,
                                  b_lean_obj_arg z, b_lean_obj_arg y, b_lean_obj_arg x,
                                  b_lean_obj_arg grad_z, b_lean_obj_arg grad_y) {
    auto z_ = borrowTensor(z);
    auto y_ = borrowTensor(y);
    auto x_ = borrowTensor(x);
    auto grad_z_ = borrowTensor(grad_z);
    auto grad_y_ = borrowTensor(grad_y);

    // First compute dz/dy
    torch::autograd::variable_list out_v1({z_});
    torch::autograd::variable_list in_v1({y_});
    torch::autograd::variable_list grad_out_v1({grad_z_});
    auto grad_y_computed = torch::autograd::grad(out_v1, in_v1, grad_out_v1, true, true);

    // Then compute d(dz/dy)/dx
    torch::autograd::variable_list out_v2({grad_y_computed[0]});
    torch::autograd::variable_list in_v2({x_});
    torch::autograd::variable_list grad_out_v2({grad_y_});
    auto grad_x_second = torch::autograd::grad(out_v2, in_v2, grad_out_v2);
    // Intermediate tensors (grad_y_computed, etc.) auto-released on scope exit
    return fromTorchTensor(grad_x_second[0]);
}



lean_object* lean_torch_tensor_softmax(lean_obj_arg shape, b_lean_obj_arg a, uint32_t dim_bits) {
  lean_dec(shape);
  const int32_t dim = static_cast<int32_t>(dim_bits);
  auto a_ = borrowTensor(a);
  auto c_ = torch::softmax(a_, dim);
  return fromTorchTensor(c_);
}

lean_object* lean_torch_tensor_cross_entropy(lean_obj_arg s, b_lean_obj_arg a, b_lean_obj_arg b) {
  auto a_ = borrowTensor(a);
  auto b_ = borrowTensor(b);
  auto c_ = torch::nn::functional::cross_entropy(a_, b_);
  lean_dec(s);
  return fromTorchTensor(c_);
}

// Cross entropy with no reduction (per-element loss)
static void validateCrossEntropyTargets(b_lean_obj_arg logits, b_lean_obj_arg targets) {
  const auto input = borrowTensor(logits);
  const auto labels = borrowTensor(targets);
  TORCH_CHECK(input.dim() == 2 && labels.dim() == 1 && input.size(0) == labels.size(0),
              "cross entropy requires logits [N,C] and targets [N]");
  TORCH_CHECK(input.is_floating_point() && !input.is_complex(),
              "cross entropy requires real floating logits");
  TORCH_CHECK(labels.scalar_type() == torch::kInt64, "cross entropy requires Int64 targets");
  TORCH_CHECK(input.device() == labels.device(), "cross entropy requires matching devices");
  // Validate before dispatch so invalid labels do not cause an asynchronous
  // device-side assertion. Preserve PyTorch's default ignore_index = -100.
  TORCH_CHECK(((labels == -100) | ((labels >= 0) & (labels < input.size(1)))).all().item<bool>(),
              "cross entropy target is outside the class range (except ignore_index -100)");
}

lean_object* lean_torch_cross_entropy_none(uint64_t /*n*/, uint64_t /*c*/, b_lean_obj_arg logits, b_lean_obj_arg targets) {
  auto logits_ = borrowTensor(logits);
  auto targets_ = borrowTensor(targets);
  auto options = torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone);
  auto result_ = torch::nn::functional::cross_entropy(logits_, targets_, options);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_cross_entropy_none_checked(uint64_t n, uint64_t c,
    b_lean_obj_arg logits, b_lean_obj_arg targets) {
  return tyr::checkedCall("crossEntropyNone", [&] {
    validateCrossEntropyTargets(logits, targets);
    return lean_torch_cross_entropy_none(n, c, logits, targets);
  });
}

lean_object *lean_torch_to_string(lean_obj_arg shape, b_lean_obj_arg t) {
  lean_dec(shape);
  auto tensor = borrowTensor(t);
  std::ostringstream stream;
  stream << tensor;

  return lean_mk_string(stream.str().c_str());
}

lean_object* lean_torch_tensor_print(lean_object /* s */, b_lean_obj_arg t) {
  auto tensor = borrowTensor(t);
  std::cout << tensor << std::endl;
  return lean_io_result_mk_ok(lean_box(0));
}

lean_object* lean_torch_linear(
  uint64_t /*m*/,
  uint64_t /*n*/,
  uint64_t /*b*/,
  b_lean_obj_arg x,
  b_lean_obj_arg M
) {
  auto M_ = borrowTensor(M);
  auto x_ = borrowTensor(x);
  auto y_ = torch::linear(x_, M_);
  return fromTorchTensor(y_);
}

lean_object* lean_torch_affine(
  lean_obj_arg /*m*/,
  lean_obj_arg /*n*/,
  lean_obj_arg /*b*/,
  b_lean_obj_arg x,
  b_lean_obj_arg M,
  b_lean_obj_arg b
) {
  auto M_ = borrowTensor(M);
  auto b_ = borrowTensor(b);
  auto x_ = borrowTensor(x);
  auto y_ = torch::linear(x_, M_, b_);
  return fromTorchTensor(y_);
}

// Linear projection for 3D input: [batch, seq, in] @ [out, in]^T -> [batch, seq, out]
lean_object* lean_torch_linear3d(
  uint64_t /*batch*/,
  uint64_t /*seq*/,
  uint64_t /*in_dim*/,
  uint64_t /*out_dim*/,
  b_lean_obj_arg x,
  b_lean_obj_arg weight
) {
  auto x_ = borrowTensor(x);
  auto weight_ = borrowTensor(weight);
  // torch::linear does x @ weight.T, which is what we want
  auto y_ = torch::linear(x_, weight_);
  return fromTorchTensor(y_);
}

// Affine for 3D input: linear + bias with broadcasting
lean_object* lean_torch_affine3d(
  uint64_t /*batch*/,
  uint64_t /*seq*/,
  uint64_t /*in_dim*/,
  uint64_t /*out_dim*/,
  b_lean_obj_arg x,
  b_lean_obj_arg weight,
  b_lean_obj_arg bias
) {
  auto x_ = borrowTensor(x);
  auto weight_ = borrowTensor(weight);
  auto bias_ = borrowTensor(bias);
  auto y_ = torch::linear(x_, weight_, bias_);
  return fromTorchTensor(y_);
}

lean_object* lean_torch_conv2d(
  lean_obj_arg /*input_shape*/,
  lean_obj_arg /*weight_shape*/,
  b_lean_obj_arg input,
  b_lean_obj_arg weight,
  lean_obj_arg stride,
  lean_obj_arg padding,
  lean_obj_arg dilation
) {
  auto input_ = borrowTensor(input);
  auto weight_ = borrowTensor(weight);
  auto stride_ = getShape(stride);
  auto padding_ = getShape(padding);
  auto dilation_ = getShape(dilation);
  lean_dec(stride);
  lean_dec(padding);
  lean_dec(dilation);

  auto options = torch::nn::functional::Conv2dFuncOptions()
    .stride(stride_)
    .padding(padding_)
    .dilation(dilation_);
  auto output_ = torch::nn::functional::conv2d(input_, weight_, options);
  return fromTorchTensor(output_);
}

lean_object* lean_torch_conv2d_bias(
  lean_obj_arg input_shape,
  lean_obj_arg weight_shape,
  lean_obj_arg bias_shape,
  b_lean_obj_arg input,
  b_lean_obj_arg weight,
  b_lean_obj_arg bias,
  lean_obj_arg stride,
  lean_obj_arg padding,
  lean_obj_arg dilation
) {
  lean_dec(input_shape);
  lean_dec(weight_shape);
  lean_dec(bias_shape);
  OwnedLeanObject stride_owner(stride, lean_dec);
  OwnedLeanObject padding_owner(padding, lean_dec);
  OwnedLeanObject dilation_owner(dilation, lean_dec);
  auto input_ = borrowTensor(input);
  auto weight_ = borrowTensor(weight);
  auto bias_ = borrowTensor(bias);
  auto stride_ = getShape(stride);
  auto padding_ = getShape(padding);
  auto dilation_ = getShape(dilation);
  stride_owner.reset();
  padding_owner.reset();
  dilation_owner.reset();

  // Use at::conv2d which takes bias as a parameter
  auto output_ = at::conv2d(input_, weight_, bias_, stride_, padding_, dilation_);
  return fromTorchTensor(output_);
}

lean_object* lean_torch_conv2d_bias_checked(
    lean_obj_arg input_shape, lean_obj_arg weight_shape, lean_obj_arg bias_shape,
    b_lean_obj_arg input, b_lean_obj_arg weight, b_lean_obj_arg bias,
    lean_obj_arg stride, lean_obj_arg padding, lean_obj_arg dilation) {
  return tyr::checkedCall("conv2dBias", [&] {
    return lean_torch_conv2d_bias(input_shape, weight_shape, bias_shape,
      input, weight, bias, stride, padding, dilation);
  });
}

lean_object* lean_torch_conv1d(
  lean_obj_arg input_shape,
  lean_obj_arg weight_shape,
  b_lean_obj_arg input,
  b_lean_obj_arg weight,
  uint64_t stride,
  uint64_t padding,
  uint64_t dilation
) {
  lean_dec(input_shape);
  lean_dec(weight_shape);
  auto input_ = borrowTensor(input);
  auto weight_ = borrowTensor(weight);
  auto options = torch::nn::functional::Conv1dFuncOptions()
    .stride(stride)
    .padding(padding) 
    .dilation(dilation);
  auto output_ = torch::nn::functional::conv1d(input_, weight_, options);
  return fromTorchTensor(output_);
}

lean_object* lean_torch_conv1d_checked(
    lean_obj_arg input_shape, lean_obj_arg weight_shape,
    b_lean_obj_arg input, b_lean_obj_arg weight,
    uint64_t stride, uint64_t padding, uint64_t dilation) {
  return tyr::checkedCall("conv1d", [&] {
    return lean_torch_conv1d(input_shape, weight_shape, input, weight, stride, padding, dilation);
  });
}

lean_object* lean_torch_conv1d_group_bias(
  lean_obj_arg input_shape,
  lean_obj_arg weight_shape,
  lean_obj_arg bias_shape,
  b_lean_obj_arg input,
  b_lean_obj_arg weight,
  b_lean_obj_arg bias,
  uint64_t stride,
  uint64_t padding,
  uint64_t dilation,
  uint64_t groups
) {
  lean_dec(input_shape);
  lean_dec(weight_shape);
  lean_dec(bias_shape);
  auto input_ = borrowTensor(input);
  auto weight_ = borrowTensor(weight);
  auto bias_ = borrowTensor(bias);
  auto output_ = at::conv1d(
    input_,
    weight_,
    bias_,
    {static_cast<int64_t>(stride)},
    {static_cast<int64_t>(padding)},
    {static_cast<int64_t>(dilation)},
    static_cast<int64_t>(groups)
  );
  return fromTorchTensor(output_);
}

lean_object* lean_torch_conv1d_group_bias_checked(
    lean_obj_arg input_shape, lean_obj_arg weight_shape, lean_obj_arg bias_shape,
    b_lean_obj_arg input, b_lean_obj_arg weight, b_lean_obj_arg bias,
    uint64_t stride, uint64_t padding, uint64_t dilation, uint64_t groups) {
  return tyr::checkedCall("conv1dGroups", [&] {
    return lean_torch_conv1d_group_bias(input_shape, weight_shape, bias_shape,
      input, weight, bias, stride, padding, dilation, groups);
  });
}

lean_object* lean_torch_adaptive_avg_pool2d(
  lean_obj_arg /*input_shape*/,
  b_lean_obj_arg input,
  lean_obj_arg output_size
) {
  auto input_ = borrowTensor(input);
  auto output_size_ = getShape(output_size);
  lean_dec(output_size);
  auto output_ = torch::nn::functional::adaptive_avg_pool2d(input_, torch::nn::functional::AdaptiveAvgPool2dFuncOptions(output_size_));
  return fromTorchTensor(output_);
}

lean_object* lean_torch_avg_pool2d(
  lean_obj_arg input_shape,
  b_lean_obj_arg input,
  lean_obj_arg kernel_size,
  lean_obj_arg stride,
  lean_obj_arg padding
) {
  lean_dec(input_shape);
  OwnedLeanObject kernel_size_owner(kernel_size, lean_dec);
  OwnedLeanObject stride_owner(stride, lean_dec);
  OwnedLeanObject padding_owner(padding, lean_dec);
  auto input_ = borrowTensor(input);
  auto kernel_size_ = getShape(kernel_size);
  auto stride_ = getShape(stride);
  auto padding_ = getShape(padding);
  kernel_size_owner.reset();
  stride_owner.reset();
  padding_owner.reset();
  
  auto options = torch::nn::functional::AvgPool2dFuncOptions(kernel_size_);
  if (stride_.size() > 0) {
    options.stride(stride_);
  }
  options.padding(padding_);
  
  auto output_ = torch::nn::functional::avg_pool2d(input_, options);
  return fromTorchTensor(output_);
}

lean_object* lean_torch_max_pool2d(
  lean_obj_arg /*input_shape*/,
  b_lean_obj_arg input,
  lean_obj_arg kernel_size,
  lean_obj_arg stride,
  lean_obj_arg padding
) {
  auto input_ = borrowTensor(input);
  auto kernel_size_ = getShape(kernel_size);
  auto stride_ = getShape(stride);
  auto padding_ = getShape(padding);
  lean_dec(kernel_size);
  lean_dec(stride);
  lean_dec(padding);
  
  auto options = torch::nn::functional::MaxPool2dFuncOptions(kernel_size_);
  if (stride_.size() > 0) {
    options.stride(stride_);
  }
  options.padding(padding_);
  
  auto output_ = torch::nn::functional::max_pool2d(input_, options);
  return fromTorchTensor(output_);
}

lean_object* lean_torch_dropout(lean_obj_arg s, b_lean_obj_arg input, double p, uint8_t training) {
  return tyr::checkedIoCall("dropout", [&] {
  lean_dec(s);
  try {
    auto input_ = borrowTensor(input);
    auto output_ = torch::nn::functional::dropout(input_, torch::nn::functional::DropoutFuncOptions().p(p).training(training));
    return lean_io_result_mk_ok(fromTorchTensor(output_));
  } catch (const c10::Error& e) {
    return mkC10IoError("dropout", e);
  } catch (const std::exception& e) {
    return mkIoUserError(std::string("dropout: ") + e.what());
  }
  });
}

lean_object* lean_torch_mse_loss(lean_obj_arg /*s*/, b_lean_obj_arg input, b_lean_obj_arg target, lean_obj_arg reduction) {
  auto input_ = borrowTensor(input);
  auto target_ = borrowTensor(target);
  auto reduction_str = lean_string_cstr(reduction);
  lean_dec(reduction);
  torch::nn::functional::MSELossFuncOptions options;
  if (std::string(reduction_str) == "mean") {
    options.reduction(torch::kMean);
  } else if (std::string(reduction_str) == "sum") {
    options.reduction(torch::kSum);
  } else {
    options.reduction(torch::kNone);
  }
  auto output_ = torch::nn::functional::mse_loss(input_, target_, options);
  return fromTorchTensor(output_);
}

lean_object* lean_torch_log_softmax(b_lean_obj_arg input, uint32_t dim) {
  auto input_ = borrowTensor(input);
  auto output_ = torch::nn::functional::log_softmax(input_, torch::nn::functional::LogSoftmaxFuncOptions(static_cast<int>(dim)));
  return fromTorchTensor(output_);
}

lean_object* lean_torch_leaky_relu(lean_obj_arg s, b_lean_obj_arg input, double negative_slope) {
  lean_dec(s);
  auto input_ = borrowTensor(input);
  auto output_ = torch::nn::functional::leaky_relu(input_, torch::nn::functional::LeakyReLUFuncOptions().negative_slope(negative_slope));
  return fromTorchTensor(output_);
}

lean_object* lean_torch_batch_norm(
  lean_obj_arg /*b*/,
  lean_obj_arg /*c*/,
  lean_obj_arg /*h*/,
  lean_obj_arg /*w*/,
  b_lean_obj_arg input,
  lean_obj_arg weight,
  lean_obj_arg bias,
  lean_obj_arg running_mean,
  lean_obj_arg running_var,
  int training,
  double momentum,
  double eps
) {
  auto input_ = borrowTensor(input);
  torch::Tensor weight_ = torch::Tensor();
  torch::Tensor bias_ = torch::Tensor();
  torch::Tensor running_mean_ = torch::Tensor();
  torch::Tensor running_var_ = torch::Tensor();
  
  // Handle optional weight/bias and running statistics
  if (!lean_is_scalar(weight)) {
    weight_ = borrowTensor(lean_ctor_get(weight, 0));
  }
  if (!lean_is_scalar(bias)) {
    bias_ = borrowTensor(lean_ctor_get(bias, 0));
  }
  if (!lean_is_scalar(running_mean)) {
    running_mean_ = borrowTensor(lean_ctor_get(running_mean, 0));
  }
  if (!lean_is_scalar(running_var)) {
    running_var_ = borrowTensor(lean_ctor_get(running_var, 0));
  }

  // Dec arguments as we own them
  lean_dec(weight);
  lean_dec(bias);
  lean_dec(running_mean);
  lean_dec(running_var);
  
  auto options = torch::nn::functional::BatchNormFuncOptions()
    .weight(weight_)
    .bias(bias_)
    .training(training)
    .momentum(momentum)
    .eps(eps);
  
  auto output_ = torch::nn::functional::batch_norm(
    input_, 
    running_mean_,
    running_var_,
    options
  );
  
  return fromTorchTensor(output_);
}

// Layer normalization
lean_object* lean_torch_layer_norm(
  lean_obj_arg /*s*/,
  b_lean_obj_arg input,
  lean_obj_arg normalized_shape,
  lean_obj_arg weight,
  lean_obj_arg bias,
  double eps
) {
  auto input_ = borrowTensor(input);
  auto shape_array = getShape(normalized_shape); lean_dec(normalized_shape);

  // Initialize to empty tensors for safe conditional release
  torch::Tensor weight_ = torch::Tensor();
  torch::Tensor bias_ = torch::Tensor();

  if (!lean_is_scalar(weight)) {
    lean_object* w = lean_ctor_get(weight, 0);
    weight_ = borrowTensor(w);
  }
  if (!lean_is_scalar(bias)) {
    lean_object* b = lean_ctor_get(bias, 0);
    bias_ = borrowTensor(b);
  }
  lean_dec(weight);
  lean_dec(bias);

  auto output_ = torch::layer_norm(
    input_,
    c10::IntArrayRef(shape_array.data(), shape_array.size()),
    weight_.defined() ? c10::optional<torch::Tensor>(weight_) : c10::nullopt,
    bias_.defined() ? c10::optional<torch::Tensor>(bias_) : c10::nullopt,
    eps
  );


  return fromTorchTensor(output_);
}

// Group normalization
lean_object* lean_torch_group_norm(
  lean_obj_arg s,
  b_lean_obj_arg input,
  uint64_t num_groups,
  lean_obj_arg weight,
  lean_obj_arg bias,
  double eps
) {
  lean_dec(s);
  OwnedLeanObject weight_owner(weight, lean_dec);
  OwnedLeanObject bias_owner(bias, lean_dec);
  auto input_ = borrowTensor(input);

  // Initialize to empty tensors for safe conditional release
  torch::Tensor weight_ = torch::Tensor();
  torch::Tensor bias_ = torch::Tensor();

  if (!lean_is_scalar(weight)) {
    lean_object* w = lean_ctor_get(weight, 0);
    weight_ = borrowTensor(w);
  }
  if (!lean_is_scalar(bias)) {
    lean_object* b = lean_ctor_get(bias, 0);
    bias_ = borrowTensor(b);
  }
  weight_owner.reset();
  bias_owner.reset();

  auto output_ = torch::group_norm(
    input_,
    num_groups,
    weight_.defined() ? c10::optional<torch::Tensor>(weight_) : c10::nullopt,
    bias_.defined() ? c10::optional<torch::Tensor>(bias_) : c10::nullopt,
    eps
  );


  return fromTorchTensor(output_);
}

// Instance normalization
lean_object* lean_torch_instance_norm(
  lean_obj_arg /*s*/,
  b_lean_obj_arg input,
  lean_obj_arg running_mean,
  lean_obj_arg running_var,
  lean_obj_arg weight,
  lean_obj_arg bias,
  uint8_t use_input_stats,
  double momentum,
  double eps
) {
  auto input_ = borrowTensor(input);

  // Initialize to empty tensors for safe conditional release
  torch::Tensor running_mean_ = torch::Tensor();
  torch::Tensor running_var_ = torch::Tensor();
  torch::Tensor weight_ = torch::Tensor();
  torch::Tensor bias_ = torch::Tensor();

  if (!lean_is_scalar(running_mean)) {
    lean_object* rm = lean_ctor_get(running_mean, 0);
    running_mean_ = borrowTensor(rm);
  }
  if (!lean_is_scalar(running_var)) {
    lean_object* rv = lean_ctor_get(running_var, 0);
    running_var_ = borrowTensor(rv);
  }
  if (!lean_is_scalar(weight)) {
    lean_object* w = lean_ctor_get(weight, 0);
    weight_ = borrowTensor(w);
  }
  if (!lean_is_scalar(bias)) {
    lean_object* b = lean_ctor_get(bias, 0);
    bias_ = borrowTensor(b);
  }
  lean_dec(running_mean);
  lean_dec(running_var);
  lean_dec(weight);
  lean_dec(bias);

  auto output_ = torch::instance_norm(
    input_,
    weight_.defined() ? c10::optional<torch::Tensor>(weight_) : c10::nullopt,
    bias_.defined() ? c10::optional<torch::Tensor>(bias_) : c10::nullopt,
    running_mean_.defined() ? c10::optional<torch::Tensor>(running_mean_) : c10::nullopt,
    running_var_.defined() ? c10::optional<torch::Tensor>(running_var_) : c10::nullopt,
    use_input_stats,
    momentum,
    eps,
    true  // cudnn_enabled
  );


  return fromTorchTensor(output_);
}

// Binary cross entropy loss
lean_object* lean_torch_binary_cross_entropy(lean_obj_arg s, b_lean_obj_arg input, b_lean_obj_arg target, lean_obj_arg weight, lean_obj_arg reduction) {
  lean_dec(s);
  OwnedLeanObject weight_owner(weight, lean_dec);
  OwnedLeanObject reduction_owner(reduction, lean_dec);
  auto input_ = borrowTensor(input);
  auto target_ = borrowTensor(target);
  
  torch::Tensor weight_;
  bool has_weight = !lean_is_scalar(weight);

  if (has_weight) {
    lean_object* w = lean_ctor_get(weight, 0);
    weight_ = borrowTensor(w);
  }
  
  auto reduction_str = lean_string_cstr(reduction);
  int64_t reduction_mode = 1; // mean
  if (strcmp(reduction_str, "none") == 0) {
    reduction_mode = 0;
  } else if (strcmp(reduction_str, "sum") == 0) {
    reduction_mode = 2;
  }
  
  weight_owner.reset();
  reduction_owner.reset();
  
  auto output_ = torch::binary_cross_entropy(
    input_, 
    target_,
    has_weight ? c10::optional<torch::Tensor>(weight_) : c10::nullopt,
    reduction_mode
  );
  
  
  return fromTorchTensor(output_);
}

// L1 loss
lean_object* lean_torch_l1_loss(lean_obj_arg /*s*/, b_lean_obj_arg input, b_lean_obj_arg target, lean_obj_arg reduction) {
  auto input_ = borrowTensor(input);
  auto target_ = borrowTensor(target);
  
  auto reduction_str = lean_string_cstr(reduction);
  int64_t reduction_mode = 1; // mean
  if (strcmp(reduction_str, "none") == 0) {
    reduction_mode = 0;
  } else if (strcmp(reduction_str, "sum") == 0) {
    reduction_mode = 2;
  }
  
  lean_dec(reduction);
  
  auto output_ = torch::l1_loss(
    input_, 
    target_,
    reduction_mode
  );
  
  
  return fromTorchTensor(output_);
}

// Smooth L1 loss 
lean_object* lean_torch_smooth_l1_loss(lean_obj_arg /*s*/, b_lean_obj_arg input, b_lean_obj_arg target, lean_obj_arg reduction, double beta) {
  auto input_ = borrowTensor(input);
  auto target_ = borrowTensor(target);
  
  auto reduction_str = lean_string_cstr(reduction);
  int64_t reduction_mode = 1; // mean
  if (strcmp(reduction_str, "none") == 0) {
    reduction_mode = 0;
  } else if (strcmp(reduction_str, "sum") == 0) {
    reduction_mode = 2;
  }
  
  lean_dec(reduction);
  
  auto output_ = torch::smooth_l1_loss(
    input_, 
    target_,
    reduction_mode,
    beta
  );
  
  
  return fromTorchTensor(output_);
}

// Conv3D
lean_object* lean_torch_conv3d(
  lean_obj_arg /*input_shape*/,
  lean_obj_arg /*weight_shape*/,
  b_lean_obj_arg input,
  b_lean_obj_arg weight,
  lean_obj_arg stride,
  lean_obj_arg padding,
  lean_obj_arg dilation
) {
  auto input_ = borrowTensor(input);
  auto weight_ = borrowTensor(weight);
  auto stride_ = getShape(stride); lean_dec(stride);
  auto padding_ = getShape(padding); lean_dec(padding);
  auto dilation_ = getShape(dilation); lean_dec(dilation);
  
  auto options = torch::nn::functional::Conv3dFuncOptions()
    .stride(stride_)
    .padding(padding_) 
    .dilation(dilation_);
  auto output_ = torch::nn::functional::conv3d(input_, weight_, options);
  
  
  return fromTorchTensor(output_);
}

// Embedding for 2D input: [batch, seq] -> [batch, seq, embed]
lean_object* lean_torch_embedding(
  uint64_t /*batch*/,
  uint64_t /*seq*/,
  uint64_t /*vocab*/,
  uint64_t /*embed*/,
  b_lean_obj_arg input,
  b_lean_obj_arg weight,
  lean_obj_arg padding_idx,
  lean_obj_arg max_norm,
  double norm_type,
  uint8_t scale_grad_by_freq,
  uint8_t sparse
) {
  lean_dec(max_norm); // Currently unused by the native embedding operation.
  OwnedLeanObject padding_idx_owner(padding_idx, lean_dec);
  auto input_ = borrowTensor(input);
  auto weight_ = borrowTensor(weight);

  int64_t padding_idx_val = -1;
  if (!lean_is_scalar(padding_idx)) {
    lean_object* idx = lean_ctor_get(padding_idx, 0);
    padding_idx_val = static_cast<int64_t>(lean_int64_of_int(idx));
  }

  padding_idx_owner.reset();

  auto output_ = torch::embedding(
    weight_,
    input_,
    padding_idx_val,
    scale_grad_by_freq,
    sparse
  );

  return fromTorchTensor(output_);
}

// Embedding for 1D input: [seq] -> [seq, embed]
lean_object* lean_torch_embedding_1d(
  uint64_t /*seq*/,
  uint64_t /*vocab*/,
  uint64_t /*embed*/,
  b_lean_obj_arg input,
  b_lean_obj_arg weight,
  lean_obj_arg padding_idx,
  lean_obj_arg max_norm,
  double norm_type,
  uint8_t scale_grad_by_freq,
  uint8_t sparse
) {
  lean_dec(max_norm); // Currently unused by the native embedding operation.
  OwnedLeanObject padding_idx_owner(padding_idx, lean_dec);
  auto input_ = borrowTensor(input);
  auto weight_ = borrowTensor(weight);

  int64_t padding_idx_val = -1;
  if (!lean_is_scalar(padding_idx)) {
    lean_object* idx = lean_ctor_get(padding_idx, 0);
    padding_idx_val = static_cast<int64_t>(lean_int64_of_int(idx));
  }

  padding_idx_owner.reset();

  auto output_ = torch::embedding(
    weight_,
    input_,
    padding_idx_val,
    scale_grad_by_freq,
    sparse
  );

  return fromTorchTensor(output_);
}

// Matrix multiplication functions (critical for transformers)
// Lean passes both Shape implicits of `matmul_impl {s1 s2 : Shape}`.
lean_object* lean_torch_matmul(lean_obj_arg s1, lean_obj_arg s2, b_lean_obj_arg input, b_lean_obj_arg other) {
  lean_dec(s1);
  lean_dec(s2);
  auto input_ = borrowTensor(input);
  auto other_ = borrowTensor(other);
  auto result_ = torch::matmul(input_, other_);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_bmm(
    uint64_t /*b*/,
    uint64_t /*m*/,
    uint64_t /*n*/,
    uint64_t /*k*/,
    b_lean_obj_arg input,
    b_lean_obj_arg mat2) {
  auto input_ = borrowTensor(input);
  auto mat2_ = borrowTensor(mat2);
  auto result_ = torch::bmm(input_, mat2_);
  return fromTorchTensor(result_);
}

// 4D batched matmul for multi-head attention: [b,h,m,k] @ [b,h,k,n] -> [b,h,m,n]
lean_object* lean_torch_bmm4d(
    uint64_t /*b*/,
    uint64_t /*h*/,
    uint64_t /*m*/,
    uint64_t /*k*/,
    uint64_t /*n*/,
    b_lean_obj_arg input,
    b_lean_obj_arg mat2
) {
  auto input_ = borrowTensor(input);
  auto mat2_ = borrowTensor(mat2);
  // torch::matmul handles 4D broadcasting correctly
  auto result_ = torch::matmul(input_, mat2_);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_mm(uint64_t m, uint64_t n, uint64_t k, b_lean_obj_arg input, b_lean_obj_arg mat2) {
  auto input_ = borrowTensor(input);
  auto mat2_ = borrowTensor(mat2);
  auto result_ = torch::mm(input_, mat2_);
  return fromTorchTensor(result_);
}

// Tensor operations (essential for attention mechanisms)
lean_object* lean_torch_transpose(lean_obj_arg s, b_lean_obj_arg input, uint64_t dim0, uint64_t dim1) {
  return tyr::checkedCall("transpose", [&] {
    lean_dec(s);
    auto input_ = borrowTensor(input);
    // libtorch accepts axis zero on a scalar; reject wrapped UInt64 axes.
    const auto axes = static_cast<uint64_t>(std::max<int64_t>(1, input_.dim()));
    TORCH_CHECK(dim0 < axes && dim1 < axes, "dimension out of range");
    return fromTorchTensor(torch::transpose(input_, static_cast<int64_t>(dim0), static_cast<int64_t>(dim1)));
  });
}

lean_object* lean_torch_cat(lean_obj_arg /*s*/, b_lean_obj_arg tensors, lean_obj_arg dim_obj) {
  int64_t dim = static_cast<int64_t>(lean_int64_of_int(dim_obj));
  lean_dec(dim_obj);
  if (!lean_is_array(tensors)) {
    lean_internal_panic("lean_torch_cat: expected array");
  }
  std::vector<torch::Tensor> tensor_list;
  size_t arr_size = lean_array_size(tensors);
  for (size_t i = 0; i < arr_size; i++) {
    auto tensor_obj = lean_array_get_core(tensors, i);
    tensor_list.push_back(borrowTensor(tensor_obj));
  }
  // tensors is borrowed (@& in Lean), do not lean_dec here
  auto result_ = torch::cat(tensor_list, dim);
  // tensor_list tensors are auto-released when vector goes out of scope
  return fromTorchTensor(result_);
}

// Direct 2-tensor concatenation without intermediate reshapes
lean_object* lean_torch_cat2(
    lean_obj_arg s1,
    lean_obj_arg s2,
    b_lean_obj_arg t1,
    b_lean_obj_arg t2,
    lean_obj_arg dim_obj
) {
  lean_dec(s1);
  lean_dec(s2);
  OwnedLeanObject dim_obj_owner(dim_obj, lean_dec);
  int64_t dim = static_cast<int64_t>(lean_int64_of_int(dim_obj));
  dim_obj_owner.reset();
  auto t1_ = borrowTensor(t1);
  auto t2_ = borrowTensor(t2);
  auto result_ = torch::cat({t1_, t2_}, dim);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_erase_shape(lean_obj_arg /*s*/, b_lean_obj_arg t) {
  auto t_ = borrowTensor(t);
  return fromTorchTensor(t_);
}

lean_object* lean_torch_cat_dyn(b_lean_obj_arg tensors, lean_obj_arg dim_obj) {
  OwnedLeanObject dim_obj_owner(dim_obj, lean_dec);
  int64_t dim = static_cast<int64_t>(lean_int64_of_int(dim_obj));
  dim_obj_owner.reset();
  if (!lean_is_array(tensors)) {
    lean_internal_panic("lean_torch_cat_dyn: expected array");
  }
  std::vector<torch::Tensor> tensor_list;
  size_t arr_size = lean_array_size(tensors);
  for (size_t i = 0; i < arr_size; i++) {
    auto tensor_obj = lean_array_get_core(tensors, i);
    tensor_list.push_back(borrowTensor(tensor_obj));
  }
  auto result_ = torch::cat(tensor_list, dim);
  return fromTorchTensor(result_);
}

// Attention mechanism functions
lean_object* lean_torch_scaled_dot_product_attention(
  lean_obj_arg /*s*/, 
  b_lean_obj_arg query, 
  b_lean_obj_arg key, 
  b_lean_obj_arg value,
  lean_obj_arg attn_mask,
  double dropout_p,
  uint8_t is_causal
) {
  auto query_ = borrowTensor(query);
  auto key_ = borrowTensor(key);
  auto value_ = borrowTensor(value);
  
  torch::Tensor attn_mask_;
  bool has_mask = !lean_is_scalar(attn_mask);
  if (has_mask) {
    lean_object* m = lean_ctor_get(attn_mask, 0);
    attn_mask_ = borrowTensor(m);
  }
  lean_dec(attn_mask);
  
  auto result_ = torch::scaled_dot_product_attention(
    query_, 
    key_, 
    value_,
    has_mask ? c10::optional<torch::Tensor>(attn_mask_) : c10::nullopt,
    dropout_p,
    is_causal
  );
  
  
  return fromTorchTensor(result_);
}

// Softmax with dimension parameter (needed for attention)
lean_object* lean_torch_softmax_dim(lean_obj_arg /*s*/, b_lean_obj_arg input, int64_t dim) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::softmax(input_, dim);
  return fromTorchTensor(result_);
}

// Cumulative sum along a dimension.
lean_object* lean_torch_cumsum(lean_obj_arg s, b_lean_obj_arg input, uint64_t dimBits) {
  lean_dec(s);
  const auto dim = std::bit_cast<int64_t>(dimBits);
  auto input_ = borrowTensor(input);
  auto result_ = torch::cumsum(input_, dim);
  return fromTorchTensor(result_);
}

// Extended activation functions with parameters
lean_object* lean_torch_hardtanh_params(lean_obj_arg /*s*/, b_lean_obj_arg input, double min_val, double max_val) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::hardtanh(input_, min_val, max_val);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_celu_params(lean_obj_arg /*s*/, b_lean_obj_arg input, double alpha) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::celu(input_, alpha);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_rrelu_params(lean_obj_arg /*s*/, b_lean_obj_arg input, double lower, double upper, uint8_t training) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::rrelu(input_, lower, upper, training);
  return fromTorchTensor(result_);
}

// Additional tensor operations
lean_object* lean_torch_split(lean_obj_arg /*s*/, b_lean_obj_arg tensor, int split_size, int dim) {
  auto tensor_ = borrowTensor(tensor);
  auto result_tensors = torch::split(tensor_, split_size, dim);
  
  // Create Lean array of tensors
  lean_object* result_array = lean_alloc_array(result_tensors.size(), result_tensors.size());
  for (size_t i = 0; i < result_tensors.size(); i++) {
    lean_array_set_core(result_array, i, fromTorchTensor(result_tensors[i]));
  }
  return result_array;
}

lean_object* lean_torch_chunk(lean_obj_arg /*s*/, b_lean_obj_arg tensor, int chunks, int dim) {
  auto tensor_ = borrowTensor(tensor);
  auto result_tensors = torch::chunk(tensor_, chunks, dim);
  
  // Create Lean array of tensors
  lean_object* result_array = lean_alloc_array(result_tensors.size(), result_tensors.size());
  for (size_t i = 0; i < result_tensors.size(); i++) {
    lean_array_set_core(result_array, i, fromTorchTensor(result_tensors[i]));
  }
  return result_array;
}

lean_object* lean_torch_unsqueeze(lean_obj_arg s, b_lean_obj_arg input, lean_obj_arg dim) {
  lean_dec(s);
  OwnedLeanObject dim_owner(dim, lean_dec);
  auto input_ = borrowTensor(input);
  auto dim_ = lean_uint64_of_nat(dim);
  auto result_ = torch::unsqueeze(input_, static_cast<int64_t>(dim_));
  return fromTorchTensor(result_);
}

lean_object* lean_torch_squeeze(lean_obj_arg s, b_lean_obj_arg input, lean_obj_arg dim) {
  lean_dec(s);
  OwnedLeanObject dim_owner(dim, lean_dec);
  auto input_ = borrowTensor(input);
  auto dim_ = lean_uint64_of_nat(dim);
  auto result_ = torch::squeeze(input_, static_cast<int64_t>(dim_));
  return fromTorchTensor(result_);
}

// Mathematical operations
lean_object* lean_torch_sqrt(lean_obj_arg s, b_lean_obj_arg input) {
  lean_dec(s);
  auto input_ = borrowTensor(input);
  auto result_ = torch::sqrt(input_);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_rsqrt(lean_obj_arg s, b_lean_obj_arg input) {
  lean_dec(s);
  auto input_ = borrowTensor(input);
  auto result_ = torch::rsqrt(input_);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_div(lean_obj_arg s, b_lean_obj_arg input, b_lean_obj_arg other) {
  lean_dec(s);
  auto input_ = borrowTensor(input);
  auto other_ = borrowTensor(other);
  auto result_ = torch::div(input_, other_);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_pow(lean_obj_arg s, b_lean_obj_arg input, double exponent) {
  lean_dec(s);
  auto input_ = borrowTensor(input);
  auto result_ = torch::pow(input_, exponent);
  return fromTorchTensor(result_);
}

// Reduction operations
lean_object* lean_torch_sum(lean_obj_arg s, b_lean_obj_arg input, lean_obj_arg dim, uint8_t keepdim) {
  lean_dec(s);
  OwnedLeanObject dim_owner(dim, lean_dec);
  auto input_ = borrowTensor(input);
  torch::Tensor result_;
  
  if (lean_is_scalar(dim)) {
    // none: sum all elements
    result_ = torch::sum(input_);
  } else {
    // some: dim is a constructor wrapping Array UInt64
    lean_object* dims_obj = lean_ctor_get(dim, 0);
    auto dims = getShape(dims_obj);
    dim_owner.reset();
    result_ = torch::sum(input_, c10::IntArrayRef(dims.data(), dims.size()), keepdim);
  }
  
  return fromTorchTensor(result_);
}

lean_object* lean_torch_mean(lean_obj_arg s, b_lean_obj_arg input, lean_obj_arg dim, uint8_t keepdim) {
  lean_dec(s);
  OwnedLeanObject dim_owner(dim, lean_dec);
  auto input_ = borrowTensor(input);
  torch::Tensor result_;

  if (lean_is_scalar(dim)) {
    // none: mean of all elements
    result_ = torch::mean(input_);
  } else {
    // some: dim is a constructor wrapping Array UInt64
    lean_object* dims_obj = lean_ctor_get(dim, 0);
    auto dims = getShape(dims_obj);
    dim_owner.reset();
    result_ = torch::mean(input_, c10::IntArrayRef(dims.data(), dims.size()), keepdim);
  }

  return fromTorchTensor(result_);
}

lean_object* lean_torch_abs(lean_obj_arg s, b_lean_obj_arg input) {
  lean_dec(s);
  auto input_ = borrowTensor(input);
  auto result_ = torch::abs(input_);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_max_all(b_lean_obj_arg input) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::max(input_);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_min_all(b_lean_obj_arg input) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::min(input_);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_max(lean_obj_arg /*s*/, b_lean_obj_arg input, int dim, uint8_t keepdim) {
  auto input_ = borrowTensor(input);
  auto result_tuple = torch::max(input_, dim, keepdim);
  
  // Return just the values (first element of tuple)
  return fromTorchTensor(std::get<0>(result_tuple));
}

lean_object* lean_torch_min(lean_obj_arg /*s*/, b_lean_obj_arg input, int dim, uint8_t keepdim) {
  auto input_ = borrowTensor(input);
  auto result_tuple = torch::min(input_, dim, keepdim);
  
  // Return just the values (first element of tuple)
  return fromTorchTensor(std::get<0>(result_tuple));
}

// Masking and conditional operations
lean_object* lean_torch_masked_fill(lean_obj_arg s, b_lean_obj_arg input, b_lean_obj_arg mask, double value) {
  lean_dec(s);
  auto input_ = borrowTensor(input);
  auto mask_ = borrowTensor(mask);
  auto result_ = input_.masked_fill(mask_, value);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_masked_select(lean_obj_arg s, b_lean_obj_arg input, b_lean_obj_arg mask) {
  lean_dec(s);
  auto input_ = borrowTensor(input);
  auto mask_ = borrowTensor(mask);
  auto result_ = input_.masked_select(mask_);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_masked_scatter(
    lean_obj_arg s, lean_obj_arg src, b_lean_obj_arg input, b_lean_obj_arg mask, b_lean_obj_arg source) {
  lean_dec(s);
  lean_dec(src);
  auto input_ = borrowTensor(input);
  auto mask_ = borrowTensor(mask);
  auto source_ = borrowTensor(source);
  auto result_ = input_.masked_scatter(mask_, source_);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_gemma4_text_experts_forward(
    uint64_t tokens,
    uint64_t num_experts,
    uint64_t top_k,
    uint64_t inter_dim,
    uint64_t hidden_size,
    b_lean_obj_arg hidden,
    b_lean_obj_arg gate_up_proj,
    b_lean_obj_arg down_proj,
    b_lean_obj_arg top_vals,
    b_lean_obj_arg top_idx) {
  auto hidden_ = borrowTensor(hidden);
  auto gate_up_proj_ = borrowTensor(gate_up_proj);
  auto down_proj_ = borrowTensor(down_proj);
  auto top_vals_ = borrowTensor(top_vals);
  auto top_idx_ = borrowTensor(top_idx).to(torch::kLong);

  auto result_ = torch::zeros(
      {static_cast<int64_t>(tokens), static_cast<int64_t>(hidden_size)},
      hidden_.options());

  auto flat_idx = top_idx_.reshape({static_cast<int64_t>(tokens * top_k)});
  auto flat_vals = top_vals_.reshape({static_cast<int64_t>(tokens * top_k)});
  auto token_ids =
      torch::arange(
          static_cast<int64_t>(tokens),
          torch::TensorOptions().dtype(torch::kLong).device(hidden_.device()))
          .unsqueeze(1)
          .expand({static_cast<int64_t>(tokens), static_cast<int64_t>(top_k)})
          .reshape({static_cast<int64_t>(tokens * top_k)});

  auto flat_idx_cpu = flat_idx.to(torch::kCPU, torch::kLong).contiguous();
  const auto* flat_idx_ptr = flat_idx_cpu.data_ptr<int64_t>();
  std::vector<uint8_t> seen(num_experts, 0);
  std::vector<int64_t> used_experts;
  used_experts.reserve(std::min<uint64_t>(num_experts, tokens * top_k));
  for (int64_t i = 0; i < static_cast<int64_t>(tokens * top_k); ++i) {
    auto expert_id = flat_idx_ptr[i];
    if (expert_id >= 0 && expert_id < static_cast<int64_t>(num_experts) &&
        !seen[expert_id]) {
      seen[expert_id] = 1;
      used_experts.push_back(expert_id);
    }
  }

  for (auto expert_id : used_experts) {
    auto mask = flat_idx.eq(expert_id);
    auto selected_tokens = token_ids.masked_select(mask);
    if (selected_tokens.numel() == 0) {
      continue;
    }

    auto selected_hidden = torch::index_select(hidden_, 0, selected_tokens);
    auto selected_weights =
        flat_vals.masked_select(mask).to(selected_hidden.scalar_type()).unsqueeze(1);

    auto gate_up = gate_up_proj_.select(0, expert_id).contiguous();
    auto gate_up_out = torch::matmul(selected_hidden, gate_up.transpose(0, 1));
    auto gate = gate_up_out.slice(1, 0, static_cast<int64_t>(inter_dim));
    auto up = gate_up_out.slice(
        1,
        static_cast<int64_t>(inter_dim),
        static_cast<int64_t>(2 * inter_dim));
    auto intermediate = at::gelu(gate) * up;

    auto down = down_proj_.select(0, expert_id).contiguous();
    auto out = torch::matmul(intermediate, down.transpose(0, 1));
    result_.index_add_(0, selected_tokens, out * selected_weights);
  }

  return fromTorchTensor(result_);
}

lean_object* lean_torch_where(lean_obj_arg s, b_lean_obj_arg condition, b_lean_obj_arg x, b_lean_obj_arg y) {
  lean_dec(s);
  auto condition_ = borrowTensor(condition);
  auto x_ = borrowTensor(x);
  auto y_ = borrowTensor(y);
  auto result_ = torch::where(condition_, x_, y_);
  return fromTorchTensor(result_);
}

// Broadcasting and expanding
lean_object* lean_torch_expand(lean_obj_arg s, b_lean_obj_arg input, lean_obj_arg size) {
  lean_dec(s);
  OwnedLeanObject size_owner(size, lean_dec);
  auto input_ = borrowTensor(input);
  auto size_ = getShape(size); size_owner.reset();
  auto result_ = input_.expand(c10::IntArrayRef(size_.data(), size_.size()));
  return fromTorchTensor(result_);
}

lean_object* lean_torch_repeat(lean_obj_arg /*s*/, b_lean_obj_arg input, lean_obj_arg repeats) {
  auto input_ = borrowTensor(input);
  auto repeats_ = getShape(repeats); lean_dec(repeats);
  auto result_ = input_.repeat(c10::IntArrayRef(repeats_.data(), repeats_.size()));
  return fromTorchTensor(result_);
}

// Scalar operations (tensor-scalar arithmetic)
lean_object* lean_torch_mul_scalar(lean_obj_arg s, b_lean_obj_arg input, double scalar) {
  lean_dec(s);
  auto input_ = borrowTensor(input);
  auto result_ = input_ * scalar;
  return fromTorchTensor(result_);
}

lean_object* lean_torch_div_scalar(lean_obj_arg s, b_lean_obj_arg input, double scalar) {
  lean_dec(s);
  auto input_ = borrowTensor(input);
  auto result_ = input_ / scalar;
  return fromTorchTensor(result_);
}

lean_object* lean_torch_add_scalar(lean_obj_arg s, b_lean_obj_arg input, double scalar) {
  lean_dec(s);
  auto input_ = borrowTensor(input);
  auto result_ = input_ + scalar;
  return fromTorchTensor(result_);
}

lean_object* lean_torch_sub_scalar(lean_obj_arg s, b_lean_obj_arg input, double scalar) {
  lean_dec(s);
  auto input_ = borrowTensor(input);
  auto result_ = input_ - scalar;
  return fromTorchTensor(result_);
}

lean_object* lean_torch_pow_tensor(lean_obj_arg /*s*/, b_lean_obj_arg input, double exponent) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::pow(input_, exponent);
  return fromTorchTensor(result_);
}

// Lower triangular (for causal masking)
lean_object* lean_torch_tril(lean_obj_arg /*s*/, b_lean_obj_arg input, int64_t diagonal) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::tril(input_, diagonal);
  return fromTorchTensor(result_);
}

// Top-k values (for sampling)
lean_object* lean_torch_topk_values(lean_obj_arg /*s*/, b_lean_obj_arg input, uint64_t k, uint64_t dim) {
  auto input_ = borrowTensor(input);
  auto dim_ = static_cast<int64_t>(dim);
  auto result_tuple = torch::topk(input_, k, dim_);
  return fromTorchTensor(std::get<0>(result_tuple));
}

// Multinomial sampling
lean_object* lean_torch_multinomial(lean_obj_arg s, b_lean_obj_arg input, uint64_t num_samples, uint8_t replacement) {
  return tyr::checkedIoCall("multinomial", [&] {
  lean_dec(s);
  try {
    auto input_ = borrowTensor(input);
    auto result_ = torch::multinomial(input_, num_samples, replacement);
    return lean_io_result_mk_ok(fromTorchTensor(result_));
  } catch (const c10::Error& e) {
    return mkC10IoError("Multinomial failed", e);
  } catch (const std::exception& e) {
    return mkStdIoError("Multinomial failed", e);
  }
  });
}

// Gradient clipping (in-place, returns norm)
lean_object* lean_torch_clip_grad_norm_(lean_obj_arg s, b_lean_obj_arg param, double max_norm) {
  return tyr::checkedIoCall("clipGradNorm", [&] {
  lean_dec(s);
  try {
    torch::NoGradGuard guard;
    auto param_ = borrowTensor(param);
    double total_norm = 0.0;
    if (param_.grad().defined()) {
      auto grad = param_.grad();
      auto norm = grad.norm();
      total_norm = norm.item<double>();
      if (total_norm > max_norm) {
        grad.mul_(max_norm / total_norm);
      }
    }
    return lean_io_result_mk_ok(lean_box_float(total_norm));
  } catch (const c10::Error& e) {
    return mkC10IoError("clip_grad_norm_ failed", e);
  } catch (const std::exception& e) {
    return mkStdIoError("clip_grad_norm_ failed", e);
  }
  });
}

// Item extraction (get scalar value from 0-d tensor)
double lean_torch_item(lean_obj_arg s, b_lean_obj_arg input) {
  lean_dec(s);
  auto input_ = borrowTensor(input);
  double result = input_.item<double>();
  return result;
}

lean_object* lean_torch_item_checked(lean_obj_arg s, b_lean_obj_arg input) {
  return tyr::checkedCall("item", [&] {
    lean_dec(s);
    return lean_box_float(borrowTensor(input).item<double>());
  });
}

// Item extraction as int64
uint64_t lean_torch_item_int(lean_obj_arg s, b_lean_obj_arg input) {
  lean_dec(s);
  auto input_ = borrowTensor(input);
  int64_t result = input_.item<int64_t>();
  return static_cast<uint64_t>(result);
}

// Argmax
lean_object* lean_torch_argmax(lean_obj_arg s, b_lean_obj_arg input, uint64_t dim) {
  lean_dec(s);
  auto input_ = borrowTensor(input);
  auto dim_ = static_cast<int64_t>(dim);
  auto result_ = torch::argmax(input_, dim_);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_argmax_checked(lean_obj_arg s, b_lean_obj_arg input, uint64_t dim) {
  return tyr::checkedCall("argmax", [&] {
    lean_dec(s);
    TORCH_CHECK(dim <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      "argmax axis is not representable as a nonnegative signed 64-bit integer");
    return fromTorchTensor(torch::argmax(borrowTensor(input), static_cast<int64_t>(dim)));
  });
}

// Create tensor from Array Int64
// Array Int64 is a regular array (not scalar array) where each element is a boxed Int64
// Int64 is stored as a ctor with the uint64 value at scalar offset 0
lean_object* lean_torch_from_int64_array(b_lean_obj_arg arr) {
  size_t len = lean_array_size(arr);

  // Allocate buffer and unbox each element
  std::vector<int64_t> data(len);
  for (size_t i = 0; i < len; i++) {
    lean_object* elem = lean_array_get_core(arr, i);
    // Int64 is stored as a ctor with uint64 at offset 0 (same as UInt64)
    data[i] = static_cast<int64_t>(lean_ctor_get_uint64(elem, 0));
  }

  // Create tensor by copying data
  auto tensor = torch::from_blob(data.data(), {static_cast<int64_t>(len)}, torch::kInt64).clone();
  return fromTorchTensor(tensor);
}

// Create tensor from Array Float
lean_object* lean_torch_from_float_array(b_lean_obj_arg arr) {
  size_t len = lean_array_size(arr);
  std::vector<float> data(len);
  for (size_t i = 0; i < len; i++) {
    data[i] = static_cast<float>(lean_unbox_float(lean_array_get_core(arr, i)));
  }
  auto tensor = torch::from_blob(data.data(), {static_cast<int64_t>(len)}, torch::kFloat32).clone();
  return fromTorchTensor(tensor);
}

// High-quality mono waveform resample using libsoxr (`soxr_hq`).
lean_object* lean_torch_resample_soxr_hq(
    b_lean_obj_arg samples,
    uint64_t orig_sr,
    uint64_t target_sr,
    lean_object* /*w*/
) {
  if (orig_sr == 0 || target_sr == 0) {
    return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string("resampleSoxrHQ: sample rates must be non-zero")));
  }

  size_t in_len = lean_array_size(samples);
  if (in_len == 0 || orig_sr == target_sr) {
    // The argument is borrowed. Retain the immutable array for the result;
    // transferring its borrowed elements individually would leave them dangling.
    lean_inc(samples);
    return lean_io_result_mk_ok(samples);
  }

  std::vector<float> in_data(in_len);
  for (size_t i = 0; i < in_len; i++) {
    in_data[i] = static_cast<float>(lean_unbox_float(lean_array_get_core(samples, i)));
  }

  double ratio = static_cast<double>(target_sr) / static_cast<double>(orig_sr);
  size_t out_len = static_cast<size_t>(std::ceil(static_cast<double>(in_len) * ratio));
  std::vector<float> out_data(out_len, 0.0f);

#if TYR_HAS_SOXR
  size_t idone = 0;
  size_t odone = 0;
  auto io_spec = soxr_io_spec(SOXR_FLOAT32_I, SOXR_FLOAT32_I);
  auto q_spec = soxr_quality_spec(SOXR_HQ, 0);
  soxr_error_t err = soxr_oneshot(
      static_cast<double>(orig_sr),
      static_cast<double>(target_sr),
      1,
      in_data.data(), in_len, &idone,
      out_data.data(), out_len, &odone,
      &io_spec, &q_spec, nullptr);
  if (err != nullptr) {
    return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string((std::string("resampleSoxrHQ failed: ") + err).c_str())));
  }
  if (odone < out_len) {
    for (size_t i = odone; i < out_len; i++) {
      out_data[i] = 0.0f;
    }
  }
#else
  return lean_io_result_mk_error(lean_mk_io_user_error(
    lean_mk_string("resampleSoxrHQ requires libsoxr (soxr.h not found at build time)")));
#endif

  lean_object* out = lean_mk_empty_array();
  for (size_t i = 0; i < out_len; i++) {
    out = lean_array_push(out, lean_box_float(static_cast<double>(out_data[i])));
  }
  return lean_io_result_mk_ok(out);
}

// Backward pass
lean_object* lean_torch_backward_unit(lean_obj_arg s, b_lean_obj_arg output, b_lean_obj_arg grad_output) {
  return tyr::checkedIoCall("backward", [&] {
  lean_dec(s);
  try {
    auto output_ = borrowTensor(output);
    auto grad_output_ = borrowTensor(grad_output);
    output_.backward(grad_output_, /*retain_graph=*/true);
    return lean_io_result_mk_ok(lean_box(0));
  } catch (const c10::Error& e) {
    return mkC10IoError("Backward failed", e);
  } catch (const std::exception& e) {
    return mkStdIoError("Backward failed", e);
  }
  });
}

// Get number of uint16 tokens in a binary file
lean_object* lean_torch_bin_file_token_count(b_lean_obj_arg path_obj, lean_object* w) {
  const char* path = lean_string_cstr(path_obj);

  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string(("Failed to open file: " + std::string(path)).c_str())));
  }

  std::streamsize size = file.tellg();
  uint64_t num_tokens = size / sizeof(uint16_t);

  return lean_io_result_mk_ok(lean_box_uint64(num_tokens));
}

// Load binary file of uint16 tokens into a 1D tensor with known size
lean_object* lean_torch_load_u16_bin(uint64_t n, b_lean_obj_arg path_obj) {
  return tyr::checkedIoCall("loadU16Bin", [&] {
  const char* path = lean_string_cstr(path_obj);
  uint64_t expected_tokens = n;

  try {
    // Open file
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
      return mkIoUserError("Failed to open file: " + std::string(path));
    }

    // Read as uint16
    std::vector<uint16_t> buffer(expected_tokens);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), expected_tokens * sizeof(uint16_t))) {
      return mkIoUserError("Failed to read file.");
    }

    // Convert to int64 tensor (PyTorch uses int64 for indices)
    std::vector<int64_t> data(expected_tokens);
    for (size_t i = 0; i < expected_tokens; i++) {
      data[i] = static_cast<int64_t>(buffer[i]);
    }

    auto tensor = torch::from_blob(data.data(), {static_cast<int64_t>(expected_tokens)},
                                    torch::kInt64).clone();

    return lean_io_result_mk_ok(fromTorchTensor(tensor));
  } catch (const c10::Error& e) {
    return mkC10IoError("load_u16_bin failed", e);
  } catch (const std::exception& e) {
    return mkStdIoError("load_u16_bin failed", e);
  }
  });
}

// Load raw float32 binary data into a tensor with known shape
lean_object* lean_torch_load_f32_bin(lean_obj_arg shape, b_lean_obj_arg path_obj, lean_object* w) {
  const char* path = lean_string_cstr(path_obj);
  auto expected_shape = getShape(shape); lean_dec(shape);

  try {
    int64_t expected_numel = 1;
    for (auto d : expected_shape) {
      expected_numel *= d;
    }
    const size_t expected_bytes = static_cast<size_t>(expected_numel) * sizeof(float);

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      return mkIoUserError("Failed to open file: " + std::string(path));
    }
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size != static_cast<std::streamsize>(expected_bytes)) {
      return mkIoUserError(
        "Float32 binary size mismatch, expected " + std::to_string(expected_bytes) +
        " bytes, got " + std::to_string(file_size));
    }

    std::vector<float> buffer(expected_numel);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), expected_bytes)) {
      return mkIoUserError("Failed to read float32 binary file.");
    }

    auto tensor = torch::from_blob(buffer.data(), expected_shape, torch::kFloat32).clone();
    return lean_io_result_mk_ok(fromTorchTensor(tensor));
  } catch (const c10::Error& e) {
    return mkC10IoError("load_f32_bin failed", e);
  } catch (const std::exception& e) {
    return mkStdIoError("load_f32_bin failed", e);
  }
}

// Index select: gather elements along a dimension
lean_object* lean_torch_index_select(
  uint64_t /*s*/,
  uint64_t /*k*/,
  b_lean_obj_arg input,
  uint64_t dim,
  b_lean_obj_arg indices
) {
  auto input_ = borrowTensor(input);
  auto indices_ = borrowTensor(indices);
  // Ensure indices are int64 (index_select requires int32 or int64)
  if (indices_.dtype() != torch::kInt64 && indices_.dtype() != torch::kInt32) {
    indices_ = indices_.to(torch::kInt64);
  }
  auto result_ = torch::index_select(input_, static_cast<int64_t>(dim), indices_);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_index_select_checked(
    uint64_t /*n*/, uint64_t /*k*/, b_lean_obj_arg input,
    uint64_t dim_bits, b_lean_obj_arg indices) {
  return tyr::checkedCall("indexSelect", [&] {
    auto data = borrowTensor(input);
    auto index = borrowTensor(indices);
    auto dim = std::bit_cast<int64_t>(dim_bits);
    TORCH_CHECK(data.dim() == 1 && index.dim() == 1,
      "indexSelect requires vector data and indices");
    TORCH_CHECK(dim == 0 || dim == -1, "indexSelect vector axis must be zero");
    TORCH_CHECK(index.scalar_type() == torch::kInt64 || index.scalar_type() == torch::kInt32,
      "indexSelect requires Int64 or Int32 indices");
    TORCH_CHECK(data.device() == index.device(), "indexSelect requires matching devices");
    // Validate before launching device indexing: CUDA out-of-bounds indexing
    // may assert asynchronously and poison the context instead of throwing.
    // Device-local comparisons copy only one validity bit back to the host.
    // If a backend lacks those kernels, the checked boundary returns an error
    // before it can launch indexing with unvalidated values.
    const auto limit = data.size(0);
    if (index.numel() > 0) {
      TORCH_CHECK((index.ge(0) & index.lt(limit)).all().item<bool>(),
        "indexSelect index is out of range for length ", limit);
    }
    return fromTorchTensor(torch::index_select(data, 0, index));
  });
}

// Slice a 1D tensor: data[start:end]
lean_object* lean_torch_slice_1d(
  lean_obj_arg /*n*/,
  b_lean_obj_arg input,
  int64_t start,
  int64_t end
) {
  auto input_ = borrowTensor(input);
  auto result_ = input_.slice(0, start, end);
  return fromTorchTensor(result_);
}

// Slice a tensor along a specified dimension: data.slice(dim, start, start+len)
// Fully dependently typed - output shape is computed from input shape and slice params
lean_object* lean_torch_slice_along_dim(
  lean_obj_arg s,
  b_lean_obj_arg input,
  uint64_t dim,
  uint64_t start,
  uint64_t len
) {
  lean_dec(s);
  auto input_ = borrowTensor(input);
  auto result_ = input_.slice(static_cast<int64_t>(dim),
                               static_cast<int64_t>(start),
                               static_cast<int64_t>(start + len));
  return fromTorchTensor(result_);
}

// Return a copy of `input` with a slice along `dim` replaced by `src` at `start`.
lean_object* lean_torch_slice_scatter_along_dim(
  lean_obj_arg s,
  lean_obj_arg src_shape,
  lean_obj_arg input,
  uint64_t dim,
  uint64_t start,
  b_lean_obj_arg src
) {
  lean_dec(s);
  lean_dec(src_shape);
  OwnedLeanObject input_owner(input, lean_dec);
  auto src_ = borrowTensor(src);
  auto input_ = takeTensor(input_owner.release());
  const auto axis = checkSliceCopy(input_, src_, dim, start);
  auto result = tensorForValueUpdate(std::move(input_));
  result.narrow(axis, static_cast<int64_t>(start), src_.size(axis)).copy_(src_);
  return fromTorchTensor(std::move(result));
}

// Allocation is effectful so separate mutable buffers cannot be commoned by Lean.
lean_object* lean_torch_clone_inference_io(lean_obj_arg shape, b_lean_obj_arg src) {
  lean_dec(shape);
  try {
    torch::NoGradGuard guard;
    return lean_io_result_mk_ok(fromTorchTensor(borrowTensor(src).clone()));
  } catch (const c10::Error& e) {
    return mkC10IoError("cloneInferenceIO", e);
  } catch (const std::exception& e) {
    return mkStdIoError("cloneInferenceIO", e);
  }
}

// Effectful cache value update: consume the destination, reuse only when safe.
lean_object* lean_torch_copy_slice_io(
  lean_obj_arg shape, lean_obj_arg src_shape, lean_obj_arg input,
  uint64_t dim, uint64_t start, b_lean_obj_arg src
) {
  lean_dec(shape);
  lean_dec(src_shape);
  OwnedLeanObject input_owner(input, lean_dec);
  try {
    torch::NoGradGuard guard;
    auto source = borrowTensor(src);
    auto destination = takeTensor(input_owner.release());
    const auto axis = checkSliceCopy(destination, source, dim, start);
    auto result = tensorForNoGradValueUpdate(std::move(destination));
    result.narrow(axis, static_cast<int64_t>(start), source.size(axis)).copy_(source);
    return lean_io_result_mk_ok(fromTorchTensor(std::move(result)));
  } catch (const c10::Error& e) {
    return mkC10IoError("copySliceIO", e);
  } catch (const std::exception& e) {
    return mkStdIoError("copySliceIO", e);
  }
}

// Slice a 2D tensor along dimension 0: data[start:start+len, :]
lean_object* lean_torch_slice_2d(
  uint64_t /*n*/,
  uint64_t /*d*/,
  b_lean_obj_arg input,
  uint64_t start,
  uint64_t len
) {
  auto input_ = borrowTensor(input);
  auto result_ = input_.slice(0,
                               static_cast<int64_t>(start),
                               static_cast<int64_t>(start + len));
  return fromTorchTensor(result_);
}

// Stack multiple 1D tensors into 2D
extern "C" LEAN_EXPORT lean_object* lean_torch_stack_1d(
  uint64_t /*k*/,
  uint64_t /*n*/,
  lean_obj_arg tensors,
  uint64_t dim
) {
  OwnedLeanObject tensors_owner(tensors, lean_dec);
  std::vector<torch::Tensor> tensor_list;
  for (size_t i = 0; i < lean_array_size(tensors); i++) {
    auto tensor_obj = lean_array_get_core(tensors, i);
    tensor_list.push_back(borrowTensor(tensor_obj));
  }
  tensors_owner.reset();
  auto result_ = torch::stack(tensor_list, static_cast<int64_t>(dim));
  return fromTorchTensor(result_);
}

// Stack multiple tensors into a new dimension
lean_object* lean_torch_stack(
  lean_obj_arg /*s*/,
  lean_obj_arg tensors,
  uint64_t dim
) {
  std::vector<torch::Tensor> tensor_list;
  for (size_t i = 0; i < lean_array_size(tensors); i++) {
    auto tensor_obj = lean_array_get_core(tensors, i);
    tensor_list.push_back(borrowTensor(tensor_obj));
  }
  lean_dec(tensors);
  auto result_ = torch::stack(tensor_list, dim);
  return fromTorchTensor(result_);
}

// Unbind a tensor along a dimension
lean_object* lean_torch_unbind(
  lean_obj_arg s,
  b_lean_obj_arg t,
  lean_obj_arg dim
) {
  OwnedLeanObject dim_owner(dim, lean_dec);
  lean_dec(s);
  // Nat is boxed in Lean, unlike UInt64/Int64. Check the axis before
  // conversion so an arbitrary-precision Nat cannot wrap into a valid axis.
  auto t_ = borrowTensor(t);
  const bool in_range = lean_nat_dec_lt(dim, lean_box(t_.dim()));
  const auto axis = lean_uint64_of_nat(dim);
  TORCH_CHECK(in_range, "unbind: dimension out of range");
  auto unbinded = torch::unbind(t_, static_cast<int64_t>(axis));
  size_t n = unbinded.size();
  lean_object* result = lean_alloc_array(n, n);
  for (size_t i = 0; i < n; i++) {
    lean_array_set_core(result, i, fromTorchTensor(unbinded[i]));
  }
  return result;
}

// Convert tensor to Long (int64) dtype
lean_object* lean_torch_to_long(lean_obj_arg s, b_lean_obj_arg input) {
  lean_dec(s);
  auto input_ = borrowTensor(input);
  auto result_ = input_.to(torch::kLong);
  return fromTorchTensor(result_);
}

// Backward pass that only requires the loss (assumes gradient of 1.0)
lean_object* lean_torch_backward_loss(b_lean_obj_arg loss) {
  return tyr::checkedIoCall("backwardLoss", [&] {
  try {
    auto loss_ = borrowTensor(loss);
    loss_.backward({}, /*retain_graph=*/true);
    return lean_io_result_mk_ok(lean_box(0));
  } catch (const c10::Error& e) {
    return mkC10IoError("Backward loss failed", e);
  } catch (const std::exception& e) {
    return mkStdIoError("Backward loss failed", e);
  }
  });
}

// Shape-aware matmul for 3D @ 2D: [batch, seq, k] @ [k, n] -> [batch, seq, n]
lean_object* lean_torch_matmul3d_2d(
  uint64_t /*batch*/,
  uint64_t /*seq*/,
  uint64_t /*k*/,
  uint64_t /*n*/,
  b_lean_obj_arg a,
  b_lean_obj_arg b
) {
  auto a_ = borrowTensor(a);
  auto b_ = borrowTensor(b);
  auto result_ = torch::matmul(a_, b_);
  return fromTorchTensor(result_);
}

// Transpose 2D tensor: [m, n] -> [n, m]
lean_object* lean_torch_transpose_2d(
  uint64_t /*m*/,
  uint64_t /*n*/,
  b_lean_obj_arg input
) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::transpose(input_, 0, 1);
  return fromTorchTensor(result_);
}

// Transpose 3D tensor dims 1 and 2: [a, b, c] -> [a, c, b]
lean_object* lean_torch_transpose3d_12(
  lean_obj_arg /*a*/,
  lean_obj_arg /*b*/,
  lean_obj_arg /*c*/,
  b_lean_obj_arg input
) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::transpose(input_, 1, 2);
  return fromTorchTensor(result_);
}

// Transpose for attention: [batch, seq, n_head, head_dim] -> [batch, n_head, seq, head_dim]
lean_object* lean_torch_transpose_for_attention(
  uint64_t /*batch*/,
  uint64_t /*seq*/,
  uint64_t /*n_head*/,
  uint64_t /*head_dim*/,
  b_lean_obj_arg input
) {
  auto input_ = borrowTensor(input);
  // [batch, seq, n_head, head_dim] -> [batch, n_head, seq, head_dim]
  auto result_ = input_.permute({0, 2, 1, 3}).contiguous();
  return fromTorchTensor(result_);
}

// Transpose from attention: [batch, n_head, seq, head_dim] -> [batch, seq, n_head, head_dim]
lean_object* lean_torch_transpose_from_attention(
  uint64_t /*batch*/,
  uint64_t /*n_head*/,
  uint64_t /*seq*/,
  uint64_t /*head_dim*/,
  b_lean_obj_arg input
) {
  auto input_ = borrowTensor(input);
  // [batch, n_head, seq, head_dim] -> [batch, seq, n_head, head_dim]
  auto result_ = input_.permute({0, 2, 1, 3}).contiguous();
  return fromTorchTensor(result_);
}

// Layer norm for 3D tensors with separate weight/bias shapes
lean_object* lean_torch_layer_norm_3d(
  uint64_t /*batch*/,
  uint64_t /*seq*/,
  uint64_t /*n*/,
  b_lean_obj_arg input,
  b_lean_obj_arg weight,
  b_lean_obj_arg bias,
  double eps
) {
  auto input_ = borrowTensor(input);
  auto weight_ = borrowTensor(weight);
  auto bias_ = borrowTensor(bias);

  // Get the last dimension size from the weight tensor
  int64_t n = weight_.size(0);
  std::vector<int64_t> normalized_shape = {n};

  auto output_ = torch::layer_norm(
    input_,
    c10::IntArrayRef(normalized_shape.data(), normalized_shape.size()),
    weight_,
    bias_,
    eps
  );


  return fromTorchTensor(output_);
}

// ============================================================================
// Fused layer_norm + activation operations
// These avoid intermediate tensor allocation between layer_norm and activation
// ============================================================================

// Fused layer_norm + GELU for transformer attention/MLP
lean_object* lean_torch_layer_norm_gelu(
  lean_obj_arg /*batch*/,
  lean_obj_arg /*seq*/,
  lean_obj_arg /*n*/,
  b_lean_obj_arg input,
  b_lean_obj_arg weight,
  b_lean_obj_arg bias,
  double eps
) {
  auto input_ = borrowTensor(input);
  auto weight_ = borrowTensor(weight);
  auto bias_ = borrowTensor(bias);

  int64_t n = weight_.size(0);
  std::vector<int64_t> normalized_shape = {n};

  auto normed = torch::layer_norm(
    input_,
    c10::IntArrayRef(normalized_shape.data(), normalized_shape.size()),
    weight_,
    bias_,
    eps
  );

  return fromTorchTensor(torch::gelu(normed));
}

// Fused layer_norm + ReLU
lean_object* lean_torch_layer_norm_relu(
  lean_obj_arg /*batch*/,
  lean_obj_arg /*seq*/,
  lean_obj_arg /*n*/,
  b_lean_obj_arg input,
  b_lean_obj_arg weight,
  b_lean_obj_arg bias,
  double eps
) {
  auto input_ = borrowTensor(input);
  auto weight_ = borrowTensor(weight);
  auto bias_ = borrowTensor(bias);

  int64_t n = weight_.size(0);
  std::vector<int64_t> normalized_shape = {n};

  auto normed = torch::layer_norm(
    input_,
    c10::IntArrayRef(normalized_shape.data(), normalized_shape.size()),
    weight_,
    bias_,
    eps
  );

  return fromTorchTensor(torch::relu(normed));
}

// Fused layer_norm + SiLU (for SwiGLU MLP)
lean_object* lean_torch_layer_norm_silu(
  lean_obj_arg /*batch*/,
  lean_obj_arg /*seq*/,
  lean_obj_arg /*n*/,
  b_lean_obj_arg input,
  b_lean_obj_arg weight,
  b_lean_obj_arg bias,
  double eps
) {
  auto input_ = borrowTensor(input);
  auto weight_ = borrowTensor(weight);
  auto bias_ = borrowTensor(bias);

  int64_t n = weight_.size(0);
  std::vector<int64_t> normalized_shape = {n};

  auto normed = torch::layer_norm(
    input_,
    c10::IntArrayRef(normalized_shape.data(), normalized_shape.size()),
    weight_,
    bias_,
    eps
  );

  return fromTorchTensor(torch::silu(normed));
}

// Cross entropy for 2D logits: [N, C] + targets [N] -> scalar
lean_object* lean_torch_cross_entropy_2d(
  uint64_t /*n*/,
  uint64_t /*c*/,
  b_lean_obj_arg logits,
  b_lean_obj_arg targets
) {
  auto logits_ = borrowTensor(logits);
  auto targets_ = borrowTensor(targets);
  auto result_ = torch::nn::functional::cross_entropy(logits_, targets_);
  return fromTorchTensor(result_);
}

// Scaled dot-product attention for 4D tensors
lean_object* lean_torch_cross_entropy_2d_checked(uint64_t n, uint64_t c,
    b_lean_obj_arg logits, b_lean_obj_arg targets) {
  return tyr::checkedCall("crossEntropy", [&] {
    validateCrossEntropyTargets(logits, targets);
    return lean_torch_cross_entropy_2d(n, c, logits, targets);
  });
}

lean_object* lean_torch_sdpa_4d(
  uint64_t /*batch*/,
  uint64_t /*n_head*/,
  uint64_t /*seq*/,
  uint64_t /*head_dim*/,
  b_lean_obj_arg query,
  b_lean_obj_arg key,
  b_lean_obj_arg value,
  double dropout_p,
  uint8_t is_causal
) {
  auto query_ = borrowTensor(query);
  auto key_ = borrowTensor(key);
  auto value_ = borrowTensor(value);

  auto result_ = torch::scaled_dot_product_attention(
    query_,
    key_,
    value_,
    c10::nullopt,  // attn_mask
    dropout_p,
    is_causal
  );


  return fromTorchTensor(result_);
}

// Fused SDPA with an explicit additive float bias [batch, n_head, seq, seq].
// Replaces the manual matmul -> bias add -> masked_fill -> softmax -> matmul
// pipeline (and its un-fused backward) for the molecule transformers.
lean_object* lean_torch_sdpa_4d_bias(
  uint64_t /*batch*/,
  uint64_t /*n_head*/,
  uint64_t /*seq*/,
  uint64_t /*head_dim*/,
  b_lean_obj_arg query,
  b_lean_obj_arg key,
  b_lean_obj_arg value,
  b_lean_obj_arg bias,
  double dropout_p,
  uint8_t is_causal
) {
  auto query_ = borrowTensor(query);
  auto key_ = borrowTensor(key);
  auto value_ = borrowTensor(value);
  auto bias_ = borrowTensor(bias);

  try {
    auto result_ = torch::scaled_dot_product_attention(
      query_,
      key_,
      value_,
      bias_,  // attn_mask: additive float bias
      dropout_p,
      is_causal
    );
    return fromTorchTensor(result_);
  } catch (const c10::Error& e) {
    std::fprintf(stderr, "[tyr] sdpa_4d_bias libtorch error:\n%s\n", e.what());
    std::fflush(stderr);
    std::abort();
  }
}

// Whole-tensor copy with matching shape/dtype and destination placement.
// Explicit optimizer/state value update. The destination is consumed and the
// updated tensor is returned; aliases observe the previous value.
lean_object* lean_torch_copy_inplace(
  lean_obj_arg s,
  lean_obj_arg dst,
  b_lean_obj_arg src
) {
  return tyr::checkedIoCall("copyInplace", [&] {
  lean_dec(s);
  OwnedLeanObject destination_owner(dst, lean_dec);
  try {
    torch::NoGradGuard guard;
    auto source = borrowTensor(src);
    auto destination = takeTensor(destination_owner.release());
    checkCopyMetadata(destination, source);
    auto result = tensorForNoGradValueUpdate(std::move(destination));
    result.copy_(source);
    return lean_io_result_mk_ok(fromTorchTensor(std::move(result)));
  } catch (const c10::Error& e) {
    return mkC10IoError("copyInplace", e);
  } catch (const std::exception& e) {
    return mkStdIoError("copyInplace", e);
  }
  });
}

// No-op that forces its tensor argument at the FFI boundary. Used to force
// lazy Lean staging computations (static buffers, in-place copies) to run
// eagerly *before* a CUDA graph capture/replay instead of inside it.
lean_object* lean_torch_touch(lean_obj_arg s, b_lean_obj_arg /*t*/) {
  return tyr::checkedIoCall("touch", [&] {
  lean_dec(s);
  return lean_io_result_mk_ok(lean_box(0));
  });
}

// Tensor × 0-dim tensor (broadcast scalar): lets a captured CUDA graph take a
// scalar input (e.g. the learning rate) through a static buffer instead of a
// baked-in Float.
lean_object* lean_torch_mul_tensor_scalar(
  lean_obj_arg s,
  b_lean_obj_arg input,
  b_lean_obj_arg scalar
) {
  lean_dec(s);
  auto input_ = borrowTensor(input);
  auto scalar_ = borrowTensor(scalar);
  return fromTorchTensor(input_ * scalar_);
}

// --- Generic CUDA graph capture/replay -------------------------------------
// A single process-wide graph slot is enough for the training-step use case:
// capture a static-shaped region once, replay it with fresh data copied into
// the same input buffers. Errors surface as Lean IO errors so callers can
// fall back to eager execution. CPU-only libtorch builds (which lack the CUDA
// cmake macros header) get stubs that report unavailability.

#if __has_include(<ATen/cuda/CUDAGraph.h>) && __has_include(<c10/cuda/impl/cuda_cmake_macros.h>)
#define TYR_WITH_CUDA_GRAPH 1
#include <ATen/cuda/CUDAGraph.h>
#else
#define TYR_WITH_CUDA_GRAPH 0
#endif

#if TYR_WITH_CUDA_GRAPH
#include <c10/cuda/CUDAStream.h>
namespace {
std::unique_ptr<at::cuda::CUDAGraph> g_cuda_graph;
}
#endif

namespace {
lean_object* tyrNoCudaGraph() {
  return lean_io_result_mk_error(lean_mk_io_user_error(
    lean_mk_string("CUDA graphs not available in this libtorch build")));
}

lean_object* lean_torch_cuda_graph_capture_begin(lean_object* w) {
#if TYR_WITH_CUDA_GRAPH
  try {
    // Capture is illegal on the legacy default stream; mirror
    // torch.cuda.graph's side-stream dance: sync, switch to a pool stream,
    // capture there, and let replay run on the caller's current stream.
    c10::cuda::getCurrentCUDAStream().synchronize();
    c10::cuda::setCurrentCUDAStream(c10::cuda::getStreamFromPool(false));
    g_cuda_graph = std::make_unique<at::cuda::CUDAGraph>();
    g_cuda_graph->capture_begin();
    return lean_io_result_mk_ok(lean_box(0));
  } catch (const std::exception& e) {
    g_cuda_graph.reset();
    c10::cuda::setCurrentCUDAStream(c10::cuda::getDefaultCUDAStream());
    return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string(("cuda graph capture_begin failed: " + std::string(e.what())).c_str())));
  }
#else
  return tyrNoCudaGraph();
#endif
}

lean_object* lean_torch_cuda_graph_capture_end(lean_object* w) {
#if TYR_WITH_CUDA_GRAPH
  try {
    g_cuda_graph->capture_end();
    c10::cuda::setCurrentCUDAStream(c10::cuda::getDefaultCUDAStream());
    return lean_io_result_mk_ok(lean_box(0));
  } catch (const std::exception& e) {
    g_cuda_graph.reset();
    c10::cuda::setCurrentCUDAStream(c10::cuda::getDefaultCUDAStream());
    return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string(("cuda graph capture_end failed: " + std::string(e.what())).c_str())));
  }
#else
  return tyrNoCudaGraph();
#endif
}

lean_object* lean_torch_cuda_graph_replay(lean_object* w) {
#if TYR_WITH_CUDA_GRAPH
  try {
    g_cuda_graph->replay();
    return lean_io_result_mk_ok(lean_box(0));
  } catch (const std::exception& e) {
    return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string(("cuda graph replay failed: " + std::string(e.what())).c_str())));
  }
#else
  return tyrNoCudaGraph();
#endif
}

lean_object* lean_torch_cuda_graph_reset(lean_object* w) {
#if TYR_WITH_CUDA_GRAPH
  g_cuda_graph.reset();
#endif
  return lean_io_result_mk_ok(lean_box(0));
}

} // anonymous namespace

// Save tensor to a binary file
lean_object* lean_torch_save_tensor(lean_obj_arg s, b_lean_obj_arg tensor, b_lean_obj_arg path_obj) {
  return tyr::checkedIoCall("saveTensor", [&] {
  lean_dec(s);
  auto tensor_ = borrowTensor(tensor);
  const char* path = lean_string_cstr(path_obj);

  try {
    // Use torch::save with a vector containing our tensor
    std::vector<torch::Tensor> tensors = {tensor_};
    torch::save(tensors, path);
    return lean_io_result_mk_ok(lean_box(0));
  } catch (const c10::Error& e) {
    return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string(("Failed to save tensor: " + std::string(e.what())).c_str())));
  }
  });
}

// Load tensor from a binary file with expected shape
lean_object* lean_torch_load_tensor(lean_obj_arg shape, b_lean_obj_arg path_obj) {
  std::unique_ptr<lean_object, void (*)(lean_object*)> shape_owner(shape, lean_dec);
  const char* path = lean_string_cstr(path_obj);

  try {
    std::vector<torch::Tensor> tensors;
    torch::load(tensors, path);

    if (tensors.empty()) {
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string("No tensors found in file")));
    }

    // Return the first tensor (reshape to expected shape if needed)
    auto tensor = tensors[0];
    auto expected_shape = getShape(shape);

    // Verify shape matches (or reshape if possible)
    if (tensor.numel() != std::accumulate(expected_shape.begin(), expected_shape.end(),
                                           1LL, std::multiplies<int64_t>())) {
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string("Tensor size mismatch with expected shape")));
    }

    auto result = tensor.reshape(expected_shape);
    return lean_io_result_mk_ok(fromTorchTensor(result));
  } catch (const c10::Error& e) {
    return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string(("Failed to load tensor: " + std::string(e.what())).c_str())));
  } catch (const std::exception& e) {
    return mkStdIoError("Failed to load tensor", e);
  }
}

lean_object* lean_torch_load_tensor_exact(lean_obj_arg shape, b_lean_obj_arg path_obj) {
  return tyr::checkedIoCall("loadTensorExact", [&] {
  std::unique_ptr<lean_object, void (*)(lean_object*)> shape_owner(shape, lean_dec);
  try {
    const auto expected_shape = getShape(shape);
    std::vector<torch::Tensor> tensors;
    torch::load(tensors, lean_string_cstr(path_obj));
    if (tensors.empty()) {
      return mkIoUserError("No tensors found in file");
    }
    if (tensors[0].sizes().vec() != expected_shape) {
      return mkIoUserError("Tensor shape mismatch with expected shape");
    }
    return lean_io_result_mk_ok(fromTorchTensor(tensors[0]));
  } catch (const c10::Error& e) {
    return mkC10IoError("Failed to load tensor", e);
  } catch (const std::exception& e) {
    return mkStdIoError("Failed to load tensor", e);
  }
  });
}

// Check if a file exists
lean_object* lean_torch_file_exists(b_lean_obj_arg path_obj, lean_object* w) {
  const char* path = lean_string_cstr(path_obj);
  std::ifstream file(path);
  return lean_io_result_mk_ok(lean_box(file.good() ? 1 : 0));
}

// Save image tensor as PPM file
// Expects tensor of shape [batch, 3, height, width] with values in [-1, 1]
lean_object* lean_torch_save_ppm(
  lean_obj_arg shape,
  b_lean_obj_arg tensor_obj,
  b_lean_obj_arg path_obj
) {
  return tyr::checkedIoCall("savePPM", [&] {
  const char* path = lean_string_cstr(path_obj);
  lean_dec(shape);

  try {
    auto tensor = borrowTensor(tensor_obj);

    // Get dimensions
    int64_t channels = tensor.size(1);
    int64_t height = tensor.size(2);
    int64_t width = tensor.size(3);

    if (channels != 3) {
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string("PPM requires 3 channels (RGB)")));
    }

    // Take first image, convert from [-1, 1] to [0, 255]
    auto img = tensor[0];  // [3, height, width]
    img = ((img + 1.0) * 127.5).clamp(0, 255).to(torch::kUInt8);

    // Transpose to [height, width, 3]
    img = img.permute({1, 2, 0}).contiguous();

    // Write PPM file
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string(("Failed to open file for writing: " + std::string(path)).c_str())));
    }

    // PPM header
    file << "P6\n" << width << " " << height << "\n255\n";

    // Write pixel data
    auto data_ptr = img.data_ptr<uint8_t>();
    file.write(reinterpret_cast<const char*>(data_ptr), height * width * 3);

    file.close();
    return lean_io_result_mk_ok(lean_box(0));
  } catch (const std::exception& e) {
    return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string(("Failed to save PPM: " + std::string(e.what())).c_str())));
  }
  });
}

// Save mono waveform as 16-bit PCM WAV.
// Tensor is flattened in row-major order and clamped to [-1, 1].
static constexpr uint64_t kWavHeaderSizeBytes = 44ULL;
static constexpr uint64_t kWavRiffChunkOverheadBytes = 36ULL;
static constexpr uint64_t kWavMaxDataSizeBytes =
  static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) - kWavRiffChunkOverheadBytes;

static bool wav_data_size_fits_riff(uint64_t data_size) {
  return data_size <= kWavMaxDataSizeBytes;
}

static bool write_wav_header(std::ostream& file, uint32_t sample_rate, uint64_t data_size64) {
  if (!wav_data_size_fits_riff(data_size64)) return false;

  const uint32_t data_size = static_cast<uint32_t>(data_size64);
  const uint32_t channels = 1;
  const uint32_t bits_per_sample = 16;
  const uint32_t byte_rate = sample_rate * channels * (bits_per_sample / 8);
  const uint16_t block_align = static_cast<uint16_t>(channels * (bits_per_sample / 8));
  const uint32_t riff_chunk_size = static_cast<uint32_t>(kWavRiffChunkOverheadBytes + data_size64);

  file.write("RIFF", 4);
  file.write(reinterpret_cast<const char*>(&riff_chunk_size), 4);
  file.write("WAVE", 4);
  file.write("fmt ", 4);
  uint32_t fmt_chunk_size = 16;
  uint16_t audio_format = 1;  // PCM
  uint16_t num_channels = static_cast<uint16_t>(channels);
  file.write(reinterpret_cast<const char*>(&fmt_chunk_size), 4);
  file.write(reinterpret_cast<const char*>(&audio_format), 2);
  file.write(reinterpret_cast<const char*>(&num_channels), 2);
  file.write(reinterpret_cast<const char*>(&sample_rate), 4);
  file.write(reinterpret_cast<const char*>(&byte_rate), 4);
  file.write(reinterpret_cast<const char*>(&block_align), 2);
  file.write(reinterpret_cast<const char*>(&bits_per_sample), 2);
  file.write("data", 4);
  file.write(reinterpret_cast<const char*>(&data_size), 4);
  return file.good();
}

static bool patch_wav_sizes(std::fstream& file, uint64_t data_size64) {
  if (!wav_data_size_fits_riff(data_size64)) return false;

  const uint32_t data_size = static_cast<uint32_t>(data_size64);
  const uint32_t riff_chunk_size = static_cast<uint32_t>(kWavRiffChunkOverheadBytes + data_size64);
  file.clear();
  file.seekp(4, std::ios::beg);
  if (!file.good()) return false;
  file.write(reinterpret_cast<const char*>(&riff_chunk_size), 4);
  file.seekp(40, std::ios::beg);
  if (!file.good()) return false;
  file.write(reinterpret_cast<const char*>(&data_size), 4);
  return file.good();
}

lean_object* lean_torch_save_wav(
  lean_obj_arg shape,
  b_lean_obj_arg tensor_obj,
  b_lean_obj_arg path_obj,
  uint64_t sample_rate
) {
  return tyr::checkedIoCall("saveWav", [&] {
  lean_dec(shape);
  const char* path = lean_string_cstr(path_obj);
  try {
    auto tensor = borrowTensor(tensor_obj);
    auto wav = tensor.detach().to(torch::kCPU).to(torch::kFloat32).contiguous().view({-1});
    int64_t n_samples = wav.numel();
    if (n_samples <= 0) {
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string("Cannot save WAV: empty tensor")));
    }

    const uint64_t data_size64 = static_cast<uint64_t>(n_samples) * sizeof(int16_t);
    if (!wav_data_size_fits_riff(data_size64)) {
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string("Cannot save WAV: data size exceeds RIFF 32-bit limit (~4GiB)")));
    }

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string(("Failed to open file for writing: " + std::string(path)).c_str())));
    }

    if (!write_wav_header(file, static_cast<uint32_t>(sample_rate), data_size64)) {
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string("Failed to write WAV header")));
    }

    const float* ptr = wav.data_ptr<float>();
    for (int64_t i = 0; i < n_samples; ++i) {
      float x = ptr[i];
      if (x > 1.0f) x = 1.0f;
      if (x < -1.0f) x = -1.0f;
      int16_t pcm = static_cast<int16_t>(std::lrintf(x * 32767.0f));
      file.write(reinterpret_cast<const char*>(&pcm), sizeof(int16_t));
    }
    file.close();
    return lean_io_result_mk_ok(lean_box(0));
  } catch (const std::exception& e) {
    return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string(("Failed to save WAV: " + std::string(e.what())).c_str())));
  }
  });
}

// Start a streaming mono WAV file with a placeholder header.
lean_object* lean_torch_wav_begin(
  b_lean_obj_arg path_obj,
  uint64_t sample_rate
) {
  return tyr::checkedIoCall("wavBegin", [&] {
  const char* path = lean_string_cstr(path_obj);
  try {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string(("Failed to open WAV for writing: " + std::string(path)).c_str())));
    }
    if (!write_wav_header(file, static_cast<uint32_t>(sample_rate), 0)) {
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string("Failed to write WAV stream header")));
    }
    file.close();
    return lean_io_result_mk_ok(lean_box(0));
  } catch (const std::exception& e) {
    return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string(("Failed to begin WAV stream: " + std::string(e.what())).c_str())));
  }
  });
}

// Append float waveform samples to an existing streaming WAV and patch sizes.
lean_object* lean_torch_wav_append(
  lean_obj_arg shape,
  b_lean_obj_arg tensor_obj,
  b_lean_obj_arg path_obj
) {
  return tyr::checkedIoCall("wavAppend", [&] {
  lean_dec(shape);
  const char* path = lean_string_cstr(path_obj);
  try {
    auto tensor = borrowTensor(tensor_obj);
    auto wav = tensor.detach().to(torch::kCPU).to(torch::kFloat32).contiguous().view({-1});
    int64_t n_samples = wav.numel();
    if (n_samples <= 0) {
      return lean_io_result_mk_ok(lean_box(0));
    }

    std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
    if (!file.is_open()) {
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string(("Failed to open WAV stream (call wavBegin first): " + std::string(path)).c_str())));
    }

    file.seekg(0, std::ios::end);
    std::streampos file_size = file.tellg();
    if (file_size == std::streampos(-1) ||
        file_size < static_cast<std::streampos>(kWavHeaderSizeBytes)) {
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string("WAV stream is invalid (too small for header)")));
    }

    const uint64_t existing_data_bytes64 = static_cast<uint64_t>(file_size) - kWavHeaderSizeBytes;
    const uint64_t append_bytes64 = static_cast<uint64_t>(n_samples) * sizeof(int16_t);
    if (!wav_data_size_fits_riff(existing_data_bytes64) ||
        append_bytes64 > kWavMaxDataSizeBytes ||
        existing_data_bytes64 > (kWavMaxDataSizeBytes - append_bytes64)) {
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string("WAV stream exceeds RIFF 32-bit data size limit (~4GiB)")));
    }

    file.clear();
    file.seekp(0, std::ios::end);
    const float* ptr = wav.data_ptr<float>();
    for (int64_t i = 0; i < n_samples; ++i) {
      float x = ptr[i];
      if (x > 1.0f) x = 1.0f;
      if (x < -1.0f) x = -1.0f;
      int16_t pcm = static_cast<int16_t>(std::lrintf(x * 32767.0f));
      file.write(reinterpret_cast<const char*>(&pcm), sizeof(int16_t));
    }

    if (!file.good()) {
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string("Failed while appending WAV stream data")));
    }

    const uint64_t data_bytes64 = existing_data_bytes64 + append_bytes64;
    if (!patch_wav_sizes(file, data_bytes64)) {
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string("Failed to patch WAV header sizes")));
    }
    file.close();
    return lean_io_result_mk_ok(lean_box(0));
  } catch (const std::exception& e) {
    return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string(("Failed to append WAV stream: " + std::string(e.what())).c_str())));
  }
  });
}

// Finalize a streaming WAV file by recomputing and patching header sizes.
lean_object* lean_torch_wav_finalize(
  b_lean_obj_arg path_obj
) {
  return tyr::checkedIoCall("wavFinalize", [&] {
  const char* path = lean_string_cstr(path_obj);
  try {
    std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
    if (!file.is_open()) {
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string(("Failed to open WAV for finalize: " + std::string(path)).c_str())));
    }
    file.seekg(0, std::ios::end);
    std::streampos file_size = file.tellg();
    if (file_size == std::streampos(-1) ||
        file_size < static_cast<std::streampos>(kWavHeaderSizeBytes)) {
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string("WAV file is invalid (too small for header)")));
    }
    uint64_t data_bytes64 = static_cast<uint64_t>(file_size) - kWavHeaderSizeBytes;
    if (!wav_data_size_fits_riff(data_bytes64)) {
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string("WAV file exceeds RIFF 32-bit data size limit (~4GiB)")));
    }
    if (!patch_wav_sizes(file, data_bytes64)) {
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string("Failed to finalize WAV header sizes")));
    }
    file.close();
    return lean_io_result_mk_ok(lean_box(0));
  } catch (const std::exception& e) {
    return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string(("Failed to finalize WAV: " + std::string(e.what())).c_str())));
  }
  });
}

// ===========================================================================
// SafeTensors loading
// ===========================================================================

// SafeTensors format: 8-byte header size (LE) + JSON header + tensor data.
struct SafeTensorInfo {
  std::string dtype;
  std::vector<int64_t> shape;
  int64_t data_offset;
  int64_t data_size;
};

struct SaveTensorEntry {
  std::string name;
  std::string dtype_tag;
  std::vector<int64_t> shape;
  torch::Tensor tensor_cpu;
  int64_t data_size = 0;
  int64_t data_start = 0;
  int64_t data_end = 0;
};

static std::string jsonEscape(const std::string& s) {
  std::ostringstream out;
  for (char c : s) {
    switch (c) {
      case '\"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          out << "\\u00";
          static const char* hex = "0123456789abcdef";
          out << hex[(c >> 4) & 0xF] << hex[c & 0xF];
        } else {
          out << c;
        }
    }
  }
  return out.str();
}

static std::optional<torch::ScalarType> parseDtype(const std::string& dtype) {
  return tyr::metadata::parseDtype(dtype);
}

// Supported storage types have matching dtypeSize and writer tags below.
static std::optional<torch::ScalarType> parseSafeTensorDtype(const std::string& dtype) {
  if (dtype == "F32" || dtype == "float32") return torch::kFloat32;
  if (dtype == "F64" || dtype == "float64") return torch::kFloat64;
  if (dtype == "F16" || dtype == "float16") return torch::kFloat16;
  if (dtype == "BF16" || dtype == "bfloat16") return torch::kBFloat16;
  if (dtype == "I64" || dtype == "int64") return torch::kInt64;
  if (dtype == "I32" || dtype == "int32") return torch::kInt32;
  if (dtype == "I16" || dtype == "int16") return torch::kInt16;
  if (dtype == "I8" || dtype == "int8") return torch::kInt8;
  if (dtype == "U8" || dtype == "uint8") return torch::kUInt8;
  if (dtype == "BOOL" || dtype == "bool") return torch::kBool;
  // FP8 types (PyTorch 2.1+)
  if (dtype == "F8_E4M3" || dtype == "float8_e4m3fn") return torch::kFloat8_e4m3fn;
  if (dtype == "F8_E5M2" || dtype == "float8_e5m2") return torch::kFloat8_e5m2;
  return std::nullopt;
}

extern "C" LEAN_EXPORT lean_object* lean_torch_cast_dtype(
    lean_obj_arg shape, b_lean_obj_arg tensor, b_lean_obj_arg tag) {
  lean_dec(shape);
  const auto dtype = parseDtype(lean_string_cstr(tag));
  if (!dtype) {
    throw std::invalid_argument("Unsupported tensor dtype tag");
  }
  return fromTorchTensor(borrowTensor(tensor).to(*dtype));
}

extern "C" LEAN_EXPORT lean_object* lean_torch_cast_dtype_checked(
    lean_obj_arg shape, b_lean_obj_arg tensor, b_lean_obj_arg tag) {
  return tyr::checkedCall("castDType", [&] {
    OwnedLeanObject shape_owner(shape, lean_dec);
    auto source = borrowTensor(tensor);
    TORCH_CHECK(source.sizes().vec() == getShape(shape), "source tensor shape mismatch");
    const auto dtype = parseDtype(lean_string_cstr(tag));
    TORCH_CHECK(dtype.has_value(), "Unsupported target dtype tag: ", lean_string_cstr(tag));
    return fromTorchTensor(source.to(*dtype));
  });
}

extern "C" LEAN_EXPORT lean_object* lean_torch_convert_tensor(
    lean_obj_arg shape, b_lean_obj_arg tensor, b_lean_obj_arg device,
    b_lean_obj_arg dtype_option) {
  return tyr::checkedIoCall("convertIO", [&] {
  std::unique_ptr<lean_object, void (*)(lean_object*)> shape_owner(shape, lean_dec);
  try {
    const auto expected_shape = getShape(shape);
    auto source = borrowTensor(tensor);
    TORCH_CHECK(source.sizes().vec() == expected_shape,
                "source tensor shape mismatch");

    auto target_dtype = source.scalar_type();
    if (!lean_is_scalar(dtype_option)) {
      const auto* tag = lean_string_cstr(lean_ctor_get(dtype_option, 0));
      const auto parsed_dtype = parseDtype(tag);
      TORCH_CHECK(parsed_dtype.has_value(), "Unsupported target dtype tag: ", tag);
      target_dtype = *parsed_dtype;
    }
    const auto target_device = getDevice(device);

    // One checked FFI operation chooses a backend-compatible conversion order.
    // PyTorch's MPS-to-CPU copy casts on MPS, so a combined request for CPU
    // Float64 would still require an unsupported MPS Float64 intermediate.
    // Transfer first in this direction; CPU-to-MPS already casts on CPU.
    auto converted = source.is_mps() && target_device.is_cpu() &&
                             source.scalar_type() != target_dtype
        ? source.to(target_device).to(target_dtype)
        : source.to(target_device, target_dtype);
    return lean_io_result_mk_ok(fromTorchTensor(converted));
  } catch (const c10::Error& e) {
    return mkC10IoError("convertIO", e);
  } catch (const std::exception& e) {
    return mkStdIoError("convertIO", e);
  }
  });
}

static std::optional<size_t> findMatchingDelim(
    const std::string& s, size_t open_pos, char open_ch, char close_ch) {
  if (open_pos >= s.size() || s[open_pos] != open_ch) return std::nullopt;
  int depth = 0;
  bool in_string = false;
  bool escaped = false;
  for (size_t i = open_pos; i < s.size(); i++) {
    char c = s[i];
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }
    if (c == '"') {
      in_string = true;
      continue;
    }
    if (c == open_ch) {
      depth++;
    } else if (c == close_ch) {
      depth--;
      if (depth == 0) return i;
      if (depth < 0) return std::nullopt;
    }
  }
  return std::nullopt;
}

static std::string trimAscii(std::string s) {
  const char* ws = " \t\r\n";
  const size_t start = s.find_first_not_of(ws);
  if (start == std::string::npos) return "";
  const size_t end = s.find_last_not_of(ws);
  return s.substr(start, end - start + 1);
}

static bool parseI64Strict(const std::string& s, int64_t& out) {
  try {
    size_t idx = 0;
    out = std::stoll(s, &idx);
    return idx == s.size();
  } catch (const std::exception&) {
    return false;
  }
}

static std::optional<uint64_t> checkedNumel(const std::vector<int64_t>& shape) {
  uint64_t numel = 1;
  for (int64_t d : shape) {
    if (d < 0) return std::nullopt;
    if (d == 0) {
      numel = 0;
      continue;
    }
    const uint64_t du = static_cast<uint64_t>(d);
    if (numel != 0 && numel > (std::numeric_limits<uint64_t>::max() / du)) {
      return std::nullopt;
    }
    numel *= du;
  }
  return numel;
}

static size_t dtypeSize(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kFloat32: return 4;
    case torch::kFloat64: return 8;
    case torch::kFloat16: return 2;
    case torch::kBFloat16: return 2;
    case torch::kFloat8_e4m3fn: return 1;
    case torch::kFloat8_e5m2: return 1;
    case torch::kInt64: return 8;
    case torch::kInt32: return 4;
    case torch::kInt16: return 2;
    case torch::kInt8: return 1;
    case torch::kUInt8: return 1;
    case torch::kBool: return 1;
    default: return 4;
  }
}

static std::string dtypeToSafeTensorTag(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kFloat64: return "F64";
    case torch::kFloat32: return "F32";
    case torch::kFloat16: return "F16";
    case torch::kBFloat16: return "BF16";
    case torch::kFloat8_e4m3fn: return "F8_E4M3";
    case torch::kFloat8_e5m2: return "F8_E5M2";
    case torch::kInt64: return "I64";
    case torch::kInt32: return "I32";
    case torch::kInt16: return "I16";
    case torch::kInt8: return "I8";
    case torch::kUInt8: return "U8";
    case torch::kBool: return "BOOL";
    default: return "";
  }
}

static std::optional<std::string> prepareSaveTensorEntry(
    const std::string& name,
    const torch::Tensor& t_in,
    SaveTensorEntry& out
) {
  if (name.empty()) {
    return std::string("SafeTensors save error: tensor name cannot be empty.");
  }
  auto t = t_in.detach().to(torch::kCPU).contiguous();
  std::string dtype_tag = dtypeToSafeTensorTag(t.scalar_type());
  if (dtype_tag.empty()) {
    return std::string("SafeTensors save error: unsupported dtype for tensor '") + name + "'.";
  }

  out.name = name;
  out.dtype_tag = dtype_tag;
  out.tensor_cpu = t;
  out.shape.clear();
  for (int64_t i = 0; i < t.dim(); i++) {
    out.shape.push_back(t.size(i));
  }
  out.data_size = static_cast<int64_t>(t.numel()) * static_cast<int64_t>(dtypeSize(t.scalar_type()));
  out.data_start = 0;
  out.data_end = 0;
  return std::nullopt;
}

static std::optional<std::string> writeSafeTensorsFile(
    const std::string& path,
    std::vector<SaveTensorEntry>& entries,
    const std::vector<std::pair<std::string, std::string>>& metadata
) {
  if (entries.empty()) {
    return std::string("SafeTensors save error: no tensors provided.");
  }

  std::unordered_set<std::string> seen_names;
  int64_t offset = 0;
  for (auto& e : entries) {
    if (!seen_names.insert(e.name).second) {
      return std::string("SafeTensors save error: duplicate tensor name '") + e.name + "'.";
    }
    e.data_start = offset;
    e.data_end = offset + e.data_size;
    offset = e.data_end;
  }

  std::ostringstream header;
  header << "{";
  bool need_comma = false;
  if (!metadata.empty()) {
    header << "\"__metadata__\":{";
    for (size_t i = 0; i < metadata.size(); i++) {
      if (i > 0) header << ",";
      header << "\"" << jsonEscape(metadata[i].first) << "\":\"" << jsonEscape(metadata[i].second) << "\"";
    }
    header << "}";
    need_comma = true;
  }

  for (size_t i = 0; i < entries.size(); i++) {
    const auto& e = entries[i];
    if (need_comma || i > 0) header << ",";
    header << "\"" << jsonEscape(e.name) << "\":{";
    header << "\"dtype\":\"" << e.dtype_tag << "\",";
    header << "\"shape\":[";
    for (size_t j = 0; j < e.shape.size(); j++) {
      if (j > 0) header << ",";
      header << e.shape[j];
    }
    header << "],";
    header << "\"data_offsets\":[" << e.data_start << "," << e.data_end << "]";
    header << "}";
    need_comma = true;
  }
  header << "}";

  std::string header_json = header.str();
  uint64_t header_size = static_cast<uint64_t>(header_json.size());

  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file.is_open()) {
    return std::string("Failed to open output safetensors file: ") + path;
  }

  file.write(reinterpret_cast<const char*>(&header_size), sizeof(uint64_t));
  file.write(header_json.data(), static_cast<std::streamsize>(header_json.size()));
  for (const auto& e : entries) {
    file.write(
      reinterpret_cast<const char*>(e.tensor_cpu.data_ptr()),
      static_cast<std::streamsize>(e.data_size)
    );
  }
  file.close();
  return std::nullopt;
}

// Save a single tensor to SafeTensors file under the given tensor name.
lean_object* lean_torch_safetensors_save(
  lean_obj_arg shape,
  b_lean_obj_arg path_obj,
  b_lean_obj_arg name_obj,
  b_lean_obj_arg tensor_obj
) {
  return tyr::checkedIoCall("saveSafeTensor", [&] {
  lean_dec(shape);
  const char* path = lean_string_cstr(path_obj);
  const char* tensor_name = lean_string_cstr(name_obj);

  try {
    SaveTensorEntry entry;
    auto prep_err = prepareSaveTensorEntry(std::string(tensor_name), borrowTensor(tensor_obj), entry);
    if (prep_err.has_value()) {
      return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(prep_err->c_str())));
    }
    std::vector<SaveTensorEntry> entries = {std::move(entry)};
    std::vector<std::pair<std::string, std::string>> metadata;
    auto err = writeSafeTensorsFile(std::string(path), entries, metadata);
    if (err.has_value()) {
      return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(err->c_str())));
    }
    return lean_io_result_mk_ok(lean_box(0));
  } catch (const std::exception& e) {
    return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string(("SafeTensors save error: " + std::string(e.what())).c_str())));
  }
  });
}

// Save multiple tensors to one SafeTensors file with optional metadata.
lean_object* lean_torch_safetensors_save_many(
  b_lean_obj_arg path_obj,
  b_lean_obj_arg names_obj,
  b_lean_obj_arg tensors_obj,
  b_lean_obj_arg meta_keys_obj,
  b_lean_obj_arg meta_vals_obj,
  lean_object* /*w*/
) {
  const char* path = lean_string_cstr(path_obj);
  try {
    size_t names_n = lean_array_size(names_obj);
    size_t tensors_n = lean_array_size(tensors_obj);
    if (names_n != tensors_n) {
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string("SafeTensors save_many error: names and tensors length mismatch.")));
    }

    std::vector<SaveTensorEntry> entries;
    entries.reserve(names_n);
    for (size_t i = 0; i < names_n; i++) {
      lean_object* name_obj = lean_array_get_core(names_obj, i);
      lean_object* tensor_obj = lean_array_get_core(tensors_obj, i);
      const char* name_c = lean_string_cstr(name_obj);
      SaveTensorEntry entry;
      auto prep_err = prepareSaveTensorEntry(std::string(name_c), borrowTensor(tensor_obj), entry);
      if (prep_err.has_value()) {
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(prep_err->c_str())));
      }
      entries.push_back(std::move(entry));
    }

    size_t meta_k_n = lean_array_size(meta_keys_obj);
    size_t meta_v_n = lean_array_size(meta_vals_obj);
    if (meta_k_n != meta_v_n) {
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string("SafeTensors save_many error: metadata keys/values length mismatch.")));
    }
    std::vector<std::pair<std::string, std::string>> metadata;
    metadata.reserve(meta_k_n);
    for (size_t i = 0; i < meta_k_n; i++) {
      const char* k = lean_string_cstr(lean_array_get_core(meta_keys_obj, i));
      const char* v = lean_string_cstr(lean_array_get_core(meta_vals_obj, i));
      metadata.emplace_back(std::string(k), std::string(v));
    }

    auto err = writeSafeTensorsFile(std::string(path), entries, metadata);
    if (err.has_value()) {
      return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(err->c_str())));
    }
    return lean_io_result_mk_ok(lean_box(0));
  } catch (const std::exception& e) {
    return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string(("SafeTensors save_many error: " + std::string(e.what())).c_str())));
  }
}

class SafeTensorsFile {
public:
  struct Entry {
    std::string dtype_name;
    torch::ScalarType dtype;
    std::vector<int64_t> shape;
    uint64_t data_start = 0;
    uint64_t data_end = 0;
  };

  std::string path;
  uint64_t file_size = 0;
  std::ifstream file;
  uint64_t header_size;
  std::string header_json;
  std::unordered_map<std::string, Entry> entries;
  mutable std::mutex io_mutex;

  static void skipWhitespace(const std::string& s, size_t& pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\r' || s[pos] == '\n')) {
      pos++;
    }
  }

  static std::string parseJsonString(const std::string& s, size_t& pos) {
    if (pos >= s.size() || s[pos] != '"') {
      throw std::runtime_error("SafeTensors load error: malformed JSON string.");
    }
    pos++;
    std::string out;
    bool escaped = false;
    while (pos < s.size()) {
      char c = s[pos++];
      if (escaped) {
        switch (c) {
          case '"': out.push_back('"'); break;
          case '\\': out.push_back('\\'); break;
          case '/': out.push_back('/'); break;
          case 'b': out.push_back('\b'); break;
          case 'f': out.push_back('\f'); break;
          case 'n': out.push_back('\n'); break;
          case 'r': out.push_back('\r'); break;
          case 't': out.push_back('\t'); break;
          case 'u':
            if (pos + 4 > s.size()) {
              throw std::runtime_error("SafeTensors load error: truncated unicode escape.");
            }
            pos += 4;
            break;
          default:
            throw std::runtime_error("SafeTensors load error: invalid escape sequence.");
        }
        escaped = false;
        continue;
      }
      if (c == '\\') {
        escaped = true;
        continue;
      }
      if (c == '"') {
        return out;
      }
      out.push_back(c);
    }
    throw std::runtime_error("SafeTensors load error: unterminated JSON string.");
  }

  static Entry parseEntryObject(const std::string& obj) {
    Entry e;

    size_t dtype_key = obj.find("\"dtype\"");
    if (dtype_key == std::string::npos) {
      throw std::runtime_error("SafeTensors load error: missing dtype field.");
    }
    size_t dtype_colon = obj.find(':', dtype_key);
    if (dtype_colon == std::string::npos) {
      throw std::runtime_error("SafeTensors load error: malformed dtype field.");
    }
    size_t dtype_quote_start = obj.find('"', dtype_colon + 1);
    if (dtype_quote_start == std::string::npos) {
      throw std::runtime_error("SafeTensors load error: malformed dtype value.");
    }
    size_t dtype_quote_end = obj.find('"', dtype_quote_start + 1);
    if (dtype_quote_end == std::string::npos) {
      throw std::runtime_error("SafeTensors load error: malformed dtype value.");
    }
    e.dtype_name = obj.substr(dtype_quote_start + 1, dtype_quote_end - dtype_quote_start - 1);
    auto dtype_opt = parseSafeTensorDtype(e.dtype_name);
    if (!dtype_opt.has_value()) {
      throw std::runtime_error("SafeTensors load error: unsupported dtype '" + e.dtype_name + "'.");
    }
    e.dtype = *dtype_opt;

    size_t shape_key = obj.find("\"shape\"");
    if (shape_key == std::string::npos) {
      throw std::runtime_error("SafeTensors load error: missing shape field.");
    }
    size_t shape_start = obj.find('[', shape_key);
    if (shape_start == std::string::npos) {
      throw std::runtime_error("SafeTensors load error: malformed shape field.");
    }
    auto shape_end_opt = findMatchingDelim(obj, shape_start, '[', ']');
    if (!shape_end_opt.has_value()) {
      throw std::runtime_error("SafeTensors load error: malformed shape array.");
    }
    std::string shape_str = obj.substr(shape_start + 1, *shape_end_opt - shape_start - 1);
    if (!trimAscii(shape_str).empty()) {
      std::istringstream shape_stream(shape_str);
      std::string dim;
      while (std::getline(shape_stream, dim, ',')) {
        dim = trimAscii(dim);
        int64_t d = 0;
        if (dim.empty() || !parseI64Strict(dim, d) || d < 0) {
          throw std::runtime_error("SafeTensors load error: invalid shape entry.");
        }
        e.shape.push_back(d);
      }
    }

    size_t offsets_key = obj.find("\"data_offsets\"");
    if (offsets_key == std::string::npos) {
      throw std::runtime_error("SafeTensors load error: missing data_offsets field.");
    }
    size_t offsets_start = obj.find('[', offsets_key);
    if (offsets_start == std::string::npos) {
      throw std::runtime_error("SafeTensors load error: malformed data_offsets field.");
    }
    auto offsets_end_opt = findMatchingDelim(obj, offsets_start, '[', ']');
    if (!offsets_end_opt.has_value()) {
      throw std::runtime_error("SafeTensors load error: malformed data_offsets array.");
    }
    std::string offsets_str = obj.substr(offsets_start + 1, *offsets_end_opt - offsets_start - 1);
    std::istringstream offsets_stream(offsets_str);
    std::vector<int64_t> offsets;
    std::string off;
    while (std::getline(offsets_stream, off, ',')) {
      off = trimAscii(off);
      int64_t v = 0;
      if (off.empty() || !parseI64Strict(off, v) || v < 0) {
        throw std::runtime_error("SafeTensors load error: invalid data_offsets entry.");
      }
      offsets.push_back(v);
    }
    if (offsets.size() != 2 || offsets[1] < offsets[0]) {
      throw std::runtime_error("SafeTensors load error: invalid data_offsets range.");
    }
    e.data_start = static_cast<uint64_t>(offsets[0]);
    e.data_end = static_cast<uint64_t>(offsets[1]);

    auto numel_opt = checkedNumel(e.shape);
    if (!numel_opt.has_value()) {
      throw std::runtime_error("SafeTensors load error: invalid or overflowing shape.");
    }
    const size_t elem_size = dtypeSize(e.dtype);
    if (*numel_opt > (std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(elem_size))) {
      throw std::runtime_error("SafeTensors load error: tensor byte size overflows.");
    }
    const uint64_t expected_bytes = *numel_opt * static_cast<uint64_t>(elem_size);
    const uint64_t actual_bytes = e.data_end - e.data_start;
    if (actual_bytes != expected_bytes) {
      throw std::runtime_error(
        "SafeTensors load error: data size mismatch (offset range " +
        std::to_string(actual_bytes) + " bytes vs expected " + std::to_string(expected_bytes) + " bytes).");
    }

    return e;
  }

  void parseHeaderEntries() {
    size_t pos = 0;
    skipWhitespace(header_json, pos);
    if (pos >= header_json.size() || header_json[pos] != '{') {
      throw std::runtime_error("SafeTensors load error: malformed header JSON.");
    }
    pos++;

    while (true) {
      skipWhitespace(header_json, pos);
      if (pos >= header_json.size()) {
        throw std::runtime_error("SafeTensors load error: truncated header JSON.");
      }
      if (header_json[pos] == '}') {
        pos++;
        break;
      }
      std::string key = parseJsonString(header_json, pos);
      skipWhitespace(header_json, pos);
      if (pos >= header_json.size() || header_json[pos] != ':') {
        throw std::runtime_error("SafeTensors load error: malformed header key/value separator.");
      }
      pos++;
      skipWhitespace(header_json, pos);
      if (pos >= header_json.size() || header_json[pos] != '{') {
        throw std::runtime_error("SafeTensors load error: malformed tensor metadata object.");
      }
      auto entry_end_opt = findMatchingDelim(header_json, pos, '{', '}');
      if (!entry_end_opt.has_value()) {
        throw std::runtime_error("SafeTensors load error: malformed tensor metadata object.");
      }
      if (key != "__metadata__") {
        std::string obj = header_json.substr(pos, *entry_end_opt - pos + 1);
        entries.emplace(std::move(key), parseEntryObject(obj));
      }
      pos = *entry_end_opt + 1;
      skipWhitespace(header_json, pos);
      if (pos < header_json.size() && header_json[pos] == ',') {
        pos++;
      }
    }
  }

  SafeTensorsFile(const std::string& p) : path(p), file(p, std::ios::binary) {
    if (!file.is_open()) throw std::runtime_error("Failed to open: " + p);

    file.seekg(0, std::ios::end);
    std::streamoff size_off = file.tellg();
    if (size_off < 0) throw std::runtime_error("Failed to determine safetensors file size: " + p);
    file_size = static_cast<uint64_t>(size_off);
    if (file_size < sizeof(uint64_t)) throw std::runtime_error("SafeTensors load error: file too small to contain metadata size.");
    file.seekg(0, std::ios::beg);

    file.read(reinterpret_cast<char*>(&header_size), sizeof(uint64_t));
    if (header_size > 100000000ULL) throw std::runtime_error("Unreasonable header size");
    if (header_size == 0) throw std::runtime_error("SafeTensors load error: metadata header is empty.");
    if (header_size > file_size - sizeof(uint64_t)) throw std::runtime_error("SafeTensors load error: metadata is truncated.");

    header_json.resize(header_size);
    file.read(&header_json[0], header_size);
    parseHeaderEntries();
  }

  const Entry* findEntry(const std::string& name) const {
    auto it = entries.find(name);
    if (it == entries.end()) {
      return nullptr;
    }
    return &it->second;
  }

  torch::Tensor loadTensor(const Entry& e, const std::vector<int64_t>& expected_shape) {
    const uint64_t n_bytes = e.data_end - e.data_start;
    const uint64_t data_base = sizeof(uint64_t) + header_size;
    if (data_base > file_size || e.data_end > file_size - data_base) {
      throw std::runtime_error("SafeTensors load error: tensor data range exceeds file payload (truncated file).");
    }

    auto tensor = torch::empty(e.shape, torch::TensorOptions().dtype(e.dtype));
    {
      std::lock_guard<std::mutex> guard(io_mutex);
      file.clear();
      file.seekg(static_cast<std::streamoff>(data_base + e.data_start), std::ios::beg);
      if (!file.good()) {
        throw std::runtime_error("SafeTensors load error: failed to seek to tensor data.");
      }
      if (n_bytes > 0 &&
          !file.read(reinterpret_cast<char*>(tensor.data_ptr()), static_cast<std::streamsize>(n_bytes))) {
        throw std::runtime_error("SafeTensors load error: failed to read tensor data (file truncated).");
      }
    }

    if (!expected_shape.empty()) {
      auto expected_numel = checkedNumel(expected_shape);
      if (!expected_numel.has_value()) {
        throw std::runtime_error("SafeTensors load error: invalid expected output shape.");
      }
      if (static_cast<uint64_t>(tensor.numel()) != *expected_numel) {
        throw std::runtime_error("SafeTensors load error: loaded tensor size does not match requested shape.");
      }
      tensor = tensor.reshape(expected_shape);
    }
    return tensor;
  }
};

static std::unordered_map<std::string, std::shared_ptr<SafeTensorsFile>> g_safetensors_cache;
static std::mutex g_safetensors_cache_mutex;

class ShardedSafeTensorsDir {
public:
  struct ResolvedTensor {
    std::shared_ptr<SafeTensorsFile> file;
    SafeTensorsFile::Entry entry;
  };

  std::string dir;
  std::vector<std::shared_ptr<SafeTensorsFile>> shards;
  std::unordered_map<std::string, ResolvedTensor> tensors;

  explicit ShardedSafeTensorsDir(const std::string& dir_) : dir(dir_) {
    std::vector<std::string> paths;
    std::unordered_set<std::string> seen;
    auto pushUniquePath = [&](const std::string& path) {
      if (seen.insert(path).second) {
        paths.push_back(path);
      }
    };

    const std::string model_path = dir + "/model.safetensors";
    if (std::filesystem::exists(model_path)) {
      pushUniquePath(model_path);
    }

    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
      if (!entry.is_regular_file()) {
        continue;
      }
      const auto path = entry.path();
      if (path.extension() != ".safetensors") {
        continue;
      }
      pushUniquePath(path.string());
    }
    std::sort(paths.begin(), paths.end());
    if (paths.empty()) {
      throw std::runtime_error("No .safetensors files found in directory: " + dir);
    }

    for (const auto& path : paths) {
      auto file = [&]() -> std::shared_ptr<SafeTensorsFile> {
        {
          std::lock_guard<std::mutex> guard(g_safetensors_cache_mutex);
          auto it = g_safetensors_cache.find(path);
          if (it != g_safetensors_cache.end()) {
            return it->second;
          }
        }
        auto created = std::make_shared<SafeTensorsFile>(path);
        std::lock_guard<std::mutex> guard(g_safetensors_cache_mutex);
        auto [it, _inserted] = g_safetensors_cache.emplace(path, created);
        return it->second;
      }();

      shards.push_back(file);
      for (const auto& kv : file->entries) {
        auto [_, inserted] = tensors.emplace(kv.first, ResolvedTensor { file, kv.second });
        if (!inserted) {
          throw std::runtime_error("Duplicate tensor found across safetensors shards: " + kv.first);
        }
      }
    }
  }

  const ResolvedTensor* find(const std::string& name) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
      return nullptr;
    }
    return &it->second;
  }
};

static std::unordered_map<std::string, std::shared_ptr<ShardedSafeTensorsDir>> g_sharded_safetensors_cache;
static std::mutex g_sharded_safetensors_cache_mutex;

static std::shared_ptr<SafeTensorsFile> getCachedSafeTensorsFile(const std::string& path) {
  {
    std::lock_guard<std::mutex> guard(g_safetensors_cache_mutex);
    auto it = g_safetensors_cache.find(path);
    if (it != g_safetensors_cache.end()) {
      return it->second;
    }
  }
  auto created = std::make_shared<SafeTensorsFile>(path);
  std::lock_guard<std::mutex> guard(g_safetensors_cache_mutex);
  auto [it, _inserted] = g_safetensors_cache.emplace(path, created);
  return it->second;
}

static std::shared_ptr<ShardedSafeTensorsDir> getCachedShardedSafeTensorsDir(const std::string& dir) {
  {
    std::lock_guard<std::mutex> guard(g_sharded_safetensors_cache_mutex);
    auto it = g_sharded_safetensors_cache.find(dir);
    if (it != g_sharded_safetensors_cache.end()) {
      return it->second;
    }
  }
  auto created = std::make_shared<ShardedSafeTensorsDir>(dir);
  std::lock_guard<std::mutex> guard(g_sharded_safetensors_cache_mutex);
  auto [it, _inserted] = g_sharded_safetensors_cache.emplace(dir, created);
  return it->second;
}

static lean_external_class* g_safetensors_handle_class = nullptr;

static void safetensors_handle_finalizer(void* ptr) {
  delete static_cast<std::shared_ptr<SafeTensorsFile>*>(ptr);
}

static void safetensors_handle_foreach(void* /*ptr*/, b_lean_obj_arg /*f*/) {}

extern "C" LEAN_EXPORT lean_object* lean_torch_safetensors_open(b_lean_obj_arg path_obj) {
  return tyr::checkedIoCall("openHandle", [&] {
  try {
    std::string path(lean_string_cstr(path_obj));
    if (!g_safetensors_handle_class) {
      g_safetensors_handle_class = lean_register_external_class(safetensors_handle_finalizer, safetensors_handle_foreach);
    }

    std::shared_ptr<SafeTensorsFile> loader = getCachedSafeTensorsFile(path);

    // We wrap the shared_ptr in an external object
    auto* wrapper = new std::shared_ptr<SafeTensorsFile>(loader);
    return lean_io_result_mk_ok(lean_alloc_external(g_safetensors_handle_class, wrapper));
  } catch (const std::exception& e) {
    return mkIoUserError(e.what());
  }
  });
}

extern "C" LEAN_EXPORT lean_object* lean_torch_safetensors_load_from_handle(
    b_lean_obj_arg handle,
    b_lean_obj_arg name_obj,
    lean_obj_arg shape_obj) {
  return tyr::checkedIoCall("safeTensorsLoadFromHandle", [&] {
  OwnedLeanObject shape_obj_owner(shape_obj, lean_dec);
  auto* wrapper = static_cast<std::shared_ptr<SafeTensorsFile>*>(lean_get_external_data(handle));
  auto loader = *wrapper;
  std::string name(lean_string_cstr(name_obj));
  auto expected_shape = getShape(shape_obj);
  shape_obj_owner.reset();

  try {
    const auto* entry = loader->findEntry(name);
    if (!entry) {
      return mkIoUserError("Tensor not found: " + name);
    }
    return lean_io_result_mk_ok(fromTorchTensor(loader->loadTensor(*entry, expected_shape)));
  } catch (const std::exception& e) {
    return mkIoUserError(e.what());
  }
  });
}

lean_object* lean_torch_safetensors_load(
  b_lean_obj_arg path_obj,
  b_lean_obj_arg name_obj,
  lean_obj_arg shape_obj
) {
  return tyr::checkedIoCall("safeTensorsLoad", [&] {
  OwnedLeanObject shape_obj_owner(shape_obj, lean_dec);
  std::string path(lean_string_cstr(path_obj));
  std::string tensor_name(lean_string_cstr(name_obj));
  auto expected_shape = getShape(shape_obj);
  shape_obj_owner.reset();

  try {
    auto loader = getCachedSafeTensorsFile(path);
    const auto* entry = loader->findEntry(tensor_name);
    if (!entry) {
      return mkIoUserError("Tensor not found: " + tensor_name);
    }
    return lean_io_result_mk_ok(fromTorchTensor(loader->loadTensor(*entry, expected_shape)));
  } catch (const std::exception& e) {
    return mkIoUserError(e.what());
  }
  });
}
// Load tensor from sharded SafeTensors files
lean_object* lean_torch_safetensors_load_sharded(
  b_lean_obj_arg dir_obj,
  b_lean_obj_arg name_obj,
  lean_obj_arg shape
) {
  return tyr::checkedIoCall("safeTensorsLoadSharded", [&] {
  OwnedLeanObject shape_owner(shape, lean_dec);
  try {
    std::string dir(lean_string_cstr(dir_obj));
    std::string tensor_name(lean_string_cstr(name_obj));
    auto expected_shape = getShape(shape);
    shape_owner.reset();

    auto sharded = getCachedShardedSafeTensorsDir(dir);
    const auto* resolved = sharded->find(tensor_name);
    if (!resolved) {
      return mkIoUserError("Tensor not found in any shard: " + tensor_name);
    }
    return lean_io_result_mk_ok(fromTorchTensor(resolved->file->loadTensor(resolved->entry, expected_shape)));
  } catch (const std::exception& e) {
    return mkIoUserError(e.what());
  }
  });
}

// Find positions of BOS tokens in a 1D tensor
// Returns a 1D int64 tensor containing indices where tokens == bos_token
lean_object* lean_torch_find_bos_positions(
    uint64_t /*n*/,
    b_lean_obj_arg tokens,
    uint64_t bos_token
) {
  return tyr::checkedIoCall("findTokenPositions", [&] {
  try {
    auto tokens_ = borrowTensor(tokens);

    // Create mask where tokens == bos_token
    auto mask = tokens_ == static_cast<int64_t>(bos_token);

    // Get indices where mask is true
    auto positions = torch::nonzero(mask);

    // nonzero returns [N, 1] for 1D input, squeeze to get [N]
    if (positions.dim() > 1) {
      positions = positions.squeeze(1);
    }

    // Ensure int64 dtype
    positions = positions.to(torch::kInt64);

    return lean_io_result_mk_ok(fromTorchTensor(positions));
  } catch (const c10::Error& e) {
    return mkC10IoError("find_bos_positions failed", e);
  } catch (const std::exception& e) {
    return mkStdIoError("find_bos_positions failed", e);
  }
  });
}

// Convert a 1D int64 tensor to a Lean Array UInt64
lean_object* lean_torch_tensor_to_uint64_array(
    uint64_t /*n*/,
    b_lean_obj_arg tensor
) {
  try {
    auto t = borrowTensor(tensor);

    // Ensure tensor is on CPU and contiguous
    t = t.to(torch::kCPU).contiguous().to(torch::kInt64);

    // Get number of elements
    int64_t numel = t.numel();

    // Create Lean array
    lean_object* arr = lean_mk_empty_array();

    // Get data pointer
    auto* ptr = t.data_ptr<int64_t>();

    // Push each element
    for (int64_t i = 0; i < numel; i++) {
      arr = lean_array_push(arr, lean_box_uint64(static_cast<uint64_t>(ptr[i])));
    }

    return lean_io_result_mk_ok(arr);
  } catch (const c10::Error& e) {
    return mkC10IoError("tensor_to_uint64_array failed", e);
  } catch (const std::exception& e) {
    return mkStdIoError("tensor_to_uint64_array failed", e);
  }
}

// Convert a dynamically-shaped int64 tensor to a Lean Array UInt64
// (Same implementation, but without the n parameter for shape-erased tensors)
lean_object* lean_torch_tensor_to_uint64_array_dynamic(
    b_lean_obj_arg tensor
) {
  return tyr::checkedIoCall("toUInt64Array", [&] {
  return lean_torch_tensor_to_uint64_array(0, tensor);
  });
}

// Convert a dynamically-shaped float tensor to a Lean Array Float
lean_object* lean_torch_tensor_to_float_array_dynamic(
    b_lean_obj_arg tensor
) {
  return tyr::checkedIoCall("toFloatArray", [&] {
  try {
    auto t = borrowTensor(tensor);
    t = t.to(torch::kCPU).contiguous().to(torch::kFloat32);
    int64_t numel = t.numel();

    lean_object* arr = lean_mk_empty_array();
    auto* ptr = t.data_ptr<float>();
    for (int64_t i = 0; i < numel; i++) {
      arr = lean_array_push(arr, lean_box_float(static_cast<double>(ptr[i])));
    }
    return lean_io_result_mk_ok(arr);
  } catch (const c10::Error& e) {
    return mkC10IoError("tensor_to_float_array_dynamic failed", e);
  } catch (const std::exception& e) {
    return mkStdIoError("tensor_to_float_array_dynamic failed", e);
  }
  });
}

// Hann window for spectral analysis
lean_object* lean_torch_hann_window(uint64_t n) {
  auto win = torch::hann_window(static_cast<int64_t>(n), torch::TensorOptions().dtype(torch::kFloat32));
  return fromTorchTensor(win);
}

// 1D STFT returning real/imag packed in the last dimension.
lean_object* lean_torch_stft_1d(
    uint64_t /*n*/,
    b_lean_obj_arg input,
    uint64_t n_fft,
    uint64_t hop_length,
    uint64_t win_length,
    b_lean_obj_arg window,
    uint8_t center,
    uint8_t normalized
) {
  auto input_ = borrowTensor(input);
  auto window_ = borrowTensor(window);
  // Match WhisperFeatureExtractor's torch path:
  // torch.stft(..., return_complex=True), then view_as_real for packing.
  auto complex_ = torch::stft(
      input_,
      static_cast<int64_t>(n_fft),
      static_cast<int64_t>(hop_length),
      static_cast<int64_t>(win_length),
      window_,
      center != 0,
      "reflect",
      normalized != 0,
      true,
      true);
  auto packed_ = torch::view_as_real(complex_);
  return fromTorchTensor(packed_);
}

lean_object* lean_torch_stft_1d_checked(
    uint64_t n,
    b_lean_obj_arg input,
    uint64_t n_fft,
    uint64_t hop_length,
    uint64_t win_length,
    b_lean_obj_arg window,
    uint8_t center,
    uint8_t normalized
) {
  return tyr::checkedCall("stft", [&] {
    return lean_torch_stft_1d(n, input, n_fft, hop_length, win_length,
                             window, center, normalized);
  });
}

// 1D RFFT returning real/imag packed in the last dimension.
lean_object* lean_torch_rfft_1d(uint64_t /*n*/, b_lean_obj_arg input) {
  auto input_ = borrowTensor(input);
  auto complex_ = at::fft_rfft(input_, c10::nullopt, -1, c10::nullopt);
  auto packed_ = torch::view_as_real(complex_);
  return fromTorchTensor(packed_);
}

// 1D inverse STFT from real/imag packed input.
lean_object* lean_torch_istft_1d(
    b_lean_obj_arg input,
    uint64_t n_fft,
    uint64_t hop_length,
    uint64_t win_length,
    b_lean_obj_arg window,
    uint8_t center,
    uint8_t normalized,
    uint64_t length
) {
  auto input_ = borrowTensor(input).contiguous();
  auto window_ = borrowTensor(window);
  auto complex_ = torch::view_as_complex(input_);
  auto out_ = torch::istft(
      complex_,
      static_cast<int64_t>(n_fft),
      static_cast<int64_t>(hop_length),
      static_cast<int64_t>(win_length),
      window_,
      center != 0,
      normalized != 0,
      true,
      length == 0 ? c10::optional<int64_t>() : c10::optional<int64_t>(static_cast<int64_t>(length)),
      false);
  return fromTorchTensor(out_);
}

// ============================================================================
// NanoProof FFI bindings
// ============================================================================

// RMSNorm: x / sqrt(mean(x^2) + eps) - no learnable parameters
lean_object* lean_torch_rms_norm(lean_obj_arg /*s*/, b_lean_obj_arg input, double eps) {
  auto input_ = borrowTensor(input);
  // Compute RMSNorm: x * rsqrt(mean(x^2) + eps)
  auto variance = input_.pow(2).mean(-1, /*keepdim=*/true);
  auto result_ = input_ * torch::rsqrt(variance + eps);
  return fromTorchTensor(result_);
}

// RMSNorm with learnable weight: (x / sqrt(mean(x^2) + eps)) * weight
// weight broadcasts over the last dimension
lean_object* lean_torch_rms_norm_weighted(lean_obj_arg /*s*/, lean_obj_arg /*w*/,
    b_lean_obj_arg input, b_lean_obj_arg weight, double eps) {
  auto input_ = borrowTensor(input);
  auto weight_ = borrowTensor(weight);
  // Compute RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
  auto variance = input_.pow(2).mean(-1, /*keepdim=*/true);
  auto result_ = input_ * torch::rsqrt(variance + eps) * weight_;
  return fromTorchTensor(result_);
}

// ReLU squared activation: relu(x)^2
lean_object* lean_torch_relu_squared(lean_obj_arg /*s*/, b_lean_obj_arg input) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::relu(input_).square();
  return fromTorchTensor(result_);
}

// Logit softcap: cap * tanh(x / cap)
lean_object* lean_torch_softcap(lean_obj_arg s, b_lean_obj_arg input, double cap) {
  lean_dec(s);
  auto input_ = borrowTensor(input);
  auto result_ = cap * torch::tanh(input_ / cap);
  return fromTorchTensor(result_);
}

// Shared implementation for computing rotary embedding frequencies
static lean_object* compute_rotary_freqs_impl(
  uint64_t seq_len,
  uint64_t head_dim,
  double base,
  const torch::Device& device
) {

  // inv_freq = 1.0 / (base ** (arange(0, head_dim, 2) / head_dim))
  auto channel_range = torch::arange(0, static_cast<int64_t>(head_dim), 2,
    torch::TensorOptions().dtype(torch::kFloat32).device(device));
  auto inv_freq = 1.0 / torch::pow(base, channel_range / static_cast<double>(head_dim));

  // t = arange(seq_len)
  auto t = torch::arange(static_cast<int64_t>(seq_len),
    torch::TensorOptions().dtype(torch::kFloat32).device(device));

  // freqs = outer(t, inv_freq) -> [seq_len, head_dim/2]
  auto freqs = torch::outer(t, inv_freq);
  auto cos = freqs.cos();
  auto sin = freqs.sin();

  // Return as Lean pair (Product type, ctor 0 with 2 fields)
  lean_object* result = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(result, 0, fromTorchTensor(cos));
  lean_ctor_set(result, 1, fromTorchTensor(sin));
  return result;
}

// Precompute rotary embedding frequencies (IO version)
// Returns (cos, sin) tensors of shape [seq_len, head_dim/2]
lean_object* lean_torch_compute_rotary_freqs(
  uint64_t seq_len,
  uint64_t head_dim,
  double base,
  lean_object* w
) {
  try {
    return lean_io_result_mk_ok(
      compute_rotary_freqs_impl(seq_len, head_dim, base, torch::Device(torch::kCPU))
    );
  } catch (const c10::Error& e) {
    return mkC10IoError("compute_rotary_freqs failed", e);
  } catch (const std::exception& e) {
    return mkStdIoError("compute_rotary_freqs failed", e);
  }
}

// Precompute rotary embedding frequencies (pure version)
// Returns (cos, sin) tensors of shape [seq_len, head_dim/2]
lean_object* lean_torch_compute_rotary_freqs_pure(
  uint64_t seq_len,
  uint64_t head_dim,
  double base
) {
  return compute_rotary_freqs_impl(seq_len, head_dim, base, torch::Device(torch::kCPU));
}

// Precompute rotary embedding frequencies on a target device (IO version)
// Returns (cos, sin) tensors of shape [seq_len, head_dim/2]
lean_object* lean_torch_compute_rotary_freqs_on_device(
  uint64_t seq_len,
  uint64_t head_dim,
  double base,
  lean_obj_arg device,
  lean_object* /*w*/
) {
  auto device_ = getDevice(device);
  lean_dec(device);
  try {
    return lean_io_result_mk_ok(compute_rotary_freqs_impl(seq_len, head_dim, base, device_));
  } catch (const c10::Error& e) {
    return mkC10IoError("compute_rotary_freqs_on_device failed", e);
  } catch (const std::exception& e) {
    return mkStdIoError("compute_rotary_freqs_on_device failed", e);
  }
}

// Precompute rotary embedding frequencies on a target device (pure version)
// Returns (cos, sin) tensors of shape [seq_len, head_dim/2]
lean_object* lean_torch_compute_rotary_freqs_on_device_pure(
  uint64_t seq_len,
  uint64_t head_dim,
  double base,
  lean_obj_arg device
) {
  OwnedLeanObject device_owner(device, lean_dec);
  auto device_ = getDevice(device);
  device_owner.reset();
  return compute_rotary_freqs_impl(seq_len, head_dim, base, device_);
}

// Apply rotary embeddings to Q or K
// x: [batch, seq, n_head, head_dim]
// cos, sin: [seq, head_dim/2] (will be broadcast)
lean_object* lean_torch_apply_rotary_emb(
  uint64_t /*batch*/,
  uint64_t /*seq*/,
  uint64_t /*n_head*/,
  uint64_t /*head_dim*/,
  b_lean_obj_arg x,
  b_lean_obj_arg cos,
  b_lean_obj_arg sin
) {
  auto x_ = borrowTensor(x);
  auto cos_ = borrowTensor(cos);
  auto sin_ = borrowTensor(sin);
  if (cos_.device() != x_.device()) {
    cos_ = cos_.to(x_.device());
  }
  if (sin_.device() != x_.device()) {
    sin_ = sin_.to(x_.device());
  }

  int64_t d = x_.size(3) / 2;
  auto x1 = x_.slice(3, 0, d);
  auto x2 = x_.slice(3, d, 2*d);

  // Broadcast cos/sin: [seq, d] -> [1, seq, 1, d]
  auto cos_b = cos_.unsqueeze(0).unsqueeze(2);
  auto sin_b = sin_.unsqueeze(0).unsqueeze(2);

  // Apply rotation (matches flux2 / standard RoPE):
  // [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
  auto y1 = x1 * cos_b - x2 * sin_b;
  auto y2 = x1 * sin_b + x2 * cos_b;
  auto result_ = torch::cat({y1, y2}, 3);

  // Ensure output dtype matches input
  result_ = result_.to(x_.dtype());

  return fromTorchTensor(result_);
}

// Create sinusoidal timestep embeddings
// t: [batch] timestep values
// dim: output dimension
// Returns: [batch, dim] positional embedding
lean_object* lean_torch_timestep_embedding(
  uint64_t /*batch*/,
  b_lean_obj_arg t_obj,
  uint64_t dim,
  double max_period,
  double time_factor
) {
  auto t = borrowTensor(t_obj);
  auto device = t.device();
  auto dtype = t.dtype();

  // Scale timesteps
  t = t * time_factor;

  // half = dim // 2
  int64_t half = dim / 2;

  // freqs = exp(-log(max_period) * arange(half) / half)
  auto indices = torch::arange(0, half, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  auto freqs = torch::exp(-std::log(max_period) * indices / static_cast<double>(half));

  // args = t[:, None] * freqs[None, :]
  auto args = t.unsqueeze(1).to(torch::kFloat32) * freqs.unsqueeze(0);

  // embedding = cat([cos(args), sin(args)], dim=-1)
  auto cos_emb = torch::cos(args);
  auto sin_emb = torch::sin(args);
  auto embedding = torch::cat({cos_emb, sin_emb}, -1);

  // Handle odd dimensions
  if (dim % 2 == 1) {
    auto zeros = torch::zeros({t.size(0), 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    embedding = torch::cat({embedding, zeros}, -1);
  }

  // Cast back to input dtype if needed
  if (torch::is_floating_point(t)) {
    embedding = embedding.to(dtype);
  }

  return fromTorchTensor(embedding);
}

// Scaled dot-product attention with Group-Query Attention (GQA) support
// Q: [batch, n_head, seq, head_dim]
// K, V: [batch, n_kv_head, seq, head_dim]
lean_object* lean_torch_sdpa_gqa(
  uint64_t /*batch*/,
  uint64_t /*n_head*/,
  uint64_t /*n_kv_head*/,
  uint64_t /*seq*/,
  uint64_t /*head_dim*/,
  b_lean_obj_arg query,
  b_lean_obj_arg key,
  b_lean_obj_arg value,
  double dropout_p,
  uint8_t is_causal,
  uint8_t enable_gqa
) {
  auto q = borrowTensor(query);
  auto k = borrowTensor(key);
  auto v = borrowTensor(value);

  auto result_ = tyr_ops::flash_attn_dispatch(
    q, k, v, c10::nullopt, dropout_p, is_causal != 0, c10::nullopt, enable_gqa != 0);

  return fromTorchTensor(result_);
}

// Scaled dot-product attention with GQA and sliding window support
// Each position can only attend to the previous window_size positions
lean_object* lean_torch_sdpa_gqa_window(
  uint64_t /*batch*/,
  uint64_t /*n_head*/,
  uint64_t /*n_kv_head*/,
  uint64_t /*seq*/,
  uint64_t /*head_dim*/,
  b_lean_obj_arg query,
  b_lean_obj_arg key,
  b_lean_obj_arg value,
  double dropout_p,
  uint8_t /*is_causal*/,
  uint8_t enable_gqa,
  uint64_t window_size
) {

  auto q = borrowTensor(query);
  auto k = borrowTensor(key);
  auto v = borrowTensor(value);
  // The public window binding has always been causal, including is_causal=false.
  // Clamp oversized windows to the sequence length before narrowing to int64_t.
  const auto window = static_cast<int64_t>(std::min<uint64_t>(window_size, q.size(2)));
  return fromTorchTensor(tyr_ops::bounded_sdpa(q, k, v, c10::nullopt,
      tyr_ops::AttentionMaskKind::Padding, dropout_p, true, c10::nullopt,
      enable_gqa != 0, window));
}

// Scaled dot-product attention with GQA and explicit attention mask
// Q: [batch, n_head, seq, head_dim]
// K, V: [batch, n_kv_head, seq, head_dim]
// attn_mask: [batch, seq] - padding mask (1 for valid, 0 for padding)
lean_object* lean_torch_sdpa_gqa_mask(
  uint64_t /*batch*/,
  uint64_t /*n_head*/,
  uint64_t /*n_kv_head*/,
  uint64_t /*seq*/,
  uint64_t /*head_dim*/,
  b_lean_obj_arg query,
  b_lean_obj_arg key,
  b_lean_obj_arg value,
  b_lean_obj_arg mask,
  double dropout_p,
  uint8_t is_causal,
  uint8_t enable_gqa
) {

  return fromTorchTensor(tyr_ops::bounded_sdpa(
      borrowTensor(query), borrowTensor(key), borrowTensor(value), borrowTensor(mask),
      tyr_ops::AttentionMaskKind::Padding, dropout_p, is_causal != 0,
      c10::nullopt, enable_gqa != 0));
}

// Scaled dot-product attention with GQA and explicit query/key mask.
// Q: [batch, n_head, q_seq, head_dim]
// K, V: [batch, n_kv_head, kv_seq, head_dim]
// attn_mask: [batch, q_seq, kv_seq] - 1 for allowed attention edges, 0 for masked edges
LEAN_EXPORT lean_object* lean_torch_sdpa_gqa_mask_qkv(
  uint64_t /*batch*/,
  uint64_t /*n_head*/,
  uint64_t /*n_kv_head*/,
  uint64_t /*q_seq*/,
  uint64_t /*kv_seq*/,
  uint64_t /*head_dim*/,
  b_lean_obj_arg query,
  b_lean_obj_arg key,
  b_lean_obj_arg value,
  b_lean_obj_arg mask,
  double dropout_p,
  uint8_t enable_gqa
) {

  return fromTorchTensor(tyr_ops::bounded_sdpa(
      borrowTensor(query), borrowTensor(key), borrowTensor(value), borrowTensor(mask),
      tyr_ops::AttentionMaskKind::Edges, dropout_p, false, c10::nullopt, enable_gqa != 0));
}

// Scaled dot-product attention with GQA where query and KV sequence lengths may differ.
// Q: [batch, n_head, q_seq, head_dim]
// K, V: [batch, n_kv_head, kv_seq, head_dim]
lean_object* lean_torch_sdpa_gqa_qkv(
  uint64_t /*batch*/,
  uint64_t /*n_head*/,
  uint64_t /*n_kv_head*/,
  uint64_t /*q_seq*/,
  uint64_t /*kv_seq*/,
  uint64_t /*head_dim*/,
  b_lean_obj_arg query,
  b_lean_obj_arg key,
  b_lean_obj_arg value,
  double dropout_p,
  uint8_t is_causal,
  uint8_t enable_gqa
) {
  auto q = borrowTensor(query);
  auto k = borrowTensor(key);
  auto v = borrowTensor(value);

  auto result_ = tyr_ops::flash_attn_dispatch(
    q, k, v, c10::nullopt, dropout_p, is_causal != 0, c10::nullopt, enable_gqa != 0);
  return fromTorchTensor(result_);
}

// ============================================================================
// Comparison and conditional operations for diffusion
// ============================================================================

// Less than comparison: returns a boolean tensor
lean_object* lean_torch_lt(lean_obj_arg s, b_lean_obj_arg a, b_lean_obj_arg b) {
  lean_dec(s);
  auto a_ = borrowTensor(a);
  auto b_ = borrowTensor(b);
  auto result_ = torch::lt(a_, b_);
  return fromTorchTensor(result_);
}

// Less than scalar comparison
lean_object* lean_torch_lt_scalar(lean_obj_arg s, b_lean_obj_arg input, double scalar) {
  lean_dec(s);
  auto input_ = borrowTensor(input);
  auto result_ = input_ < scalar;
  return fromTorchTensor(result_);
}

// Greater than comparison
lean_object* lean_torch_gt(lean_obj_arg s, b_lean_obj_arg a, b_lean_obj_arg b) {
  lean_dec(s);
  auto a_ = borrowTensor(a);
  auto b_ = borrowTensor(b);
  auto result_ = torch::gt(a_, b_);
  return fromTorchTensor(result_);
}

// Greater than or equal comparison
lean_object* lean_torch_ge(lean_obj_arg s, b_lean_obj_arg a, b_lean_obj_arg b) {
  lean_dec(s);
  auto a_ = borrowTensor(a);
  auto b_ = borrowTensor(b);
  auto result_ = torch::ge(a_, b_);
  return fromTorchTensor(result_);
}

// Equality comparison: returns a boolean tensor
lean_object* lean_torch_eq(lean_obj_arg /*s*/, b_lean_obj_arg a, b_lean_obj_arg b) {
  auto a_ = borrowTensor(a);
  auto b_ = borrowTensor(b);
  auto result_ = torch::eq(a_, b_);
  return fromTorchTensor(result_);
}

// Equality with scalar comparison
lean_object* lean_torch_eq_scalar(lean_obj_arg s, b_lean_obj_arg input, uint64_t scalar) {
  lean_dec(s);
  auto input_ = borrowTensor(input);
  auto result_ = input_ == static_cast<int64_t>(scalar);
  return fromTorchTensor(result_);
}

// Full tensor with given shape and scalar value (int64 version for token ids)
lean_object* lean_torch_full_int(lean_obj_arg s, uint64_t value_bits) {
  OwnedLeanObject s_owner(s, lean_dec);
  const int64_t value = static_cast<int64_t>(value_bits);
  auto shape = getShape(s);
  s_owner.reset();
  auto result_ = torch::full(shape, value, torch::kInt64);
  return fromTorchTensor(result_);
}

// Maximum along dimension with indices
lean_object* lean_torch_max_dim(lean_obj_arg /*s*/, b_lean_obj_arg input, uint64_t dim) {
  auto input_ = borrowTensor(input);
  auto [values, indices] = torch::max(input_, static_cast<int64_t>(dim));
  // Return as Lean pair
  lean_object* result = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(result, 0, fromTorchTensor(values));
  lean_ctor_set(result, 1, fromTorchTensor(indices));
  return result;
}

// Maximum along dimension for 3D tensors with proper output shape
// d0, d1, d2 are implicit type parameters
lean_object* lean_torch_max_dim_3d(uint64_t /*d0*/, uint64_t /*d1*/, uint64_t /*d2*/,
    b_lean_obj_arg input, uint64_t dim) {
  auto input_ = borrowTensor(input);
  auto [values, indices] = torch::max(input_, static_cast<int64_t>(dim));
  // Return as Lean pair
  lean_object* result = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(result, 0, fromTorchTensor(values));
  lean_ctor_set(result, 1, fromTorchTensor(indices));
  return result;
}

lean_object* lean_torch_max_dim_3d_checked(uint64_t /*d0*/, uint64_t /*d1*/, uint64_t /*d2*/,
    b_lean_obj_arg input, uint64_t dim) {
  return tyr::checkedCall("maxLast3d", [&] {
    TORCH_CHECK(dim <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      "maxLast3d axis is not representable as a nonnegative signed 64-bit integer");
    auto [values, indices] = torch::max(borrowTensor(input), static_cast<int64_t>(dim));
    OwnedLeanObject values_owner(fromTorchTensor(values), lean_dec);
    OwnedLeanObject indices_owner(fromTorchTensor(indices), lean_dec);
    auto* result = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(result, 0, values_owner.release());
    lean_ctor_set(result, 1, indices_owner.release());
    return result;
  });
}

// Boolean any (returns true if any element is true)
uint8_t lean_torch_any(lean_obj_arg s, b_lean_obj_arg input) {
  lean_dec(s);
  auto input_ = borrowTensor(input);
  return input_.any().item<bool>();
}

// Logical NOT for boolean tensors
lean_object* lean_torch_logical_not(lean_obj_arg s, b_lean_obj_arg input) {
  lean_dec(s);
  auto input_ = borrowTensor(input);
  auto result_ = torch::logical_not(input_);
  return fromTorchTensor(result_);
}

// Logical AND for boolean tensors
lean_object* lean_torch_logical_and(lean_obj_arg s, b_lean_obj_arg a, b_lean_obj_arg b) {
  lean_dec(s);
  auto a_ = borrowTensor(a);
  auto b_ = borrowTensor(b);
  auto result_ = torch::logical_and(a_, b_);
  return fromTorchTensor(result_);
}

// Convert to float dtype
lean_object* lean_torch_to_float(lean_obj_arg s, b_lean_obj_arg input) {
  lean_dec(s);
  auto input_ = borrowTensor(input);
  auto result_ = input_.to(torch::kFloat32);
  return fromTorchTensor(result_);
}

// Convert to bfloat16 dtype
lean_object* lean_torch_to_bfloat16(lean_obj_arg s, b_lean_obj_arg input) {
  lean_dec(s);
  auto input_ = borrowTensor(input);
  auto result_ = input_.to(torch::kBFloat16);
  return fromTorchTensor(result_);
}

// Index select for 1D indexing into first dimension
lean_object* lean_torch_index_select_1d(lean_obj_arg /*s*/, b_lean_obj_arg input, b_lean_obj_arg indices) {
  auto input_ = borrowTensor(input);
  auto indices_ = borrowTensor(indices);
  auto result_ = input_.index_select(0, indices_);
  return fromTorchTensor(result_);
}

// Clamp values to a range
lean_object* lean_torch_clamp(lean_obj_arg /*s*/, b_lean_obj_arg input, int64_t min_val, int64_t max_val) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::clamp(input_, min_val, max_val);
  return fromTorchTensor(result_);
}

// Clamp values to a floating-point range
lean_object* lean_torch_clamp_float(lean_obj_arg s, b_lean_obj_arg input, double min_val, double max_val) {
  lean_dec(s);
  auto input_ = borrowTensor(input);
  auto result_ = torch::clamp(input_, min_val, max_val);
  return fromTorchTensor(result_);
}

// Top-k values and indices
lean_object* lean_torch_topk(lean_obj_arg /*s*/, b_lean_obj_arg input, uint64_t k, uint64_t dim) {
  auto input_ = borrowTensor(input);
  auto [values, indices] = torch::topk(input_, k, dim);
  // Return as Lean pair
  lean_object* result = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(result, 0, fromTorchTensor(values));
  lean_ctor_set(result, 1, fromTorchTensor(indices));
  return result;
}

// Top-k for 2D tensors with proper output shape
// d1, d2 are the implicit shape parameters
lean_object* lean_torch_topk_2d(uint64_t /*d1*/, uint64_t /*d2*/,
    b_lean_obj_arg input, uint64_t k, uint64_t dim) {
  auto input_ = borrowTensor(input);
  auto [values, indices] = torch::topk(input_, k, dim);
  // Return as Lean pair
  lean_object* result = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(result, 0, fromTorchTensor(values));
  lean_ctor_set(result, 1, fromTorchTensor(indices));
  return result;
}

lean_object* lean_torch_topk_2d_checked(uint64_t /*d1*/, uint64_t /*d2*/,
    b_lean_obj_arg input, uint64_t k, uint64_t dim) {
  return tyr::checkedCall("topKLast2d", [&] {
    TORCH_CHECK(k <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) &&
      dim <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      "topKLast2d k and axis must be nonnegative signed 64-bit integers");
    auto [values, indices] = torch::topk(borrowTensor(input), static_cast<int64_t>(k),
      static_cast<int64_t>(dim));
    OwnedLeanObject values_owner(fromTorchTensor(values), lean_dec);
    OwnedLeanObject indices_owner(fromTorchTensor(indices), lean_dec);
    auto* result = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(result, 0, values_owner.release());
    lean_ctor_set(result, 1, indices_owner.release());
    return result;
  });
}

// ============================================================================
// New operations: Device availability, gather/scatter, einsum, interpolate
// ============================================================================

// Check if CUDA is available
lean_object* lean_torch_cuda_is_available(lean_object* /*w*/) {
  try {
    return lean_io_result_mk_ok(lean_box(torch::cuda::is_available()));
  } catch (const c10::Error& e) {
    return mkC10IoError("cuda_is_available failed", e);
  } catch (const std::exception& e) {
    return mkStdIoError("cuda_is_available failed", e);
  }
}

// Get the current CUDA stream as an opaque UInt64 handle.
lean_object* lean_torch_cuda_current_stream(lean_object* /*w*/) {
  try {
    if (!torch::cuda::is_available()) {
      return lean_io_result_mk_ok(lean_box_uint64(0));
    }
#if TYR_HAS_CUDA_API
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto raw = reinterpret_cast<uint64_t>(stream.stream());
    return lean_io_result_mk_ok(lean_box_uint64(raw));
#else
    return lean_io_result_mk_ok(lean_box_uint64(0));
#endif
  } catch (const c10::Error& e) {
    return mkC10IoError("cuda_current_stream failed", e);
  } catch (const std::exception& e) {
    return mkStdIoError("cuda_current_stream failed", e);
  }
}

// Synchronize CUDA device for deterministic validation around external
// launches. Calls cudart directly rather than `c10::cuda::device_synchronize`
// because the latter throws `c10::Error` on failure, which cannot be
// caught across the libc++abi/libstdc++ ABI boundary in Tyr binaries
// (see the comment block above `lean_torch_manual_seed`). `cudaDeviceSynchronize`
// returns a `cudaError_t` and never throws.
lean_object* lean_torch_cuda_synchronize(lean_object* /*w*/) {
  if (!torch::cuda::is_available()) {
    return lean_io_result_mk_ok(lean_box(0));
  }
#if TYR_HAS_CUDA_API
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string((std::string("cuda_synchronize failed: ") +
                      cudaGetErrorString(err)).c_str())));
  }
#endif
  return lean_io_result_mk_ok(lean_box(0));
}

#if TYR_HAS_CUDA_API
static cudaEvent_t tyr_cuda_event_from_raw(uint64_t raw) {
  return reinterpret_cast<cudaEvent_t>(raw);
}

static lean_object* tyr_cuda_error(const char* operation, cudaError_t err) {
  return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(
    (std::string(operation) + " failed: " + cudaGetErrorString(err)).c_str())));
}
#endif

lean_object* lean_torch_cuda_event_create(lean_object* /*w*/) {
#if TYR_HAS_CUDA_API
  cudaEvent_t event{};
  const cudaError_t err = cudaEventCreate(&event);
  if (err != cudaSuccess) return tyr_cuda_error("cudaEventCreate", err);
  return lean_io_result_mk_ok(lean_box_uint64(reinterpret_cast<uint64_t>(event)));
#else
  return lean_io_result_mk_error(lean_mk_io_user_error(
    lean_mk_string("CUDA event timing requires the CUDA runtime API")));
#endif
}

lean_object* lean_torch_cuda_event_record(uint64_t raw_event, uint64_t raw_stream,
                                           lean_object* /*w*/) {
#if TYR_HAS_CUDA_API
  const cudaError_t err = cudaEventRecord(
    tyr_cuda_event_from_raw(raw_event), reinterpret_cast<cudaStream_t>(raw_stream));
  if (err != cudaSuccess) return tyr_cuda_error("cudaEventRecord", err);
  return lean_io_result_mk_ok(lean_box(0));
#else
  return lean_io_result_mk_error(lean_mk_io_user_error(
    lean_mk_string("CUDA event timing requires the CUDA runtime API")));
#endif
}

lean_object* lean_torch_cuda_event_synchronize(uint64_t raw_event, lean_object* /*w*/) {
#if TYR_HAS_CUDA_API
  const cudaError_t err = cudaEventSynchronize(tyr_cuda_event_from_raw(raw_event));
  if (err != cudaSuccess) return tyr_cuda_error("cudaEventSynchronize", err);
  return lean_io_result_mk_ok(lean_box(0));
#else
  return lean_io_result_mk_error(lean_mk_io_user_error(
    lean_mk_string("CUDA event timing requires the CUDA runtime API")));
#endif
}

lean_object* lean_torch_cuda_event_elapsed_ms(uint64_t raw_start, uint64_t raw_stop,
                                               lean_object* /*w*/) {
#if TYR_HAS_CUDA_API
  float elapsed_ms = 0.0f;
  const cudaError_t err = cudaEventElapsedTime(
    &elapsed_ms, tyr_cuda_event_from_raw(raw_start), tyr_cuda_event_from_raw(raw_stop));
  if (err != cudaSuccess) return tyr_cuda_error("cudaEventElapsedTime", err);
  return lean_io_result_mk_ok(lean_box_float(static_cast<double>(elapsed_ms)));
#else
  return lean_io_result_mk_error(lean_mk_io_user_error(
    lean_mk_string("CUDA event timing requires the CUDA runtime API")));
#endif
}

lean_object* lean_torch_cuda_event_destroy(uint64_t raw_event, lean_object* /*w*/) {
#if TYR_HAS_CUDA_API
  const cudaError_t err = cudaEventDestroy(tyr_cuda_event_from_raw(raw_event));
  if (err != cudaSuccess) return tyr_cuda_error("cudaEventDestroy", err);
#endif
  return lean_io_result_mk_ok(lean_box(0));
}

// Check if MPS is available
lean_object* lean_torch_mps_is_available(lean_object* /*w*/) {
#ifdef __APPLE__
  try {
    const int debug_flag = std::getenv("TYR_DEBUG_MPS") != nullptr ? 1 : 0;
    const bool available = tyr_apple_mps_is_available(debug_flag);
    return lean_io_result_mk_ok(lean_box(available));
  } catch (const c10::Error& e) {
    if (std::getenv("TYR_DEBUG_MPS") != nullptr) {
      std::cerr << "[tyr:mps] c10::Error: " << e.what() << std::endl;
    }
    return mkC10IoError("mps_is_available failed", e);
  } catch (const std::exception& e) {
    if (std::getenv("TYR_DEBUG_MPS") != nullptr) {
      std::cerr << "[tyr:mps] std::exception: " << e.what() << std::endl;
    }
    return mkStdIoError("mps_is_available failed", e);
  }
#else
  return lean_io_result_mk_ok(lean_box(false));
#endif
}

// Transposed 2D convolution
lean_object* lean_torch_conv_transpose2d(
    lean_obj_arg /*input_shape*/, lean_obj_arg /*weight_shape*/,
    b_lean_obj_arg input, b_lean_obj_arg weight,
    lean_obj_arg stride, lean_obj_arg padding,
    lean_obj_arg output_padding, lean_obj_arg dilation) {
  auto input_ = borrowTensor(input);
  auto weight_ = borrowTensor(weight);
  auto stride_ = getShape(stride); lean_dec(stride);
  auto padding_ = getShape(padding); lean_dec(padding);
  auto output_padding_ = getShape(output_padding); lean_dec(output_padding);
  auto dilation_ = getShape(dilation); lean_dec(dilation);

  auto result_ = torch::nn::functional::conv_transpose2d(input_, weight_,
    torch::nn::functional::ConvTranspose2dFuncOptions()
      .stride(stride_)
      .padding(padding_)
      .output_padding(output_padding_)
      .dilation(dilation_));
  return fromTorchTensor(result_);
}

lean_object* lean_torch_conv_transpose1d_bias(
    lean_obj_arg input_shape, lean_obj_arg weight_shape, lean_obj_arg bias_shape,
    b_lean_obj_arg input, b_lean_obj_arg weight, b_lean_obj_arg bias,
    uint64_t stride, uint64_t padding, uint64_t output_padding, uint64_t dilation) {
  lean_dec(input_shape);
  lean_dec(weight_shape);
  lean_dec(bias_shape);
  auto input_ = borrowTensor(input);
  auto weight_ = borrowTensor(weight);
  auto bias_ = borrowTensor(bias);
  auto result_ = torch::nn::functional::conv_transpose1d(
    input_,
    weight_,
    torch::nn::functional::ConvTranspose1dFuncOptions()
      .bias(bias_)
      .stride(static_cast<int64_t>(stride))
      .padding(static_cast<int64_t>(padding))
      .output_padding(static_cast<int64_t>(output_padding))
      .dilation(static_cast<int64_t>(dilation)));
  return fromTorchTensor(result_);
}

lean_object* lean_torch_conv_transpose1d_bias_checked(
    lean_obj_arg input_shape, lean_obj_arg weight_shape, lean_obj_arg bias_shape,
    b_lean_obj_arg input, b_lean_obj_arg weight, b_lean_obj_arg bias,
    uint64_t stride, uint64_t padding, uint64_t output_padding, uint64_t dilation) {
  return tyr::checkedCall("convTranspose1d", [&] {
    return lean_torch_conv_transpose1d_bias(input_shape, weight_shape, bias_shape,
      input, weight, bias, stride, padding, output_padding, dilation);
  });
}

// 2D spatial dropout
lean_object* lean_torch_dropout2d(
    lean_obj_arg /*n*/, lean_obj_arg /*c*/, lean_obj_arg /*h*/, lean_obj_arg /*w*/,
    b_lean_obj_arg input, double p, uint8_t training, lean_object* /*w*/) {
  try {
    auto input_ = borrowTensor(input);
    auto result_ = torch::nn::functional::dropout2d(input_,
      torch::nn::functional::Dropout2dFuncOptions()
        .p(p)
        .training(training != 0));
    return lean_io_result_mk_ok(fromTorchTensor(result_));
  } catch (const c10::Error& e) {
    return mkC10IoError("dropout2d failed", e);
  } catch (const std::exception& e) {
    return mkStdIoError("dropout2d failed", e);
  }
}

// 3D spatial dropout
lean_object* lean_torch_dropout3d(
    lean_obj_arg /*n*/, lean_obj_arg /*c*/, lean_obj_arg /*d*/, lean_obj_arg /*h*/, lean_obj_arg /*w_dim*/,
    b_lean_obj_arg input, double p, uint8_t training, lean_object* /*w*/) {
  try {
    auto input_ = borrowTensor(input);
    auto result_ = torch::nn::functional::dropout3d(input_,
      torch::nn::functional::Dropout3dFuncOptions()
        .p(p)
        .training(training != 0));
    return lean_io_result_mk_ok(fromTorchTensor(result_));
  } catch (const c10::Error& e) {
    return mkC10IoError("dropout3d failed", e);
  } catch (const std::exception& e) {
    return mkStdIoError("dropout3d failed", e);
  }
}

// NLL loss
lean_object* lean_torch_nll_loss(
    lean_obj_arg /*n*/, lean_obj_arg /*c*/,
    b_lean_obj_arg log_probs, b_lean_obj_arg targets,
    lean_obj_arg reduction) {
  auto log_probs_ = borrowTensor(log_probs);
  auto targets_ = borrowTensor(targets);
  auto reduction_str = std::string(lean_string_cstr(reduction));
  lean_dec(reduction);

  torch::nn::functional::NLLLossFuncOptions options;
  if (reduction_str == "mean") options.reduction(torch::kMean);
  else if (reduction_str == "sum") options.reduction(torch::kSum);
  else options.reduction(torch::kNone);

  auto result_ = torch::nn::functional::nll_loss(log_probs_, targets_, options);
  return fromTorchTensor(result_);
}

// NLL loss with no reduction
lean_object* lean_torch_nll_loss_none(
    lean_obj_arg /*n*/, lean_obj_arg /*c*/,
    b_lean_obj_arg log_probs, b_lean_obj_arg targets) {
  auto log_probs_ = borrowTensor(log_probs);
  auto targets_ = borrowTensor(targets);
  auto result_ = torch::nn::functional::nll_loss(log_probs_, targets_,
    torch::nn::functional::NLLLossFuncOptions().reduction(torch::kNone));
  return fromTorchTensor(result_);
}

// Gather along dimension
lean_object* lean_torch_gather(
    lean_obj_arg input_shape, lean_obj_arg index_shape,
    b_lean_obj_arg input, uint64_t dimBits, b_lean_obj_arg indices) {
  lean_dec(input_shape);
  lean_dec(index_shape);
  const auto dim = std::bit_cast<int64_t>(dimBits);
  auto input_ = borrowTensor(input);
  auto indices_ = borrowTensor(indices);
  auto result_ = torch::gather(input_, dim, indices_);
  return fromTorchTensor(result_);
}

// Scatter along dimension
lean_object* lean_torch_scatter(
    lean_obj_arg /*s*/,
    b_lean_obj_arg input, int64_t dim, b_lean_obj_arg indices, b_lean_obj_arg src) {
  auto input_ = borrowTensor(input);
  auto indices_ = borrowTensor(indices);
  auto src_ = borrowTensor(src);
  auto result_ = input_.clone().scatter_(dim, indices_, src_);
  return fromTorchTensor(result_);
}

// Scatter with add reduction
lean_object* lean_torch_scatter_add(
    lean_obj_arg s,
    b_lean_obj_arg input, uint64_t dim, b_lean_obj_arg indices, b_lean_obj_arg src) {
  lean_dec(s);
  auto input_ = borrowTensor(input);
  auto indices_ = borrowTensor(indices);
  auto src_ = borrowTensor(src);
  auto result_ = input_.clone().scatter_add_(static_cast<int64_t>(dim), indices_, src_);
  return fromTorchTensor(result_);
}

// Scatter with different shapes (for top-k style operations)
// input: [batch, seq], indices: [batch, k], src: [batch, k] -> [batch, seq]
// batch, seq, k are the implicit shape parameters
lean_object* lean_torch_scatter_2d(uint64_t /*batch*/, uint64_t /*seq*/, uint64_t /*k*/,
    b_lean_obj_arg input, uint64_t dim, b_lean_obj_arg indices, b_lean_obj_arg src) {
  auto input_ = borrowTensor(input);
  auto indices_ = borrowTensor(indices);
  auto src_ = borrowTensor(src);
  auto result_ = input_.clone().scatter_(static_cast<int64_t>(dim), indices_, src_);
  return fromTorchTensor(result_);
}

// Einsum with array of tensors
lean_object* lean_torch_einsum(lean_obj_arg equation, b_lean_obj_arg tensors) {
  auto eq_str = std::string(lean_string_cstr(equation));
  lean_dec(equation);

  std::vector<torch::Tensor> tensor_vec;
  size_t n = lean_array_size(tensors);
  for (size_t i = 0; i < n; i++) {
    tensor_vec.push_back(borrowTensor(lean_array_get_core(tensors, i)));
  }

  auto result_ = torch::einsum(eq_str, tensor_vec);
  return fromTorchTensor(result_);
}

// Einsum with 2 tensors (common case)
lean_object* lean_torch_einsum2(
    lean_obj_arg s1, lean_obj_arg s2,
    lean_obj_arg equation, b_lean_obj_arg a, b_lean_obj_arg b) {
  lean_dec(s1);
  lean_dec(s2);
  OwnedLeanObject equation_owner(equation, lean_dec);
  auto eq_str = std::string(lean_string_cstr(equation));
  equation_owner.reset();
  auto a_ = borrowTensor(a);
  auto b_ = borrowTensor(b);
  auto result_ = torch::einsum(eq_str, {a_, b_});
  return fromTorchTensor(result_);
}

// Interpolate to target size
lean_object* lean_torch_interpolate(
    lean_obj_arg s, b_lean_obj_arg input,
    lean_obj_arg size, lean_obj_arg mode, uint8_t align_corners) {
  lean_dec(s);
  OwnedLeanObject size_owner(size, lean_dec);
  OwnedLeanObject mode_owner(mode, lean_dec);
  auto input_ = borrowTensor(input);
  auto size_ = getShape(size); size_owner.reset();
  auto mode_str = std::string(lean_string_cstr(mode));
  mode_owner.reset();

  torch::nn::functional::InterpolateFuncOptions options;
  options.size(size_);

  if (mode_str == "nearest") options.mode(torch::kNearest);
  else if (mode_str == "linear") options.mode(torch::kLinear);
  else if (mode_str == "bilinear") options.mode(torch::kBilinear);
  else if (mode_str == "bicubic") options.mode(torch::kBicubic);
  else if (mode_str == "trilinear") options.mode(torch::kTrilinear);
  else if (mode_str == "area") options.mode(torch::kArea);
  else options.mode(torch::kNearest);

  if (mode_str != "nearest" && mode_str != "area") {
    options.align_corners(align_corners != 0);
  }

  auto result_ = torch::nn::functional::interpolate(input_, options);
  return fromTorchTensor(result_);
}

// Interpolate with scale factor
lean_object* lean_torch_interpolate_scale(
    lean_obj_arg /*s*/, b_lean_obj_arg input,
    lean_obj_arg scale_factor, lean_obj_arg mode, uint8_t align_corners) {
  auto input_ = borrowTensor(input);
  auto mode_str = std::string(lean_string_cstr(mode));
  lean_dec(mode);

  // Extract scale factors
  std::vector<double> scales;
  size_t n = lean_array_size(scale_factor);
  for (size_t i = 0; i < n; i++) {
    scales.push_back(lean_unbox_float(lean_array_get_core(scale_factor, i)));
  }
  lean_dec(scale_factor);

  torch::nn::functional::InterpolateFuncOptions options;
  options.scale_factor(scales);

  if (mode_str == "nearest") options.mode(torch::kNearest);
  else if (mode_str == "linear") options.mode(torch::kLinear);
  else if (mode_str == "bilinear") options.mode(torch::kBilinear);
  else if (mode_str == "bicubic") options.mode(torch::kBicubic);
  else if (mode_str == "trilinear") options.mode(torch::kTrilinear);
  else if (mode_str == "area") options.mode(torch::kArea);
  else options.mode(torch::kNearest);

  if (mode_str != "nearest" && mode_str != "area") {
    options.align_corners(align_corners != 0);
  }

  auto result_ = torch::nn::functional::interpolate(input_, options);
  return fromTorchTensor(result_);
}

// Clip gradient values element-wise
lean_object* lean_torch_clip_grad_value_(lean_obj_arg s, b_lean_obj_arg param, double clip_value, lean_object* /*w*/) {
  lean_dec(s);
  try {
    torch::NoGradGuard guard;
    auto param_ = borrowTensor(param);
    if (param_.grad().defined()) {
      param_.grad().clamp_(-clip_value, clip_value);
    }
    return lean_io_result_mk_ok(lean_box(0));
  } catch (const c10::Error& e) {
    return mkC10IoError("clip_grad_value_ failed", e);
  } catch (const std::exception& e) {
    return mkStdIoError("clip_grad_value_ failed", e);
  }
}

// ============================================================================
// Sampling utilities for text generation
// ============================================================================

// Top-k filtering: set all logits outside top-k to -infinity
// logits: [..., vocab_size], k: number of top logits to keep
lean_object* lean_torch_topk_filter(
    lean_obj_arg s,
    b_lean_obj_arg logits,
    uint64_t k
) {
  lean_dec(s);
  auto logits_ = borrowTensor(logits);

  if (k == 0) {
    return fromTorchTensor(logits_.clone());
  }
  if (logits_.dim() == 0) {
    return fromTorchTensor(logits_.clone());
  }
  auto vocab_size = logits_.size(-1);
  if (vocab_size <= 0) {
    return fromTorchTensor(logits_.clone());
  }
  auto k_eff = std::min<int64_t>(static_cast<int64_t>(k), vocab_size);
  if (k_eff <= 0 || k_eff >= vocab_size) {
    return fromTorchTensor(logits_.clone());
  }

  // Get top-k values along last dimension
  auto [topk_values, topk_indices] = logits_.topk(k_eff, -1, true, true);

  // Get the minimum value in top-k (threshold)
  auto threshold = std::get<0>(topk_values.min(-1, true));

  // Mask: set logits below threshold to -inf
  auto mask = logits_ < threshold;
  auto result_ = logits_.masked_fill(mask, -std::numeric_limits<float>::infinity());

  return fromTorchTensor(result_);
}

// Top-p (nucleus) filtering: set logits outside cumulative probability p to -infinity
// logits: [..., vocab_size], p: cumulative probability threshold
lean_object* lean_torch_topp_filter(
    lean_obj_arg s,
    b_lean_obj_arg logits,
    double p
) {
  lean_dec(s);
  auto logits_ = borrowTensor(logits);

  if (p >= 1.0) {
    return fromTorchTensor(logits_.clone());
  }

  // Sort logits in descending order
  auto [sorted_logits, sorted_indices] = logits_.sort(-1, true);

  // Compute cumulative softmax probabilities
  auto softmax_sorted = torch::softmax(sorted_logits, -1);
  auto cumulative_probs = softmax_sorted.cumsum(-1);

  // Create mask for tokens to remove (cumulative prob > p, shifted by 1)
  // We shift by 1 so we always keep at least one token
  auto shifted_cumprobs = torch::zeros_like(cumulative_probs);
  if (cumulative_probs.size(-1) > 1) {
    shifted_cumprobs.slice(-1, 1) = cumulative_probs.slice(-1, 0, -1);
  }
  auto sorted_indices_to_remove = shifted_cumprobs > p;

  // Scatter the remove mask back to original positions
  auto indices_to_remove = torch::zeros_like(sorted_indices_to_remove);
  indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove);

  // Set removed positions to -inf
  auto result_ = logits_.masked_fill(indices_to_remove, -std::numeric_limits<float>::infinity());

  return fromTorchTensor(result_);
}

// Squeeze tensor along a specific dimension (with negative index support)
lean_object* lean_torch_squeeze_dim(
    lean_obj_arg /*s*/,
    b_lean_obj_arg input,
    int64_t dim
) {
  auto input_ = borrowTensor(input);
  auto result_ = input_.squeeze(dim);
  return fromTorchTensor(result_);
}

// ============================================================================
// Linear Algebra operations for manifold optimization
// ============================================================================

// QR Decomposition: A = Q @ R
// Returns (Q, R) where Q is orthogonal and R is upper triangular
// For an m×n matrix, returns Q as m×m (complete mode) and R as m×n
lean_object* lean_torch_qr(
    uint64_t /*m*/,
    uint64_t /*n*/,
    b_lean_obj_arg A_obj
) {
  auto A = borrowTensor(A_obj);
  // Use full QR to get m×m Q matrix
  auto [Q, R] = torch::linalg_qr(A, "complete");

  lean_object* result = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(result, 0, fromTorchTensor(Q));
  lean_ctor_set(result, 1, fromTorchTensor(R));
  return result;
}

// Reduced QR: returns Q as m×min(m,n) and R as min(m,n)×n
// More efficient when m > n
lean_object* lean_torch_qr_reduced(
    uint64_t /*m*/,
    uint64_t /*n*/,
    b_lean_obj_arg A_obj
) {
  auto A = borrowTensor(A_obj);
  // Default "reduced" mode
  auto result_tuple = torch::linalg_qr(A);
  auto Q = std::get<0>(result_tuple);
  auto R = std::get<1>(result_tuple);

#ifndef NDEBUG
  // Debug: verify Q is orthogonal
  auto QtQ = torch::mm(Q.t(), Q);
  auto I = torch::eye(Q.size(1), Q.options());
  auto diff = (QtQ - I).abs().max().item<float>();
  if (diff > 1e-4) {
    std::cerr << "WARNING: QR produced non-orthogonal Q! Max diff from I: " << diff << std::endl;
    std::cerr << "A shape: " << A.sizes() << ", Q shape: " << Q.sizes() << std::endl;
  }
#endif

  lean_object* result = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(result, 0, fromTorchTensor(Q));
  lean_ctor_set(result, 1, fromTorchTensor(R));
  return result;
}

// Matrix exponential for square matrices: exp(A)
// Uses Padé approximation internally
lean_object* lean_torch_matrix_exp(
    uint64_t /*n*/,
    b_lean_obj_arg A_obj
) {
  auto A = borrowTensor(A_obj);
  auto result = torch::linalg_matrix_exp(A);
  return fromTorchTensor(result);
}

// Matrix logarithm for square matrices: log(A)
// Implemented via eigendecomposition: log(A) = V @ diag(log(L)) @ V^{-1}
lean_object* lean_torch_matrix_log(
    uint64_t /*n*/,
    b_lean_obj_arg A_obj
) {
  auto A = borrowTensor(A_obj);
  // Compute eigendecomposition: A = V @ diag(L) @ V^{-1}
  auto eig_result = torch::linalg_eig(A);
  auto L = std::get<0>(eig_result);  // eigenvalues (complex)
  auto V = std::get<1>(eig_result);  // eigenvectors (complex)
  // Take log of eigenvalues
  auto log_L = torch::log(L);
  // Reconstruct: log(A) = V @ diag(log_L) @ V^{-1}
  auto V_inv = torch::linalg_inv(V);
  auto result = torch::matmul(torch::matmul(V, torch::diag_embed(log_L)), V_inv);
  auto imag_abs_max = torch::imag(result).abs().max().item<double>();
  auto real_result = torch::real(result);
  if (imag_abs_max > 1e-6) {
    std::cerr << "WARNING: matrix_log produced non-negligible imaginary component (max="
              << imag_abs_max << "); returning NaN tensor." << std::endl;
    auto nan_result = torch::full_like(real_result, std::numeric_limits<double>::quiet_NaN());
    return fromTorchTensor(nan_result);
  }
  return fromTorchTensor(real_result);
}

// Matrix inverse for square matrices: inv(A)
lean_object* lean_torch_inv(
    uint64_t /*n*/,
    b_lean_obj_arg A_obj
) {
  auto A = borrowTensor(A_obj);
  auto result = torch::linalg_inv(A);
  return fromTorchTensor(result);
}

// SVD decomposition: A = U @ diag(S) @ V^T
// Returns (U, S, Vh) where Vh = V^T
lean_object* lean_torch_svd(
    uint64_t /*m*/,
    uint64_t /*n*/,
    b_lean_obj_arg A_obj
) {
  auto A = borrowTensor(A_obj);
  // full_matrices=false gives reduced SVD
  auto result = torch::linalg_svd(A, false);
  auto U = std::get<0>(result);
  auto S = std::get<1>(result);
  auto Vh = std::get<2>(result);

  // Return as nested pairs (U, (S, Vh)): Lean's `A × B × C` is
  // `Prod A (Prod B C)`, i.e. two 2-field ctors, not a flat 3-field ctor.
  lean_object* inner = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(inner, 0, fromTorchTensor(S));
  lean_ctor_set(inner, 1, fromTorchTensor(Vh));
  lean_object* tuple = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(tuple, 0, fromTorchTensor(U));
  lean_ctor_set(tuple, 1, inner);
  return tuple;
}

// SVD values only (singular values): returns just S from A = U @ diag(S) @ V^T
lean_object* lean_torch_svdvals(
    uint64_t /*m*/,
    uint64_t /*n*/,
    b_lean_obj_arg A_obj
) {
  auto A = borrowTensor(A_obj);
  auto S = torch::linalg_svdvals(A);
  return fromTorchTensor(S);
}

// Extract diagonal of a matrix
lean_object* lean_torch_diag(
    uint64_t /*n*/,
    b_lean_obj_arg A_obj
) {
  auto A = borrowTensor(A_obj);
  auto result = torch::diag(A);
  return fromTorchTensor(result);
}

// Create diagonal matrix from vector
lean_object* lean_torch_diagflat(
    uint64_t /*n*/,
    b_lean_obj_arg v_obj
) {
  auto v = borrowTensor(v_obj);
  auto result = torch::diag(v);
  return fromTorchTensor(result);
}

// ============================================================================
// Modular Norm operations
// ============================================================================

// Spectral norm: largest singular value σ_max(A)
double lean_torch_spectral_norm(
    uint64_t /*m*/,
    uint64_t /*n*/,
    b_lean_obj_arg A_obj
) {
  auto A = borrowTensor(A_obj);
  auto sv = torch::linalg_svdvals(A);
  return sv[0].item<double>();
}

// Nuclear norm: sum of singular values Σσᵢ(A)
double lean_torch_nuclear_norm(
    uint64_t /*m*/,
    uint64_t /*n*/,
    b_lean_obj_arg A_obj
) {
  auto A = borrowTensor(A_obj);
  auto sv = torch::linalg_svdvals(A);
  return sv.sum().item<double>();
}

// Row-wise L2 norms: ||a_i||₂ for each row
lean_object* lean_torch_row_norms(
    uint64_t /*n*/,
    uint64_t /*d*/,
    b_lean_obj_arg A_obj
) {
  auto A = borrowTensor(A_obj);
  auto norms = A.norm(2, /*dim=*/1);
  return fromTorchTensor(norms);
}

// Max row norm: max_i ||a_i||₂
double lean_torch_max_row_norm(
    uint64_t /*n*/,
    uint64_t /*d*/,
    b_lean_obj_arg A_obj
) {
  auto A = borrowTensor(A_obj);
  auto norms = A.norm(2, /*dim=*/1);
  return norms.max().item<double>();
}

// L2 norm of a 1D tensor
double lean_torch_l2_norm(
    uint64_t /*n*/,
    b_lean_obj_arg v_obj
) {
  auto v = borrowTensor(v_obj);
  return v.norm(2).item<double>();
}

// Frobenius norm of a matrix
double lean_torch_frobenius_norm(
    uint64_t /*m*/,
    uint64_t /*n*/,
    b_lean_obj_arg A_obj
) {
  auto A = borrowTensor(A_obj);
  return A.norm().item<double>();
}

// Checked native entry points. Legacy pure implementations above are called
// entirely inside the native firewall; they never unwind through Lean code.
lean_object* lean_torch_tensor_add_checked(lean_obj_arg s, b_lean_obj_arg a, b_lean_obj_arg b) {
  return tyr::checkedCall("add", [&] { return lean_torch_tensor_add(s, a, b); });
}

lean_object* lean_torch_tensor_sub_checked(lean_obj_arg s, b_lean_obj_arg a, b_lean_obj_arg b) {
  return tyr::checkedCall("sub", [&] { return lean_torch_tensor_sub(s, a, b); });
}

lean_object* lean_torch_tensor_mul_checked(lean_obj_arg s, b_lean_obj_arg a, b_lean_obj_arg b) {
  return tyr::checkedCall("mul", [&] { return lean_torch_tensor_mul(s, a, b); });
}

lean_object* lean_torch_div_checked(lean_obj_arg s, b_lean_obj_arg a, b_lean_obj_arg b) {
  return tyr::checkedCall("div", [&] { return lean_torch_div(s, a, b); });
}

lean_object* lean_torch_tensor_relu_checked(lean_obj_arg s, b_lean_obj_arg t) {
  return tyr::checkedCall("relu", [&] { return lean_torch_tensor_relu(s, t); });
}

lean_object* lean_torch_abs_checked(lean_obj_arg s, b_lean_obj_arg t) {
  return tyr::checkedCall("abs", [&] { return lean_torch_abs(s, t); });
}

lean_object* lean_torch_expand_checked(lean_obj_arg s, b_lean_obj_arg t, lean_obj_arg target) {
  return tyr::checkedCall("expand", [&] { return lean_torch_expand(s, t, target); });
}

lean_object* lean_torch_reshape_exact_checked(lean_obj_arg s, b_lean_obj_arg t, b_lean_obj_arg target) {
  return tyr::checkedCall("reshape", [&] { return lean_torch_reshape_exact(s, t, target); });
}

lean_object* lean_torch_sum_checked(lean_obj_arg s, b_lean_obj_arg t, lean_obj_arg dims, uint8_t keepdim) {
  return tyr::checkedCall("sum", [&] { return lean_torch_sum(s, t, dims, keepdim); });
}

lean_object* lean_torch_mean_checked(lean_obj_arg s, b_lean_obj_arg t, lean_obj_arg dims, uint8_t keepdim) {
  return tyr::checkedCall("mean", [&] { return lean_torch_mean(s, t, dims, keepdim); });
}

lean_object* lean_torch_max_all_checked(b_lean_obj_arg t) {
  return tyr::checkedCall("maxAll", [&] { return lean_torch_max_all(t); });
}

lean_object* lean_torch_where_checked(lean_obj_arg s, b_lean_obj_arg condition, b_lean_obj_arg a, b_lean_obj_arg b) {
  return tyr::checkedCall("whereSelect", [&] { return lean_torch_where(s, condition, a, b); });
}

lean_object* lean_torch_matmul_checked(lean_obj_arg s1, lean_obj_arg s2, b_lean_obj_arg a, b_lean_obj_arg b) {
  return tyr::checkedCall("matmul", [&] { return lean_torch_matmul(s1, s2, a, b); });
}

lean_object* lean_torch_mm_checked(uint64_t m, uint64_t n, uint64_t k, b_lean_obj_arg a, b_lean_obj_arg b) {
  return tyr::checkedCall("mm", [&] { return lean_torch_mm(m, n, k, a, b); });
}

lean_object* lean_torch_linear_checked(uint64_t m, uint64_t n, uint64_t batch, b_lean_obj_arg a, b_lean_obj_arg b) {
  return tyr::checkedCall("linear", [&] { return lean_torch_linear(m, n, batch, a, b); });
}

lean_object* lean_torch_linear3d_checked(uint64_t batch, uint64_t seq, uint64_t k, uint64_t n, b_lean_obj_arg a, b_lean_obj_arg b) {
  return tyr::checkedCall("linear3d", [&] { return lean_torch_linear3d(batch, seq, k, n, a, b); });
}

lean_object* lean_torch_affine3d_checked(uint64_t batch, uint64_t seq, uint64_t k, uint64_t n,
    b_lean_obj_arg input, b_lean_obj_arg weight, b_lean_obj_arg bias) {
  return tyr::checkedCall("affine3d", [&] {
    return lean_torch_affine3d(batch, seq, k, n, input, weight, bias);
  });
}

lean_object* lean_torch_matmul3d_2d_checked(uint64_t batch, uint64_t seq, uint64_t k, uint64_t n, b_lean_obj_arg a, b_lean_obj_arg b) {
  return tyr::checkedCall("matmul3d", [&] { return lean_torch_matmul3d_2d(batch, seq, k, n, a, b); });
}

lean_object* lean_torch_bmm4d_checked(uint64_t batch, uint64_t heads, uint64_t m, uint64_t k, uint64_t n, b_lean_obj_arg a, b_lean_obj_arg b) {
  return tyr::checkedCall("bmm4d", [&] { return lean_torch_bmm4d(batch, heads, m, k, n, a, b); });
}

lean_object* lean_torch_row_norms_checked(uint64_t m, uint64_t n, b_lean_obj_arg t) {
  return tyr::checkedCall("rowNorms", [&] { return lean_torch_row_norms(m, n, t); });
}

lean_object* lean_torch_svd_checked(uint64_t m, uint64_t n, b_lean_obj_arg t) {
  return tyr::checkedCall("svd", [&] { return lean_torch_svd(m, n, t); });
}

lean_object* lean_torch_svdvals_checked(uint64_t m, uint64_t n, b_lean_obj_arg t) {
  return tyr::checkedCall("svdvals", [&] { return lean_torch_svdvals(m, n, t); });
}

lean_object* lean_torch_spectral_norm_checked(uint64_t m, uint64_t n, b_lean_obj_arg t) {
  return tyr::checkedCall("spectralNorm", [&] { return lean_box_float(lean_torch_spectral_norm(m, n, t)); });
}

lean_object* lean_torch_nuclear_norm_checked(uint64_t m, uint64_t n, b_lean_obj_arg t) {
  return tyr::checkedCall("nuclearNorm", [&] { return lean_box_float(lean_torch_nuclear_norm(m, n, t)); });
}

lean_object* lean_torch_add_scalar_checked(lean_obj_arg s, b_lean_obj_arg t, double value) {
  return tyr::checkedCall("addScalar", [&] { return lean_torch_add_scalar(s, t, value); });
}

lean_object* lean_torch_sub_scalar_checked(lean_obj_arg s, b_lean_obj_arg t, double value) {
  return tyr::checkedCall("subScalar", [&] { return lean_torch_sub_scalar(s, t, value); });
}

lean_object* lean_torch_mul_scalar_checked(lean_obj_arg s, b_lean_obj_arg t, double value) {
  return tyr::checkedCall("mulScalar", [&] { return lean_torch_mul_scalar(s, t, value); });
}

lean_object* lean_torch_div_scalar_checked(lean_obj_arg s, b_lean_obj_arg t, double value) {
  return tyr::checkedCall("divScalar", [&] { return lean_torch_div_scalar(s, t, value); });
}

lean_object* lean_torch_l2_norm_checked(uint64_t n, b_lean_obj_arg t) {
  return tyr::checkedCall("l2Norm", [&] { return lean_box_float(lean_torch_l2_norm(n, t)); });
}

lean_object* lean_torch_frobenius_norm_checked(uint64_t m, uint64_t n, b_lean_obj_arg t) {
  return tyr::checkedCall("frobeniusNorm", [&] { return lean_box_float(lean_torch_frobenius_norm(m, n, t)); });
}

lean_object* lean_torch_max_row_norm_checked(uint64_t m, uint64_t n, b_lean_obj_arg t) {
  return tyr::checkedCall("maxRowNorm", [&] { return lean_box_float(lean_torch_max_row_norm(m, n, t)); });
}

lean_object* lean_torch_mul_tensor_scalar_checked(lean_obj_arg s, b_lean_obj_arg input, b_lean_obj_arg scalar) {
  return tyr::checkedCall("mulTensorScalar", [&] {
    return lean_torch_mul_tensor_scalar(s, input, scalar);
  });
}

lean_object* lean_torch_cat_dyn_checked(b_lean_obj_arg tensors, lean_obj_arg dim) {
  return tyr::checkedCall("catMany", [&] {
    OwnedLeanObject dim_owner(dim, lean_dec);
    TORCH_CHECK(lean_is_array(tensors), "expected array");
    const auto axis = lean_int64_of_int(dim);
    OwnedLeanObject roundtrip(lean_int64_to_int(axis), lean_dec);
    TORCH_CHECK(lean_int_dec_eq(dim, roundtrip.get()), "axis must fit in signed 64 bits");
    std::vector<torch::Tensor> inputs;
    for (size_t i = 0; i < lean_array_size(tensors); ++i)
      inputs.push_back(borrowTensor(lean_array_get_core(tensors, i)));
    return fromTorchTensor(torch::cat(inputs, axis));
  });
}

lean_object* lean_torch_rms_norm_weighted_checked(lean_obj_arg shape, lean_obj_arg weight_shape,
    b_lean_obj_arg input, b_lean_obj_arg weight, double eps) {
  return tyr::checkedCall("rmsNormWeighted", [&] {
    lean_dec(shape);
    lean_dec(weight_shape);
    auto xf = borrowTensor(input).to(torch::kFloat32);
    auto w = borrowTensor(weight);
    auto variance = torch::mean(xf * xf, {std::max<int64_t>(0, xf.dim() - 1)}, true);
    return fromTorchTensor((xf * torch::rsqrt(variance + eps)) * w);
  });
}

lean_object* lean_torch_matrix_log_checked(uint64_t n, b_lean_obj_arg input) {
  return tyr::checkedCall("matrixLog", [&] {
    auto tensor = borrowTensor(input);
    TORCH_CHECK(at::isFloatingType(tensor.scalar_type()), "requires a real floating dtype");
    TORCH_CHECK(torch::isfinite(tensor).all().item<bool>(), "requires finite input");
    auto result = lean_torch_matrix_log(n, input);
    OwnedLeanObject owner(result, lean_dec);
    auto logarithm = borrowTensor(result);
    TORCH_CHECK(torch::isfinite(logarithm).all().item<bool>(),
      "matrix logarithm has no finite real result");
    // Eigenvector reconstruction is not reliable for defective or sufficiently
    // ill-conditioned matrices. A finite result alone is not evidence of a
    // logarithm: e.g. a Jordan block can silently produce the zero matrix.
    // Check a scale-invariant reconstruction residual without retaining an
    // additional autograd graph or changing the returned logarithm's graph.
    auto reference = tensor.detach();
    auto reconstructed = torch::linalg_matrix_exp(logarithm.detach());
    TORCH_CHECK(torch::isfinite(reconstructed).all().item<bool>(),
      "matrix logarithm reconstruction is nonfinite");
    const double scale = reference.abs().max().item<double>();
    TORCH_CHECK(scale > 0, "matrix logarithm requires a nonzero input");
    // Normalize before subtraction, so tiny matrices do not pass an absolute
    // tolerance and large matrices do not overflow while forming the residual.
    const double residual = ((reconstructed / scale) - (reference / scale))
      .abs().max().item<double>();
    const double epsilon = tensor.scalar_type() == torch::kFloat64
      ? std::numeric_limits<double>::epsilon()
      : std::numeric_limits<float>::epsilon();
    const double tolerance = 64.0 * std::max<int64_t>(1, tensor.size(0)) * epsilon;
    TORCH_CHECK(std::isfinite(residual) && residual <= tolerance,
      "eigendecomposition cannot reliably compute this matrix logarithm: "
      "scaled reconstruction residual ", residual, " exceeds ", tolerance);
    return owner.release();
  });
}

lean_object* lean_torch_tensor_exp_checked(lean_obj_arg s, b_lean_obj_arg t) {
  return tyr::checkedCall("exp", [&] { return lean_torch_tensor_exp(s, t); });
}

lean_object* lean_torch_sqrt_checked(lean_obj_arg s, b_lean_obj_arg t) {
  return tyr::checkedCall("sqrt", [&] { return lean_torch_sqrt(s, t); });
}

lean_object* lean_torch_rsqrt_checked(lean_obj_arg s, b_lean_obj_arg t) {
  return tyr::checkedCall("rsqrt", [&] { return lean_torch_rsqrt(s, t); });
}

lean_object* lean_torch_tensor_sigmoid_checked(lean_obj_arg s, b_lean_obj_arg t) {
  return tyr::checkedCall("sigmoid", [&] { return lean_torch_tensor_sigmoid(s, t); });
}

lean_object* lean_torch_tensor_tanh_checked(lean_obj_arg s, b_lean_obj_arg t) {
  return tyr::checkedCall("tanh", [&] { return lean_torch_tensor_tanh(s, t); });
}

lean_object* lean_torch_tensor_gelu_checked(lean_obj_arg s, b_lean_obj_arg t) {
  return tyr::checkedCall("gelu", [&] { return lean_torch_tensor_gelu(s, t); });
}

lean_object* lean_torch_tensor_silu_checked(lean_obj_arg s, b_lean_obj_arg t) {
  return tyr::checkedCall("silu", [&] { return lean_torch_tensor_silu(s, t); });
}

lean_object* lean_torch_pow_checked(lean_obj_arg s, b_lean_obj_arg t, double value) {
  return tyr::checkedCall("pow", [&] { return lean_torch_pow(s, t, value); });
}

lean_object* lean_torch_tensor_atan_checked(lean_obj_arg s, b_lean_obj_arg t) {
  return tyr::checkedCall("atan", [&] { return lean_torch_tensor_atan(s, t); });
}

lean_object* lean_torch_tensor_softmax_checked(lean_obj_arg s, b_lean_obj_arg t, uint32_t dim) {
  return tyr::checkedCall("softmax", [&] { return lean_torch_tensor_softmax(s, t, dim); });
}

lean_object* lean_torch_log_softmax_checked(b_lean_obj_arg t, uint32_t dim) {
  return tyr::checkedCall("logSoftmax", [&] { return lean_torch_log_softmax(t, dim); });
}

lean_object* lean_torch_tensor_log_checked(lean_obj_arg s, b_lean_obj_arg t) {
  return tyr::checkedCall("log", [&] { return lean_torch_tensor_log(s, t); });
}

lean_object* lean_torch_tensor_log10_checked(lean_obj_arg s, b_lean_obj_arg t) {
  return tyr::checkedCall("log10", [&] { return lean_torch_tensor_log10(s, t); });
}

lean_object* lean_torch_tensor_sin_checked(lean_obj_arg s, b_lean_obj_arg t) {
  return tyr::checkedCall("sin", [&] { return lean_torch_tensor_sin(s, t); });
}

lean_object* lean_torch_tensor_cos_checked(lean_obj_arg s, b_lean_obj_arg t) {
  return tyr::checkedCall("cos", [&] { return lean_torch_tensor_cos(s, t); });
}

lean_object* lean_torch_tensor_elu_checked(lean_obj_arg s, b_lean_obj_arg t) {
  return tyr::checkedCall("elu", [&] { return lean_torch_tensor_elu(s, t); });
}

lean_object* lean_torch_maximum_checked(lean_obj_arg s, b_lean_obj_arg a, b_lean_obj_arg b) {
  return tyr::checkedCall("maximum", [&] {
    lean_dec(s);
    return fromTorchTensor(torch::maximum(borrowTensor(a), borrowTensor(b)));
  });
}

lean_object* lean_torch_gt_checked(lean_obj_arg s, b_lean_obj_arg a, b_lean_obj_arg b) {
  return tyr::checkedCall("gt", [&] { return lean_torch_gt(s, a, b); });
}

lean_object* lean_torch_logical_and_checked(lean_obj_arg s, b_lean_obj_arg a, b_lean_obj_arg b) {
  return tyr::checkedCall("logicalAnd", [&] { return lean_torch_logical_and(s, a, b); });
}

lean_object* lean_torch_cat2_checked(lean_obj_arg s1, lean_obj_arg s2,
    b_lean_obj_arg a, b_lean_obj_arg b, lean_obj_arg dim) {
  return tyr::checkedCall("cat", [&] {
    lean_dec(s1);
    lean_dec(s2);
    OwnedLeanObject dim_owner(dim, lean_dec);
    const auto axis = lean_int64_of_int(dim);
    OwnedLeanObject roundtrip(lean_int64_to_int(axis), lean_dec);
    TORCH_CHECK(lean_int_dec_eq(dim, roundtrip.get()), "axis must fit in signed 64 bits");
    return fromTorchTensor(torch::cat({borrowTensor(a), borrowTensor(b)}, axis));
  });
}

lean_object* lean_torch_lt_checked(lean_obj_arg s, b_lean_obj_arg a, b_lean_obj_arg b) {
  return tyr::checkedCall("lt", [&] { return lean_torch_lt(s, a, b); });
}

lean_object* lean_torch_ge_checked(lean_obj_arg s, b_lean_obj_arg a, b_lean_obj_arg b) {
  return tyr::checkedCall("ge", [&] { return lean_torch_ge(s, a, b); });
}

}
