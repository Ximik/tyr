#include "generated/tyr_ffi.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

extern "C" void lean_initialize_runtime_module();

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

Object samples(size_t count) {
  auto* array = lean_alloc_array(count, count);
  for (size_t i = 0; i < count; ++i)
    lean_array_set_core(array, i, lean_box_float(0.25));
  return Object(array, lean_dec);
}

void expectError(lean_object* result) {
  Object owned(result, lean_dec);
  require(lean_io_result_is_error(result), "expected IO.error");
}

void expectSuccess(lean_object* result) {
  Object owned(result, lean_dec);
  require(lean_io_result_is_ok(result), "expected IO.ok");
}

bool userErrorContains(lean_object* result, const char* fragment) {
  if (!lean_io_result_is_error(result)) return false;
  auto* error = lean_io_result_get_error(result);
  Object sample(lean_mk_io_user_error(lean_mk_string("")), lean_dec);
  if (lean_obj_tag(error) != lean_obj_tag(sample.get())) return false;
  auto* message = lean_ctor_get(error, 0);
  return lean_is_string(message) &&
    std::string(lean_string_cstr(message)).find(fragment) != std::string::npos;
}

void resampleSuccess(uint64_t from, uint64_t to, size_t count, size_t expected) {
  auto input = samples(count);
  require(lean_is_exclusive(input.get()), "input should initially have one owner");
  {
    Object result(lean_torch_resample_soxr_hq(input.get(), Nat(from), Nat(to)), lean_dec);
    require(lean_io_result_is_ok(result.get()), "resampling returned IO.error");
    auto* output = lean_io_result_get_value(result.get());
    require(lean_array_size(output) == expected, "unexpected resampled length");
    for (size_t i = 0; i < expected; ++i)
      require(std::isfinite(lean_unbox_float(lean_array_get_core(output, i))),
        "nonfinite resampled output");
    if (from == to || count == 0) {
      require(output == input.get(), "fast path should retain the immutable input array");
      require(!lean_is_exclusive(input.get()), "returned borrowed array was not retained");
    }
    if (count) require(lean_unbox_float(lean_array_get_core(input.get(), 0)) == 0.25,
      "borrowed input changed during resampling");
  }
  require(lean_is_exclusive(input.get()), "resampling leaked an input reference");
}

} // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  lean_initialize_runtime_module();
  size_t passed = 0, failed = 0, skipped = 0;
  auto test = [&](const char* name, const std::function<void()>& run) {
    try {
      run();
      ++passed;
      std::printf("PASS %s\n", name);
    } catch (const std::exception& error) {
      ++failed;
      std::printf("FAIL %s: native exception escaped or assertion failed: %s\n", name, error.what());
    } catch (...) {
      ++failed;
      std::printf("FAIL %s: unknown native exception escaped\n", name);
    }
  };

  // This baseline fails with std::length_error without attempting a huge allocation.
  test("oversized resampler output is IO.error", [] {
    auto input = samples(1);
    expectError(lean_torch_resample_soxr_hq(input.get(), Nat(1), Nat(uint64_t{1} << 63)));
    require(lean_is_exclusive(input.get()), "error path retained borrowed input");
  });
#ifndef TYR_TORCH_BASELINE
  // The old unchecked double-to-size_t conversion is undefined for these inputs.
  // Exercise them only after the bounds guard is installed, not in the baseline.
  test("resampler UInt64 rate rounding overflow is IO.error", [] {
    auto input = samples(1);
    expectError(lean_torch_resample_soxr_hq(input.get(), Nat(1), Nat(std::numeric_limits<uint64_t>::max())));
  });
  test("resampler length-times-ratio overflow is IO.error", [] {
    auto input = samples(2);
    expectError(lean_torch_resample_soxr_hq(input.get(), Nat(1), Nat(uint64_t{1} << 63)));
  });
  test("resampler rejects unrepresentable Lean array capacity", [] {
    auto input = samples(1);
    const auto tooMany = (std::numeric_limits<size_t>::max() -
      sizeof(lean_array_object)) / sizeof(void*) + 1;
    Object result(lean_torch_resample_soxr_hq(input.get(), Nat(1), Nat(tooMany)), lean_dec);
    require(userErrorContains(result.get(), "output length exceeds the native array limit"),
      "unrepresentable Lean array capacity must be rejected before allocation");
  });
#endif
  test("same-rate resampler retains borrowed input", [] { resampleSuccess(24000, 24000, 16, 16); });
  test("empty resampler retains borrowed input", [] { resampleSuccess(48000, 24000, 0, 0); });
  test("valid downsample", [] { resampleSuccess(48000, 24000, 16, 8); });
  test("non-integral resample length rounds upward", [] { resampleSuccess(2, 3, 3, 5); });
  test("zero source rate is IO.error", [] {
    auto input = samples(1);
    expectError(lean_torch_resample_soxr_hq(input.get(), Nat(0), Nat(24000)));
  });
  test("zero target rate is IO.error", [] {
    auto input = samples(1);
    expectError(lean_torch_resample_soxr_hq(input.get(), Nat(24000), Nat(0)));
  });
  test("missing token file is IO.error", [] {
    Object path(lean_mk_string("/tyr-nonexistent-runtime-test-directory/tokens.bin"), lean_dec);
    expectError(lean_torch_bin_file_token_count(path.get()));
    require(lean_is_exclusive(path.get()), "borrowed path reference leaked");
  });
  test("missing fileExists is false", [] {
    Object path(lean_mk_string("/tyr-nonexistent-runtime-test-directory/file"), lean_dec);
    Object result(lean_torch_file_exists(path.get()), lean_dec);
    require(lean_io_result_is_ok(result.get()), "fileExists returned IO.error");
    require(lean_unbox(lean_io_result_get_value(result.get())) == 0, "missing path exists");
  });
  test("missing WAV directory is IO.error", [] {
    Object path(lean_mk_string("/tyr-nonexistent-runtime-test-directory/audio.wav"), lean_dec);
    expectError(lean_torch_wav_begin(path.get(), lean_box(24000)));
    expectError(lean_torch_wav_finalize(path.get()));
  });
  test("CPU RNG seed and live tensor count carriers", [] {
    expectSuccess(lean_torch_manual_seed(123));
    Object result(lean_torch_get_live_tensors(), lean_dec);
    require(lean_io_result_is_ok(result.get()), "live tensor count returned IO.error");
    (void)lean_unbox_uint64(lean_io_result_get_value(result.get()));
  });

  bool cudaUnavailable = false;
  test("CUDA and MPS availability carriers", [&] {
    Object cuda(lean_torch_cuda_is_available(), lean_dec);
    require(lean_io_result_is_ok(cuda.get()), "CUDA availability returned IO.error");
    const auto cudaValue = lean_io_result_get_value(cuda.get());
    require(lean_is_scalar(cudaValue) && lean_unbox(cudaValue) <= 1, "CUDA availability is not Bool");
    Object mps(lean_torch_mps_is_available(), lean_dec);
    require(lean_io_result_is_ok(mps.get()), "MPS availability returned IO.error");
    const auto mpsValue = lean_io_result_get_value(mps.get());
    require(lean_is_scalar(mpsValue) && lean_unbox(mpsValue) <= 1, "MPS availability is not Bool");
    cudaUnavailable = lean_unbox(cudaValue) == 0;
  });
  if (cudaUnavailable) {
    test("unavailable CUDA stream and synchronization", [] {
      Object stream(lean_torch_cuda_current_stream(), lean_dec);
      require(lean_io_result_is_ok(stream.get()), "unavailable CUDA stream returned IO.error");
      require(lean_unbox_uint64(lean_io_result_get_value(stream.get())) == 0,
        "unavailable CUDA stream should be zero");
      expectSuccess(lean_torch_cuda_synchronize());
    });
    bool eventStub = false;
    test("unavailable CUDA event creation", [&] {
      Object event(lean_torch_cuda_event_create(), lean_dec);
      if (lean_io_result_is_ok(event.get())) {
        // Clean up any genuine unexpected handle; never manufacture a GPU handle.
        expectSuccess(lean_torch_cuda_event_destroy(lean_unbox_uint64(lean_io_result_get_value(event.get()))));
        throw std::runtime_error("CUDA event created while CUDA unavailable");
      }
      eventStub = userErrorContains(event.get(), "requires the CUDA runtime API");
    });
    if (eventStub) {
      test("CPU CUDA-event stubs return correct carriers", [] {
        expectError(lean_torch_cuda_event_record(0, 0));
        expectError(lean_torch_cuda_event_synchronize(0));
        expectError(lean_torch_cuda_event_elapsed_ms(0, 0));
        expectSuccess(lean_torch_cuda_event_destroy(0));
      });
    } else {
      ++skipped;
      std::puts("SKIP event handle calls: native CUDA runtime is present; no fake handles tested");
    }
    bool graphStub = false;
    test("unavailable CUDA graph begin", [&] {
      Object graph(lean_torch_cuda_graph_capture_begin(), lean_dec);
      require(lean_io_result_is_error(graph.get()), "CUDA graph began while CUDA unavailable");
      graphStub = userErrorContains(graph.get(), "CUDA graphs not available in this libtorch build");
    });
    if (graphStub) {
      test("CPU CUDA-graph stubs return correct carriers", [] {
        expectError(lean_torch_cuda_graph_capture_end());
        expectError(lean_torch_cuda_graph_replay());
        expectSuccess(lean_torch_cuda_graph_reset());
      });
    } else {
      ++skipped;
      std::puts("SKIP CUDA graph lifecycle: CUDA-enabled graph state repair is a separate task");
    }
  } else {
    ++skipped;
    std::puts("SKIP CPU-only CUDA/event/graph cases: CUDA available or availability check failed");
  }

  std::printf("Torch native recovery: %zu passed, %zu failed, %zu skipped\n", passed, failed, skipped);
  return failed ? 1 : 0;
}
