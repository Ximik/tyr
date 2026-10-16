// Check the C++ runtime before a harness deliberately triggers native errors.
// In particular, exercise libc10's error backtrace and libtorch's throw paths,
// not just a throw/catch pair compiled in this translation unit.
#include <ATen/Functions.h>
#include <c10/util/Exception.h>
#include <lean/lean.h>
#include "../src/generated/tyr_ffi.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#ifdef __linux__
#include <dlfcn.h>
#endif

extern "C" void lean_initialize_runtime_module();

static bool symbolProviders() {
#ifdef __linux__
  for (const char* symbol : {"__cxa_throw", "__cxa_begin_catch",
                              "_Unwind_RaiseException", "_Unwind_Backtrace"}) {
    void* address = dlsym(RTLD_DEFAULT, symbol);
    Dl_info provider{};
    if (!address || !dladdr(address, &provider) || !provider.dli_fname) {
      std::fprintf(stderr, "Could not identify native exception symbol %s\n", symbol);
      return false;
    }
    std::printf("[exception-runtime] %s: %s\n", symbol, provider.dli_fname);
    // Both the system-built harness and pinned Linux libtorch use libstdc++.
    // Do not accept Lean's bundled runtime as their C++ exception provider.
    if (std::strncmp(symbol, "__cxa_", 6) == 0 &&
        !std::strstr(provider.dli_fname, "libstdc++")) {
      std::fprintf(stderr, "System C++ exception runtime must precede Lean: %s\n", symbol);
      return false;
    }
    if (std::strncmp(symbol, "_Unwind_", 8) == 0 &&
        !std::strstr(provider.dli_fname, "libgcc_s")) {
      std::fprintf(stderr, "System unwinder must precede Lean: %s\n", symbol);
      return false;
    }
  }
#endif
  return true;
}

static bool checkExceptions() {
  std::puts("[exception-runtime] std::runtime_error throw/catch");
  try {
    throw std::runtime_error("tyr runtime smoke");
  } catch (const std::runtime_error& error) {
    if (std::strcmp(error.what(), "tyr runtime smoke") != 0) return false;
  }
  std::puts("[exception-runtime] libc10 throw/catch and backtrace");
  bool caught = false;
  try {
    TORCH_CHECK(false, "tyr c10 smoke");
  } catch (const c10::Error& error) {
    // what() forces backtrace materialization; the failing metadata test also
    // uses it to check the expected device error text.
    caught = std::strstr(error.what_without_backtrace(), "tyr c10 smoke") &&
             std::strstr(error.what(), "tyr c10 smoke");
  }
  if (!caught) return false;
  std::puts("[exception-runtime] libtorch invalid-shape throw/catch");
  caught = false;
  try {
    (void)at::empty({-1});
  } catch (const c10::Error& error) {
    caught = std::strstr(error.what(), "negative dimension") != nullptr;
  }
  return caught;
}

int main() {
  // Flush phase markers even if a runtime conflict aborts or exhausts stack.
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (!symbolProviders()) return 1;
  lean_initialize_runtime_module();
  // Keep libTyrC in DT_NEEDED even with --as-needed. Its static constructors
  // and runtime dependencies must be present just as in the real harness.
  lean_object* live = lean_torch_get_live_tensors();
  const bool live_ok = lean_io_result_is_ok(live);
  lean_dec(live);
  if (!live_ok) return 1;
  try {
    if (!checkExceptions()) {
      std::fputs("Native exception runtime smoke check failed\n", stderr);
      return 1;
    }
  } catch (...) {
    std::fputs("Native exception runtime smoke check escaped its expected catch\n", stderr);
    return 1;
  }
  std::puts("[exception-runtime] all throw/catch checks passed");
  return 0;
}
