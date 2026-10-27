// Unit mode compiles the real implementation with local symbol names and fake
// Work/Backend state. It neither interposes library exports nor opens sockets.
#ifdef TYR_DISTRIBUTED_UNIT
#define lean_torch_dist_wait unit_lean_torch_dist_wait
#define lean_torch_dist_init_process_group unit_lean_torch_dist_init_process_group
#define lean_torch_dist_set_cuda_device unit_lean_torch_dist_set_cuda_device
#define lean_torch_dist_get_rank unit_lean_torch_dist_get_rank
#define lean_torch_dist_get_world_size unit_lean_torch_dist_get_world_size
#define lean_torch_dist_is_initialized unit_lean_torch_dist_is_initialized
#define lean_torch_dist_destroy_process_group unit_lean_torch_dist_destroy_process_group
#define lean_torch_dist_barrier unit_lean_torch_dist_barrier
#define lean_torch_dist_all_reduce unit_lean_torch_dist_all_reduce
#define lean_torch_dist_broadcast unit_lean_torch_dist_broadcast
#define lean_torch_dist_reduce_scatter_tensor unit_lean_torch_dist_reduce_scatter_tensor
#define lean_torch_dist_reduce_scatter_list unit_lean_torch_dist_reduce_scatter_list
#define lean_torch_dist_all_gather_into_tensor unit_lean_torch_dist_all_gather_into_tensor
#define lean_torch_dist_all_gather_list unit_lean_torch_dist_all_gather_list
#define lean_torch_dist_is_completed unit_lean_torch_dist_is_completed
#define lean_torch_xxt unit_lean_torch_xxt
#define lean_torch_ba_plus_cAA unit_lean_torch_ba_plus_cAA
#define lean_torch_newton_schulz_step unit_lean_torch_newton_schulz_step
#define lean_torch_polar_express unit_lean_torch_polar_express
#define lean_torch_muon_orthogonalize unit_lean_torch_muon_orthogonalize
#define lean_torch_cautious_update unit_lean_torch_cautious_update
#define borrowTensor unitBorrowTensor
#define fromTorchTensor unitFromTorchTensor
#include "../src/tyr_distributed.cpp"
#include "../src/tyr_polar.cpp"
#undef borrowTensor
#undef fromTorchTensor
#include <torch/csrc/distributed/c10d/HashStore.hpp>

// Local handles are used only by the included implementation. Real ABI mode
// constructs actual library handles through the checked allocation export.
lean_object* unitFromTorchTensor(torch::Tensor tensor) {
  static auto* cls = lean_register_external_class(
    [](void* value) { delete static_cast<torch::Tensor*>(value); },
    [](void*, b_lean_obj_arg) {});
  return lean_alloc_external(cls, new torch::Tensor(std::move(tensor)));
}
torch::Tensor unitBorrowTensor(b_lean_obj_arg value) {
  return *static_cast<torch::Tensor*>(lean_get_external_data(value));
}
#endif

#include "generated/tyr_ffi.h"
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

extern "C" void lean_initialize_runtime_module();

#ifdef TYR_DISTRIBUTED_BASELINE
// Baseline-only aliases match the old library's actual ABI. Do not invoke the
// corrected one-argument declaration against its obsolete world parameter.
#ifdef __APPLE__
#define TYR_SYMBOL(name) "_" name
#else
#define TYR_SYMBOL(name) name
#endif
extern "C" lean_object* baselineSetDevice(uint64_t, lean_object*)
  asm(TYR_SYMBOL("lean_torch_dist_set_cuda_device"));
extern "C" lean_object* baselineInitialized(lean_object*)
  asm(TYR_SYMBOL("lean_torch_dist_is_initialized"));
#endif

namespace {
using Object = std::unique_ptr<lean_object, decltype(&lean_dec)>;
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void expectError(lean_object* result) {
  Object owned(result, lean_dec);
  require(lean_io_result_is_error(result), "expected IO.error");
}
void expectSuccess(lean_object* result) {
  Object owned(result, lean_dec);
  require(lean_io_result_is_ok(result), "expected IO.ok");
}
bool initialized() {
#ifdef TYR_DISTRIBUTED_BASELINE
  Object result(baselineInitialized(lean_box(0)), lean_dec);
#else
  Object result(lean_torch_dist_is_initialized(), lean_dec);
#endif
  require(lean_io_result_is_ok(result.get()), "initialized query failed");
  auto* value = lean_io_result_get_value(result.get());
  require(lean_is_scalar(value) && lean_unbox(value) <= 1, "invalid Bool carrier");
  return lean_unbox(value) != 0;
}
lean_object* setDevice(uint64_t index) {
#ifdef TYR_DISTRIBUTED_BASELINE
  return baselineSetDevice(index, lean_box(0));
#else
  return lean_torch_dist_set_cuda_device(index);
#endif
}
lean_object* shape(std::initializer_list<uint64_t> dims) {
  auto* result = lean_alloc_array(dims.size(), dims.size());
  size_t i = 0;
  for (auto dim : dims) lean_array_set_core(result, i++, lean_box_uint64(dim));
  return result;
}
Object tensor(std::initializer_list<uint64_t> dims) {
  // Device.CPU is constructor 1; constructor 0 is CUDA with an index payload.
  return Object(lean_torch_ones(shape(dims), 0, lean_box(1)), lean_dec);
}
} // namespace

#ifdef TYR_DISTRIBUTED_UNIT
namespace {
void clearFakeState();
struct Counts {
  int waits = 0;
  int destroyed = 0;
  // Assertions must not leave a global Work referring to dead stack counters.
  ~Counts() { clearFakeState(); }
};
struct FakeWork final : c10d::Work {
  Counts& counts;
  int waitError;
  int pollError;
  bool complete;
  bool waitSucceeded;
  bool successful;
  bool hasException;
  FakeWork(Counts& counts, int waitError = 0, int pollError = 0, bool complete = true,
      bool waitSucceeded = true, bool successful = true, bool hasException = true)
    : counts(counts), waitError(waitError), pollError(pollError), complete(complete),
      waitSucceeded(waitSucceeded), successful(successful), hasException(hasException) {}
  ~FakeWork() override { ++counts.destroyed; }
  static void fail(int kind) {
    if (kind == 1) throw std::runtime_error("injected standard Work exception");
    if (kind == 2) throw 17;
  }
  bool wait(std::chrono::milliseconds = kNoTimeout) override {
    ++counts.waits; fail(waitError); return waitSucceeded;
  }
  bool isCompleted() override { fail(pollError); return complete; }
  bool isSuccess() const override { return successful; }
  std::exception_ptr exception() const override {
    return successful || !hasException ? nullptr :
      std::make_exception_ptr(std::runtime_error("injected failed Work"));
  }
};
struct FakeBackend final : c10d::Backend {
  Counts& counts;
  int broadcasts = 0;
  bool waitSucceeded;
  explicit FakeBackend(Counts& counts, bool waitSucceeded = true)
    : Backend(0, 1), counts(counts), waitSucceeded(waitSucceeded) {}
  c10::intrusive_ptr<c10d::Work> work() {
    return c10::make_intrusive<FakeWork>(counts, 0, 0, true, waitSucceeded);
  }
  const std::string getBackendName() const override { return "fake-unit-only"; }
  c10::intrusive_ptr<c10d::Work> allreduce(std::vector<at::Tensor>&,
      const c10d::AllreduceOptions&) override { return work(); }
  c10::intrusive_ptr<c10d::Work> broadcast(std::vector<at::Tensor>&,
      const c10d::BroadcastOptions&) override {
    ++broadcasts; return work();
  }
  c10::intrusive_ptr<c10d::Work> reduce_scatter(std::vector<at::Tensor>&,
      std::vector<std::vector<at::Tensor>>&, const c10d::ReduceScatterOptions&) override {
    return work();
  }
  c10::intrusive_ptr<c10d::Work> allgather(std::vector<std::vector<at::Tensor>>&,
      std::vector<at::Tensor>&, const c10d::AllgatherOptions&) override {
    return work();
  }
};
uint64_t uintResult(lean_object* result) {
  Object owned(result, lean_dec);
  require(lean_io_result_is_ok(result), "expected IO.ok UInt64");
  auto* payload = lean_io_result_get_value(result);
  require(!lean_is_scalar(payload), "UInt64 success payload was an immediate Unit");
  return lean_unbox_uint64(payload);
}
void clearFakeState() {
  g_pending_work.clear(); g_process_group.reset(); g_store.reset();
  g_initialized = false; g_rank = 0; g_world_size = 1;
}
} // namespace
#endif

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  lean_initialize_runtime_module();
  size_t passed = 0, failed = 0, skipped = 0;
  auto test = [&](const char* name, const std::function<void()>& run) {
    try { run(); ++passed; std::printf("PASS %s\n", name); }
    catch (const std::exception& e) {
      ++failed; std::printf("FAIL %s: %s\n", name, e.what());
    } catch (...) { ++failed; std::printf("FAIL %s: native exception escaped\n", name); }
  };
#ifdef TYR_DISTRIBUTED_UNIT
  std::puts("Unit-only fake Work/Backend tests; no networking or Gloo qualification");
  test("three synchronous UInt64 zero carriers", [] {
    Counts counts;
    g_process_group = c10::make_intrusive<FakeBackend>(counts);
    g_initialized = true;
    Object raw(unitFromTorchTensor(torch::ones({2, 2})), lean_dec);
    require(uintResult(lean_torch_dist_all_reduce(raw.get(), 0, 0)) == 0, "allreduce ID");
    require(uintResult(lean_torch_dist_reduce_scatter_tensor(raw.get(), raw.get(), 0, 0)) == 0,
      "reduce-scatter ID");
    require(uintResult(lean_torch_dist_all_gather_into_tensor(raw.get(), raw.get(), 0)) == 0,
      "all-gather ID");
    require(counts.waits == 3 && counts.destroyed == 3, "synchronous work lifetime");
    require(lean_is_exclusive(raw.get()), "borrowed tensor reference changed");
    require(uintResult(lean_torch_dist_get_rank()) == 0, "rank carrier");
    require(uintResult(lean_torch_dist_get_world_size()) == 1, "world size carrier");
    clearFakeState();
  });
  test("broadcast rejects narrowing before backend dispatch", [] {
    Counts counts;
    auto backend = c10::make_intrusive<FakeBackend>(counts);
    g_process_group = backend; g_initialized = true;
    Object raw(unitFromTorchTensor(torch::ones({2})), lean_dec);
    expectError(lean_torch_dist_broadcast(raw.get(), UINT64_MAX));
    expectError(lean_torch_dist_broadcast(raw.get(), 1));
    require(backend->broadcasts == 0, "invalid rank reached backend");
    expectSuccess(lean_torch_dist_broadcast(raw.get(), 0));
    require(backend->broadcasts == 1, "valid broadcast was rejected");
    clearFakeState();
  });
  test("async work retained until wait and released once", [] {
    Counts counts;
    g_process_group = c10::make_intrusive<FakeBackend>(counts); g_initialized = true;
    Object raw(unitFromTorchTensor(torch::ones({2})), lean_dec);
    const auto id = uintResult(lean_torch_dist_all_reduce(raw.get(), 0, 1));
    require(id != 0 && g_pending_work.size() == 1 && counts.destroyed == 0, "pending work not retained");
    expectSuccess(lean_torch_dist_wait(id));
    require(g_pending_work.empty() && counts.waits == 1 && counts.destroyed == 1, "wait lifetime");
    expectError(lean_torch_dist_wait(id));
    clearFakeState();
  });
  test("false async wait returns error and releases once", [] {
    Counts counts;
    g_pending_work[40] = c10::make_intrusive<FakeWork>(counts, 0, 0, true, false);
    expectError(lean_torch_dist_wait(40));
    require(g_pending_work.empty() && counts.waits == 1 && counts.destroyed == 1,
      "aborted wait did not release work once");
  });
  test("false synchronous waits return errors", [] {
    Counts counts;
    g_process_group = c10::make_intrusive<FakeBackend>(counts, false); g_initialized = true;
    Object raw(unitFromTorchTensor(torch::ones({2, 2})), lean_dec);
    expectError(lean_torch_dist_all_reduce(raw.get(), 0, 0));
    expectError(lean_torch_dist_reduce_scatter_tensor(raw.get(), raw.get(), 0, 0));
    expectError(lean_torch_dist_all_gather_into_tensor(raw.get(), raw.get(), 0));
    require(counts.waits == 3 && counts.destroyed == 3, "aborted synchronous work lifetime");
  });
  test("false teardown wait still drains all and resets", [] {
    Counts counts;
    g_process_group = c10::make_intrusive<FakeBackend>(counts); g_initialized = true;
    g_pending_work[1] = c10::make_intrusive<FakeWork>(counts, 0, 0, true, false);
    g_pending_work[2] = c10::make_intrusive<FakeWork>(counts);
    expectError(lean_torch_dist_destroy_process_group());
    require(counts.waits == 2 && counts.destroyed == 2, "aborted teardown skipped other work");
    require(!g_initialized && !g_process_group && g_pending_work.empty(), "aborted teardown retained state");
  });
  for (int error : {1, 2}) {
    test(error == 1 ? "standard wait exception contained and erased" : "unknown wait exception contained and erased", [=] {
      Counts counts;
      g_pending_work[41] = c10::make_intrusive<FakeWork>(counts, error);
      expectError(lean_torch_dist_wait(41));
      require(g_pending_work.empty() && counts.waits == 1 && counts.destroyed == 1, "failed wait leaked work");
      clearFakeState();
    });
    test(error == 1 ? "standard poll exception contained" : "unknown poll exception contained", [=] {
      Counts counts;
      g_pending_work[42] = c10::make_intrusive<FakeWork>(counts, 0, error);
      expectError(lean_torch_dist_is_completed(42));
      require(g_pending_work.size() == 1, "failed poll lost retry/drain handle");
      expectSuccess(lean_torch_dist_destroy_process_group());
      require(counts.waits == 1 && counts.destroyed == 1, "failed poll was not drainable");
      clearFakeState();
    });
    test(error == 1 ? "standard teardown failure drains all and resets" : "unknown teardown failure drains all and resets", [=] {
      Counts counts;
      g_process_group = c10::make_intrusive<FakeBackend>(counts); g_initialized = true;
      auto store = c10::make_intrusive<c10d::HashStore>(); g_store = store;
      g_rank = 2; g_world_size = 3;
      g_pending_work[1] = c10::make_intrusive<FakeWork>(counts, error);
      g_pending_work[2] = c10::make_intrusive<FakeWork>(counts);
      g_pending_work[3] = c10::make_intrusive<FakeWork>(counts, error);
      expectError(lean_torch_dist_destroy_process_group());
      require(counts.waits == 3 && counts.destroyed == 3, "teardown stopped after failure");
      require(!g_initialized && !g_process_group && !g_store && g_pending_work.empty(), "teardown retained state");
      require(g_rank == 0 && g_world_size == 1 && store.use_count() == 1, "teardown metadata/store leak");
      expectSuccess(lean_torch_dist_destroy_process_group());
      clearFakeState();
    });
  }
  test("idle teardown releases residual store and work", [] {
    Counts counts;
    auto store = c10::make_intrusive<c10d::HashStore>(); g_store = store;
    g_pending_work[7] = c10::make_intrusive<FakeWork>(counts);
    require(!g_initialized, "fixture should be idle");
    expectSuccess(lean_torch_dist_destroy_process_group());
    require(!g_store && store.use_count() == 1 && g_pending_work.empty(), "residual resources retained");
    require(counts.waits == 1 && counts.destroyed == 1, "residual work not drained");
  });
  test("poll is observational and successful poll then wait consumes once", [] {
    Counts counts;
    auto work = c10::make_intrusive<FakeWork>(counts, 0, 0, false);
    g_pending_work[9] = work;
    Object pending(lean_torch_dist_is_completed(9), lean_dec);
    require(lean_io_result_is_ok(pending.get()) &&
      lean_unbox(lean_io_result_get_value(pending.get())) == 0, "pending Bool carrier");
    require(work.use_count() == 2, "pending ownership changed");
    work->complete = true;
    Object complete(lean_torch_dist_is_completed(9), lean_dec);
    require(lean_io_result_is_ok(complete.get()) &&
      lean_unbox(lean_io_result_get_value(complete.get())) == 1, "completed Bool carrier");
    require(work.use_count() == 2 && g_pending_work.size() == 1, "poll consumed the wait handle");
    expectSuccess(lean_torch_dist_wait(9));
    require(work.use_count() == 1 && g_pending_work.empty() && counts.waits == 1,
      "poll then wait did not consume once");
  });
  test("completed failed work reports error and remains drainable", [] {
    Counts counts;
    g_pending_work[10] = c10::make_intrusive<FakeWork>(counts, 1, 0, true, true, false);
    expectError(lean_torch_dist_is_completed(10));
    require(g_pending_work.size() == 1, "failed poll consumed the wait handle");
    expectError(lean_torch_dist_wait(10));
    require(g_pending_work.empty() && counts.waits == 1 && counts.destroyed == 1,
      "failed completed work could not be drained");
  });
  test("unknown poll ID reports error", [] {
    expectError(lean_torch_dist_is_completed(987));
  });
  test("completed failure without exception reports error and remains drainable", [] {
    Counts counts;
    g_pending_work[11] = c10::make_intrusive<FakeWork>(counts, 0, 0, true, false, false, false);
    expectError(lean_torch_dist_is_completed(11));
    require(g_pending_work.size() == 1, "failed poll without exception consumed the handle");
    expectError(lean_torch_dist_wait(11));
    require(g_pending_work.empty() && counts.waits == 1 && counts.destroyed == 1,
      "failed work without exception could not be drained");
  });
  std::printf("Distributed fake-state unit tests: %zu passed, %zu failed\n", passed, failed);
  return failed ? 1 : 0;
#else
  test("initially uninitialized", [] { require(!initialized(), "unexpected process group"); });
  bool cudaAvailable = true;
  test("CUDA availability", [&] {
    Object result(lean_torch_cuda_is_available(), lean_dec);
    require(lean_io_result_is_ok(result.get()), "availability query failed");
    cudaAvailable = lean_unbox(lean_io_result_get_value(result.get())) != 0;
  });
#ifdef TYR_DISTRIBUTED_BASELINE
  // Never select a real GPU through the old narrowing bug.
  if (cudaAvailable) {
    ++skipped; std::puts("SKIP old-library invalid-device baseline on CUDA");
  } else
#endif
  {
    test("CUDA index 128 rejected", [] { expectError(setDevice(128)); });
    test("CUDA index 256 rejected", [] { expectError(setDevice(256)); });
    test("CUDA index UInt64.max rejected", [] { expectError(setDevice(UINT64_MAX)); });
  }
  if (!cudaAvailable) {
    test("CPU device zero no-op", [] { expectSuccess(setDevice(0)); });
    test("CPU representable device no-op", [] { expectSuccess(setDevice(127)); });
  } else { ++skipped; std::puts("SKIP CPU no-op policy on CUDA"); }
#ifndef TYR_DISTRIBUTED_BASELINE
  // Run only after preflight is installed: the previous initializer opened a
  // TCPStore before checking backend/rank/world size. These calls must not.
  auto invalidInit = [&](const char* backend, uint64_t rank, uint64_t size, uint64_t port) {
    Object name(lean_mk_string(backend), lean_dec);
    Object host(lean_mk_string("127.0.0.1"), lean_dec);
    expectError(lean_torch_dist_init_process_group(name.get(), host.get(), port, rank, size));
    require(lean_is_exclusive(name.get()) && lean_is_exclusive(host.get()), "borrowed strings leaked");
    require(!initialized(), "failed init committed state");
  };
  test("unknown backend rejected before network", [&] { invalidInit("invalid", 0, 1, 0); });
  test("zero world size rejected before network", [&] { invalidInit("gloo", 0, 0, 0); });
  test("oversized world size rejected before network", [&] { invalidInit("gloo", 0, UINT64_MAX, 0); });
  test("rank equal to world size rejected before network", [&] { invalidInit("gloo", 1, 1, 0); });
  test("oversized rank rejected before network", [&] { invalidInit("gloo", UINT64_MAX, 1, 0); });
  test("oversized port rejected before network", [&] { invalidInit("gloo", 0, 1, 65536); });
  test("uninitialized rank/world/barrier errors", [] {
    expectError(lean_torch_dist_get_rank()); expectError(lean_torch_dist_get_world_size());
    expectError(lean_torch_dist_barrier());
  });
  test("unknown work ID policy", [] {
    expectError(lean_torch_dist_wait(42));
    expectError(lean_torch_dist_is_completed(42));
  });
  test("uninitialized collectives preserve borrowed tensors", [] {
    auto raw = tensor({2, 2});
    Object array(lean_alloc_array(1, 1), lean_dec);
    lean_inc(raw.get()); lean_array_set_core(array.get(), 0, raw.get());
    expectError(lean_torch_dist_all_reduce(raw.get(), 0, 0));
    expectError(lean_torch_dist_all_reduce(raw.get(), 0, 1));
    expectError(lean_torch_dist_broadcast(raw.get(), 0));
    expectError(lean_torch_dist_reduce_scatter_tensor(raw.get(), raw.get(), 0, 0));
    expectError(lean_torch_dist_reduce_scatter_list(raw.get(), array.get(), 0));
    expectError(lean_torch_dist_all_gather_into_tensor(raw.get(), raw.get(), 0));
    expectError(lean_torch_dist_all_gather_list(array.get(), raw.get()));
    require(lean_is_exclusive(array.get()), "borrowed array leaked");
    array.reset();
    require(lean_is_exclusive(raw.get()), "borrowed tensor leaked");
  });
  test("polar invalid rank returns IO errors", [] {
    auto raw = tensor({2});
    expectError(lean_torch_xxt(raw.get()));
    expectError(lean_torch_ba_plus_cAA(raw.get(), 1, 1));
    expectError(lean_torch_newton_schulz_step(raw.get(), 1, 1, 1));
    expectError(lean_torch_polar_express(raw.get(), 1));
    expectError(lean_torch_muon_orthogonalize(raw.get(), 1));
    require(lean_is_exclusive(raw.get()), "borrowed polar input leaked");
  });
  test("all six polar success carriers preserve input", [] {
    auto raw = tensor({2, 2});
    expectSuccess(lean_torch_xxt(raw.get()));
    expectSuccess(lean_torch_ba_plus_cAA(raw.get(), 1, 1));
    expectSuccess(lean_torch_newton_schulz_step(raw.get(), 1, 1, 1));
    expectSuccess(lean_torch_polar_express(raw.get(), 1));
    expectSuccess(lean_torch_muon_orthogonalize(raw.get(), 1));
    expectSuccess(lean_torch_cautious_update(raw.get(), raw.get(), 0.1, 0.1));
    require(lean_is_exclusive(raw.get()), "borrowed polar input leaked");
    Object values(lean_torch_get_values(shape({}), raw.get(), 4), lean_dec);
    require(lean_sarray_size(values.get()) == 4, "input shape changed");
    auto* data = reinterpret_cast<double*>(lean_sarray_cptr(values.get()));
    for (size_t i = 0; i < 4; ++i) require(data[i] == 1, "borrowed polar input mutated");
  });
  test("idle teardown is repeatable", [] {
    expectSuccess(lean_torch_dist_destroy_process_group());
    expectSuccess(lean_torch_dist_destroy_process_group());
  });
#endif
  test("no process group created", [] { require(!initialized(), "unexpected process group"); });
  std::printf("Distributed ABI recovery: %zu passed, %zu failed, %zu skipped; no networking\n",
    passed, failed, skipped);
  return failed ? 1 : 0;
#endif
}
