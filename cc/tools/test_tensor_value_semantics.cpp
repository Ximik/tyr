#include "tyr_raw_tensor_abi.h"
#include "tyr_neural_abi.h"
#include "tyr_branching_abi.h"
#include "tyr_owned_kv_abi.h"
#include <torch/torch.h>
#include <atomic>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

extern "C" void lean_initialize();

namespace {
using Object = std::unique_ptr<lean_object, decltype(&lean_dec)>;
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
lean_object* shape(std::initializer_list<uint64_t> dims) {
  auto* result = lean_alloc_array(dims.size(), dims.size());
  size_t i = 0;
  for (auto dim : dims) lean_array_set_core(result, i++, lean_box_uint64(dim));
  return result;
}
lean_object* retain(lean_object* value) { lean_inc(value); return value; }
Object full(double value, bool grad = false, bool row = false) {
  return Object(lean_torch_full(row ? shape({1, 2}) : shape({2, 2}), value, grad, lean_box(1)), lean_dec);
}
// Test-only inspection of the actual native ownership graph. Do not retain
// this handle across a uniqueness test unless that native alias is intentional.
torch::Tensor native(lean_object* value) {
  require(lean_is_external(value), "Expected RawTensor external carrier");
  auto* impl = static_cast<c10::TensorImpl*>(lean_get_external_data(value));
  c10::raw::intrusive_ptr::incref(impl);
  return torch::Tensor(c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>::reclaim(impl));
}
// Install a native fixture in a fresh, real RawTensor carrier. Keep the same
// external class/finalizer and balance both intrusive references explicitly.
Object wrapNative(torch::Tensor tensor) {
  auto carrier = full(0);
  auto* old = static_cast<c10::TensorImpl*>(lean_get_external_data(carrier.get()));
  auto* impl = tensor.unsafeGetTensorImpl();
  c10::raw::intrusive_ptr::incref(impl);
  auto* replaced = lean_set_external_data(carrier.release(), impl);
  c10::raw::intrusive_ptr::decref(old);
  return Object(replaced, lean_dec);
}
Object ioTensor(lean_object* result) {
  Object owner(result, lean_dec);
  require(lean_io_result_is_ok(result), "Unexpected IO failure");
  auto* value = lean_io_result_get_value(result);
  require(!lean_is_scalar(value) && lean_is_external(value), "IO update did not return RawTensor");
  return Object(retain(value), lean_dec);
}
enum class Update { Copy, Branching, CacheSlice, PureSlice };
const char* name(Update update) {
  switch (update) {
    case Update::Copy: return "copyInto";
    case Update::Branching: return "copyInplace";
    case Update::CacheSlice: return "copySliceIO";
    case Update::PureSlice: return "sliceScatter";
  }
  return "unknown";
}
bool sliced(Update update) { return update == Update::CacheSlice || update == Update::PureSlice; }
Object apply(Update update, lean_object* destination, lean_object* source) {
  lean_object* result;
  if (update == Update::Copy) result = lean_torch_copy_(shape({2, 2}), destination, source);
  else if (update == Update::Branching) result = lean_torch_copy_inplace(shape({2, 2}), destination, source);
  else if (update == Update::CacheSlice) result = lean_torch_copy_slice_io(shape({2, 2}), shape({1, 2}), destination, 0, 0, source);
  else result = lean_torch_slice_scatter_along_dim(shape({2, 2}), shape({1, 2}), destination, 0, 0, source);
#ifdef TYR_LEGACY_VALUE_UPDATES
  // Baseline-only adapter for the previous borrowed destination / Unit ABI.
  if (update == Update::Copy || update == Update::CacheSlice) {
    Object io(result, lean_dec);
    require(lean_io_result_is_ok(result), "Unexpected legacy IO failure");
    return Object(destination, lean_dec);
  }
  lean_dec(destination);
  return Object(result, lean_dec);
#else
  if (update == Update::PureSlice) return Object(result, lean_dec);
  return ioTensor(result);
#endif
}
Object requiresGrad(lean_object* input, bool enabled) {
  auto* result = lean_torch_set_requires_grad(shape({2, 2}), input, enabled);
#ifdef TYR_LEGACY_VALUE_UPDATES
  lean_dec(input);
#endif
  return Object(result, lean_dec);
}
Object gradient(lean_object* input) {
#ifdef TYR_LEGACY_VALUE_UPDATES
  return Object(lean_torch_grad_of(shape({2, 2}), input), lean_dec);
#else
  return ioTensor(lean_torch_grad_of(shape({2, 2}), input));
#endif
}
void zeroGrad(lean_object* input) {
  Object result(lean_torch_zero_grad(shape({2, 2}), input), lean_dec);
#ifndef TYR_LEGACY_VALUE_UPDATES
  require(lean_io_result_is_ok(result.get()), "zeroGrad failed");
#endif
}
void isFull(lean_object* input, double value, const char* message) {
  require(native(input).eq(value).all().item<bool>(), message);
}
void updated(lean_object* input, Update update) {
  auto tensor = native(input);
  require(tensor[0].eq(7).all().item<bool>(), "Updated row has wrong values");
  require(tensor[1].eq(sliced(update) ? 1 : 7).all().item<bool>(), "Other row has wrong values");
}

void sharedLean(Update update) {
  auto input = full(1), source = full(7, false, sliced(update));
  auto output = apply(update, retain(input.get()), source.get());
  isFull(input.get(), 1, "Update mutated surviving Lean alias");
  updated(output.get(), update);
  isFull(source.get(), 7, "Update mutated borrowed source");
  require(input->m_rc == 1 && source->m_rc == 1, "Owned/borrowed Lean reference imbalance");
}
void nativeAlias(Update update) {
  auto input = full(1), source = full(7, false, sliced(update));
  auto alias = native(input.get());
  auto output = apply(update, input.release(), source.get());
  require(alias.eq(1).all().item<bool>(), "Update mutated native TensorImpl alias");
  updated(output.get(), update);
}
void viewAlias(Update update, bool detached) {
  auto input = full(1), source = full(7, false, sliced(update));
  auto alias = detached ? native(input.get()).detach() : native(input.get()).view({4});
  auto output = apply(update, input.release(), source.get());
  require(alias.eq(1).all().item<bool>(), "Update mutated view/detach storage alias");
  updated(output.get(), update);
}
void uniqueReuse(Update update) {
  auto input = full(1), source = full(7, false, sliced(update));
  auto* identity = lean_get_external_data(input.get());
  auto output = apply(update, input.release(), source.get());
  require(lean_get_external_data(output.get()) == identity, "Unique ordinary CPU tensor did not reuse TensorImpl");
  updated(output.get(), update);
}
void weakObservers(Update update, bool storage) {
  auto input = full(1), source = full(7, false, sliced(update));
  auto tensor = native(input.get());
  auto* originalImpl = tensor.unsafeGetTensorImpl();
  auto* originalData = tensor.data_ptr();
  if (storage) {
    auto weak = tensor.storage().getWeakStorageImpl();
    tensor.reset();
    auto output = apply(update, input.release(), source.get());
    require(native(output.get()).data_ptr() != originalData, "Update reused weakly observed storage");
  } else {
    c10::weak_intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl> weak(tensor.getIntrusivePtr());
    tensor.reset();
    auto output = apply(update, input.release(), source.get());
    require(native(output.get()).unsafeGetTensorImpl() != originalImpl, "Update reused weakly observed TensorImpl");
  }
}
void promotedStorageWeak(Update update) {
  auto input = full(1), source = full(7, false, sliced(update));
  auto weak = native(input.get()).storage().getWeakStorageImpl();
  auto alias = weak.lock();
  weak.reset();
  require(alias && alias.use_count() == 2 && alias.weak_use_count() == 1,
    "Promoted storage fixture has unexpected ownership counts");
  auto output = apply(update, input.release(), source.get());
  const auto* values = static_cast<const float*>(alias->data());
  for (size_t i = 0; i < 4; ++i)
    require(values[i] == 1, "Update mutated storage held by promoted weak observer");
  require(alias.use_count() == 1 && alias.weak_use_count() == 1,
    "Storage ownership observation changed reference counts");
  updated(output.get(), update);
}
void concurrentStoragePromotion(Update update) {
  // There is continuously either a weak observer or a strong storage alias.
  // A split strong/weak count check could miss the transition between them.
  // This stress case does not depend on scheduling the exact old race window.
  for (size_t trial = 0; trial < 256; ++trial) {
    auto input = full(1), source = full(7, false, sliced(update));
    auto* identity = lean_get_external_data(input.get());
    auto weak = native(input.get()).storage().getWeakStorageImpl();
    std::atomic<bool> start{false}, done{false};
    bool preserved = true;
    std::thread worker([weak = std::move(weak), &start, &done, &preserved]() mutable {
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
      auto alias = weak.lock();
      weak.reset();
      while (!done.load(std::memory_order_acquire)) std::this_thread::yield();
      // Read only after the update completes, avoiding a data race even when
      // testing an implementation that incorrectly reuses the aliased storage.
      if (alias) {
        const auto* values = static_cast<const float*>(alias->data());
        for (size_t i = 0; i < 4; ++i) preserved = preserved && values[i] == 1;
      }
    });
    struct JoinWorker {
      std::atomic<bool>& done;
      std::thread& worker;
      ~JoinWorker() {
        done.store(true, std::memory_order_release);
        if (worker.joinable()) worker.join();
      }
    } cleanup{done, worker};
    start.store(true, std::memory_order_release);
    auto output = apply(update, input.release(), source.get());
    done.store(true, std::memory_order_release);
    worker.join();
    require(lean_get_external_data(output.get()) != identity,
      "Update reused storage during weak-to-strong ownership transition");
    require(preserved, "Update mutated concurrently promoted storage alias");
    updated(output.get(), update);
  }
}
void savedInput(Update update) {
  auto input = full(1), source = full(7, false, sliced(update));
  auto other = torch::ones({2, 2}, torch::TensorOptions().requires_grad(true));
  auto loss = (native(input.get()) * other).sum();
  auto output = apply(update, input.release(), source.get());
  loss.backward();
  require(other.grad().eq(1).all().item<bool>(), "Update changed autograd saved input");
  updated(output.get(), update);
}
void versionObserver(Update update) {
  auto input = full(1), source = full(7, false, sliced(update));
  auto tensor = native(input.get());
  auto version = tensor.unsafeGetTensorImpl()->version_counter();
  auto* identity = tensor.unsafeGetTensorImpl();
  tensor.reset();
  auto output = apply(update, input.release(), source.get());
  require(native(output.get()).unsafeGetTensorImpl() != identity, "Update reused shared version counter");
  require(version.current_version() == 0, "Update modified observed version counter");
}
void externalBuffer(Update update) {
  std::vector<float> buffer(4, 1);
  auto input = wrapNative(torch::from_blob(buffer.data(), {2, 2}, torch::kFloat32));
  auto source = full(7, false, sliced(update));
  auto output = apply(update, input.release(), source.get());
  for (auto value : buffer) require(value == 1, "Update mutated external buffer");
  updated(output.get(), update);
}
void failureCleanup(Update update) {
  auto input = full(1);
  auto source = wrapNative(torch::ones({1, 3}));
  bool rejected = false;
  try { auto output = apply(update, retain(input.get()), source.get()); }
  catch (const std::exception&) { rejected = true; }
  require(rejected, "Mismatched source shape was accepted");
  require(input->m_rc == 1, "Failed update leaked consumed destination");
  require(source->m_rc == 1, "Failed update changed borrowed source reference count");
  isFull(input.get(), 1, "Failed update mutated destination alias");
}
void sameObjectSource(Update update) {
  auto input = full(1);
  // Both parameters refer to the same Lean object with just one owned ref.
  // Borrowing the source must precede release of the destination's Lean ref.
  auto* source = input.get();
  auto output = apply(update, input.release(), source);
  isFull(output.get(), 1, "Self-copy failed to preserve source lifetime");
}
void requiresGradAliases() {
  auto input = full(1);
  auto output = requiresGrad(retain(input.get()), true);
  require(!native(input.get()).requires_grad(), "requiresGrad mutated surviving alias");
  require(native(output.get()).requires_grad() && native(output.get()).is_leaf(), "requiresGrad result is not leaf");
}
void requiresGradGraph() {
  auto input = full(1, true);
  auto leaf = native(input.get());
  Object nonleaf(lean_torch_tensor_mul(shape({2, 2}), input.get(), input.get()), lean_dec);
  auto* identity = lean_get_external_data(nonleaf.get());
  auto output = requiresGrad(nonleaf.release(), true);
  require(lean_get_external_data(output.get()) == identity, "No-op requiresGrad replaced autograd identity");
  native(output.get()).sum().backward();
  require(leaf.grad().eq(2).all().item<bool>(), "requiresGrad disconnected graph");
  auto disabled = requiresGrad(retain(input.get()), false);
  require(leaf.requires_grad() && !native(disabled.get()).requires_grad(), "Disabling requiresGrad mutated leaf alias");
}
void requiresGradUniqueAndRejectedNonleaf() {
  auto input = full(1);
  auto* identity = lean_get_external_data(input.get());
  auto enabled = requiresGrad(input.release(), true);
  require(lean_get_external_data(enabled.get()) == identity, "Unique requiresGrad update did not reuse TensorImpl");
  Object nonleaf(lean_torch_tensor_mul(shape({2, 2}), enabled.get(), enabled.get()), lean_dec);
  bool rejected = false;
  try { auto result = requiresGrad(retain(nonleaf.get()), false); }
  catch (const std::exception&) { rejected = true; }
  require(rejected && nonleaf->m_rc == 1, "Nonleaf requiresGrad rejection/ownership changed");
  require(native(nonleaf.get()).requires_grad(), "Rejected requiresGrad modified nonleaf");
}
void copyAutogradSemantics(Update update) {
  auto input = full(1, true), source = full(7, true, sliced(update));
  auto original = native(input.get());
  auto sourceLeaf = native(source.get());
  auto output = apply(update, input.release(), source.get());
  require(native(output.get()).requires_grad(), "Update lost requires_grad on destination");
  require(original.eq(1).all().item<bool>(), "Update modified original parameter");
  if (update == Update::Copy || update == Update::PureSlice) {
    native(output.get()).sum().backward();
    require(sourceLeaf.grad().eq(1).all().item<bool>(), "Differentiable copy lost source gradient");
  } else {
    require(native(output.get()).is_leaf(), "No-grad update did not return trainable leaf");
  }
}
void gradientSnapshot() {
  auto input = full(1, true);
  native(input.get()).sum().backward();
  auto saved = gradient(input.get());
  Object clipped(lean_torch_clip_grad_norm_(shape({2, 2}), input.get(), 1), lean_dec);
  require(lean_io_result_is_ok(clipped.get()), "Gradient clipping failed");
  isFull(saved.get(), 1, "Gradient clipping mutated prior gradient snapshot");
  zeroGrad(input.get());
  isFull(saved.get(), 1, "zeroGrad mutated prior gradient snapshot");
  require(!native(saved.get()).requires_grad(), "Gradient snapshot retained autograd graph");
  auto missing = full(1);
  auto zero = gradient(missing.get());
  isFull(zero.get(), 0, "Missing gradient did not produce zero snapshot");
}
} // namespace

int main() {
  lean_initialize();
  int failed = 0;
  auto run = [&](const std::string& label, const std::function<void()>& test) {
    try { test(); std::cout << "PASS " << label << '\n'; }
    catch (const std::exception& error) {
      ++failed; std::cout << "FAIL " << label << ": " << error.what() << '\n';
    }
  };
  for (auto update : {Update::Copy, Update::Branching, Update::CacheSlice, Update::PureSlice}) {
    auto prefix = std::string(name(update)) + "/";
    run(prefix + "shared-Lean", [&] { sharedLean(update); });
    run(prefix + "native-alias", [&] { nativeAlias(update); });
    run(prefix + "view", [&] { viewAlias(update, false); });
    run(prefix + "detach", [&] { viewAlias(update, true); });
    run(prefix + "weak-TensorImpl", [&] { weakObservers(update, false); });
    run(prefix + "weak-storage", [&] { weakObservers(update, true); });
    run(prefix + "saved-autograd-input", [&] { savedInput(update); });
    run(prefix + "unique-reuse", [&] { uniqueReuse(update); });
#ifndef TYR_LEGACY_VALUE_UPDATES
    run(prefix + "promoted-storage-weak", [&] { promotedStorageWeak(update); });
    run(prefix + "concurrent-storage-promotion", [&] { concurrentStoragePromotion(update); });
    run(prefix + "version-observer", [&] { versionObserver(update); });
    run(prefix + "external-buffer", [&] { externalBuffer(update); });
    run(prefix + "failure-cleanup", [&] { failureCleanup(update); });
    run(prefix + "autograd-result", [&] { copyAutogradSemantics(update); });
    if (!sliced(update)) run(prefix + "self-copy", [&] { sameObjectSource(update); });
#endif
  }
  run("requiresGrad/shared-Lean", requiresGradAliases);
  run("requiresGrad/graph", requiresGradGraph);
#ifndef TYR_LEGACY_VALUE_UPDATES
  run("requiresGrad/unique-and-nonleaf", requiresGradUniqueAndRejectedNonleaf);
#endif
  run("gradOf/snapshot", gradientSnapshot);
  std::cout << failed << " native value-semantics failures\n";
  return failed == 0 ? 0 : 1;
}
