#include "generated/tyr_ffi.h"
#include <torch/torch.h>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

extern "C" void lean_initialize();

namespace {
using Object = std::unique_ptr<lean_object, decltype(&lean_dec)>;
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
lean_object* shape(std::initializer_list<uint64_t> dimensions) {
  auto* result = lean_alloc_array(dimensions.size(), dimensions.size());
  size_t i = 0;
  for (auto dimension : dimensions)
    lean_array_set_core(result, i++, lean_box_uint64(dimension));
  return result;
}
torch::Tensor native(lean_object* value) {
  require(lean_is_external(value), "Expected RawTensor external carrier");
  auto* impl = static_cast<c10::TensorImpl*>(lean_get_external_data(value));
  c10::raw::intrusive_ptr::incref(impl);
  return torch::Tensor(c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>::reclaim(impl));
}
Object wrap(torch::Tensor tensor) {
  Object carrier(lean_torch_full(shape({}), 0, 0, lean_box(1)), lean_dec);
  auto* old = static_cast<c10::TensorImpl*>(lean_get_external_data(carrier.get()));
  auto* impl = tensor.unsafeGetTensorImpl();
  c10::raw::intrusive_ptr::incref(impl);
  auto* result = lean_set_external_data(carrier.release(), impl);
  c10::raw::intrusive_ptr::decref(old);
  return Object(result, lean_dec);
}
Object backwardLoss(lean_object* loss, bool retain = false, bool create = false) {
#ifdef TYR_LEGACY_AUTOGRAD_ABI
  return Object(lean_torch_backward_loss(loss), lean_dec);
#else
  return Object(lean_torch_backward_loss(loss, retain, create), lean_dec);
#endif
}
Object backward(lean_object* output, lean_object* cotangent, bool retain = false,
                bool create = false) {
#ifdef TYR_LEGACY_AUTOGRAD_ABI
  return Object(lean_torch_backward_unit(output, cotangent), lean_dec);
#else
  return Object(lean_torch_backward_unit(output, cotangent, retain, create), lean_dec);
#endif
}
#ifndef TYR_LEGACY_AUTOGRAD_ABI
Object grad(lean_object* output, lean_object* input, lean_object* cotangent,
            bool retain = false, bool create = false) {
  return Object(lean_torch_tensor_grad(output, input,
                                       cotangent, retain, create), lean_dec);
}
#endif
void ok(const Object& result) {
  require(lean_io_result_is_ok(result.get()), "Expected successful autograd IO result");
}
void error(const Object& result) {
  require(!lean_io_result_is_ok(result.get()), "Consumed graph must return IO.error");
}
torch::Tensor leaf() {
  return torch::tensor({2.0f, 3.0f}, torch::TensorOptions().requires_grad(true));
}
void lossConsumes() {
  auto input = leaf();
  auto loss = wrap(input.square().sum());
  ok(backwardLoss(loss.get()));
  error(backwardLoss(loss.get()));
  require(input.grad().equal(torch::tensor({4.0f, 6.0f})), "Wrong first backward gradient");
}
void explicitConsumes() {
  auto input = leaf();
  auto output = wrap(input.square());
  auto cotangent = wrap(torch::ones_like(input));
  ok(backward(output.get(), cotangent.get()));
  error(backward(output.get(), cotangent.get()));
}
void retainedGraphCanBeConsumed() {
  auto input = leaf();
  auto loss = wrap(input.square().sum());
  ok(backwardLoss(loss.get(), true));
  ok(backwardLoss(loss.get()));
  error(backwardLoss(loss.get()));
  require(input.grad().equal(torch::tensor({8.0f, 12.0f})), "Retained graph accumulated wrong gradient");
}
void sharedIntermediateIsConsumed() {
  auto input = leaf();
  auto shared = input.square();
  auto first = wrap(shared.sum());
  auto second = wrap(shared.square().sum());
  ok(backwardLoss(first.get()));
  error(backwardLoss(second.get()));
}
#ifndef TYR_LEGACY_AUTOGRAD_ABI
void gradConsumesWithoutAccumulating() {
  auto input = leaf();
  auto inputObject = wrap(input);
  auto output = wrap(input.square());
  auto cotangent = wrap(torch::ones_like(input));
  auto first = grad(output.get(), inputObject.get(), cotangent.get());
  ok(first);
  require(native(lean_io_result_get_value(first.get())).equal(torch::tensor({4.0f, 6.0f})),
          "Wrong functional gradient");
  require(!input.grad().defined(), "grad unexpectedly accumulated into input.grad");
  error(grad(output.get(), inputObject.get(), cotangent.get()));
}
void higherOrderGrad() {
  auto input = leaf();
  auto inputObject = wrap(input);
  auto output = wrap(input.pow(3));
  auto cotangent = wrap(torch::ones_like(input));
  auto first = grad(output.get(), inputObject.get(), cotangent.get(), true, true);
  ok(first);
  auto* firstGradient = lean_io_result_get_value(first.get());
  require(native(firstGradient).requires_grad(), "createGraph did not retain derivative history");
  auto second = grad(firstGradient, inputObject.get(), cotangent.get());
  ok(second);
  require(native(lean_io_result_get_value(second.get())).equal(torch::tensor({12.0f, 18.0f})),
          "Wrong second derivative");
}
void higherOrderBackward(bool scalarLoss) {
  auto input = leaf();
  auto inputObject = wrap(input);
  auto output = wrap(scalarLoss ? input.pow(3).sum() : input.pow(3));
  auto cotangent = wrap(torch::ones_like(input));
  if (scalarLoss) ok(backwardLoss(output.get(), true, true));
  else ok(backward(output.get(), cotangent.get(), true, true));
  auto firstGradient = wrap(input.grad());
  // PyTorch documents the cycle created by backward(create_graph=true).
  // Exercise the public cleanup boundary, retaining our derivative value.
  Object cleared(lean_torch_zero_grad(inputObject.get(), 1), lean_dec);
  ok(cleared);
  require(!input.grad().defined(), "zeroGrad(setToNone=true) did not clear derivative history");
  require(native(firstGradient.get()).requires_grad(), "Backward ignored createGraph");
  auto second = grad(firstGradient.get(), inputObject.get(), cotangent.get());
  ok(second);
  require(native(lean_io_result_get_value(second.get())).equal(torch::tensor({12.0f, 18.0f})),
          "Backward produced incorrect higher derivative");
}
void gradientCleanupModes() {
  auto input = leaf();
  auto inputObject = wrap(input), loss = wrap(input.square().sum());
  ok(backwardLoss(loss.get()));
  Object snapshotResult(lean_torch_grad_of(inputObject.get()), lean_dec);
  ok(snapshotResult);
  auto snapshot = native(lean_io_result_get_value(snapshotResult.get()));
  Object zeroed(lean_torch_zero_grad(inputObject.get(), 0), lean_dec);
  ok(zeroed);
  require(input.grad().defined() && input.grad().equal(torch::zeros_like(input)),
          "zeroGrad(setToNone=false) did not preserve a zeroed gradient buffer");
  Object cleared(lean_torch_zero_grad(inputObject.get(), 1), lean_dec);
  ok(cleared);
  require(!input.grad().defined(), "zeroGrad(setToNone=true) did not remove the gradient buffer");
  require(snapshot.equal(torch::tensor({4.0f, 6.0f})), "Gradient cleanup changed a detached snapshot");
  Object missingResult(lean_torch_grad_of(inputObject.get()), lean_dec);
  ok(missingResult);
  require(native(lean_io_result_get_value(missingResult.get())).equal(torch::zeros_like(input)),
          "Missing gradient no longer reads as a zero snapshot");
}
void manyInputsShareOneTraversal() {
  auto left = leaf(), right = leaf(), unused = leaf();
  auto leftObject = wrap(left), rightObject = wrap(right), unusedObject = wrap(unused);
  auto output = wrap(left * right);
  auto cotangent = wrap(torch::ones_like(left));
  Object inputs(lean_mk_empty_array(), lean_dec);
  for (auto* input : {leftObject.get(), rightObject.get(), unusedObject.get()}) {
    lean_inc(input);
    inputs.reset(lean_array_push(inputs.release(), input));
  }
  Object result(lean_torch_tensor_grad_many(output.get(), inputs.get(), cotangent.get(), 0, 0), lean_dec);
  ok(result);
  auto* gradients = lean_io_result_get_value(result.get());
  require(lean_array_size(gradients) == 3, "Wrong multi-input gradient count");
  require(native(lean_array_get_core(gradients, 0)).equal(right), "Wrong left gradient");
  require(native(lean_array_get_core(gradients, 1)).equal(left), "Wrong right gradient");
  require(native(lean_array_get_core(gradients, 2)).equal(torch::zeros_like(unused)),
          "Unused multi-input gradient was not materialized as zeros");
  require(!left.grad().defined() && !right.grad().defined(), "gradMany mutated accumulated gradients");
  Object repeated(lean_torch_tensor_grad_many(output.get(), inputs.get(), cotangent.get(), 0, 0), lean_dec);
  error(repeated);
}
void invalidCotangentIsRecoverable() {
  auto input = leaf();
  auto inputObject = wrap(input);
  auto output = wrap(input.square());
  auto wrong = wrap(torch::ones({3}));
  error(grad(output.get(), inputObject.get(), wrong.get()));
  error(backward(output.get(), wrong.get()));
  auto right = wrap(torch::ones_like(input));
  ok(grad(output.get(), inputObject.get(), right.get()));
}
void failedGradReturnsError() {
  auto input = leaf();
  auto inputObject = wrap(input), output = wrap(input.square()), wrong = wrap(torch::ones({3}));
  Object result(lean_torch_tensor_grad(output.get(), inputObject.get(), wrong.get(), 0, 0), lean_dec);
  error(result);
}
#endif

// The closure owns both arguments, matching Lean's compiled callback convention.
lean_object* capturedCallback(lean_object* captured, lean_object* arguments) {
  Object capturedOwner(captured, lean_dec), argumentsOwner(arguments, lean_dec);
  auto temporary = native(lean_array_get_core(arguments, 0));
  return wrap((native(captured) * temporary).sum()).release();
}
lean_object* independentCallback(lean_object* captured, lean_object* arguments) {
  lean_dec(arguments);
  return captured;
}
void functionalGradPreservesCapturedGraph(bool independent) {
  auto input = leaf();
  auto capturedNative = input.square();
  auto captured = wrap(independent ? capturedNative.sum() : capturedNative);
  auto inputObject = wrap(input);
  lean_inc(inputObject.get());
  Object inputs(lean_array_push(lean_mk_empty_array(), inputObject.get()), lean_dec);
  for (unsigned i = 0; i < 3; ++i) {
    auto callback = independent ? independentCallback : capturedCallback;
    auto* closure = lean_alloc_closure(reinterpret_cast<void*>(callback), 2, 1);
    lean_inc(captured.get());
    lean_closure_set(closure, 0, captured.get());
    Object result(lean_torch_functional_grad(inputs.get(), closure), lean_dec);
    require(lean_obj_tag(result.get()) == 1, "functionalGrad failed with captured graph");
    auto gradients = lean_ctor_get(result.get(), 0);
    auto gradient = native(lean_array_get_core(gradients, 0));
    require(gradient.equal(independent ? torch::zeros({2}) : torch::tensor({4.0f, 9.0f})),
            "Wrong gradient with captured graph");
    require(!gradient.requires_grad(), "functionalGrad exported derivative history");
    require(!input.grad().defined(), "functionalGrad mutated captured parameter gradients");
  }
  capturedNative.sum().backward();
  require(input.grad().equal(torch::tensor({4.0f, 6.0f})),
          "functionalGrad invalidated the captured outer graph");
}
}

int main(int argc, char** argv) {
  lean_initialize();
  try {
    const bool capturesOnly = argc > 1 && std::string(argv[1]) == "--captures-only";
    if (!capturesOnly) {
      lossConsumes();
      explicitConsumes();
      retainedGraphCanBeConsumed();
      sharedIntermediateIsConsumed();
#ifndef TYR_LEGACY_AUTOGRAD_ABI
      gradConsumesWithoutAccumulating();
      higherOrderGrad();
      higherOrderBackward(false);
      higherOrderBackward(true);
      gradientCleanupModes();
      manyInputsShareOneTraversal();
      invalidCotangentIsRecoverable();
      failedGradReturnsError();
#endif
    }
    functionalGradPreservesCapturedGraph(false);
    functionalGradPreservesCapturedGraph(true);
    std::cout << "Autograd graph lifetime tests passed\n";
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  }
}
