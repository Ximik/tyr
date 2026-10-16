#include "generated/tyr_ffi.h"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

extern "C" {
void lean_initialize();
lean_object* initialize_FunctionalGradCallback(uint8_t);
lean_object* tyr_test_callback_sum(lean_object*, lean_object*);
lean_object* tyr_test_callback_identity(lean_object*, lean_object*);
lean_object* tyr_test_callback_throw(lean_object*, lean_object*);
}

using Object = std::unique_ptr<lean_object, decltype(&lean_dec)>;
using Callback = lean_object* (*)(lean_object*, lean_object*);

static void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

static uint64_t liveTensors() {
  Object result(lean_torch_get_live_tensors(), lean_dec);
  require(lean_io_result_is_ok(result.get()), "live tensor query failed");
  return lean_unbox_uint64(lean_io_result_get_value(result.get()));
}

static lean_object* shape() {
  auto* result = lean_alloc_array(1, 1);
  lean_array_set_core(result, 0, lean_box_uint64(2));
  return result;
}

static void run(const std::string& mode) {
  const bool integer = mode == "setup-error";
  Object input(integer ? lean_torch_full_int(shape(), 2)
                       : lean_torch_full(shape(), 2.0, 0, lean_box(1)), lean_dec);
  lean_inc(input.get());
  Object inputs(lean_array_push(lean_mk_empty_array(), input.get()), lean_dec);
  Callback callback = mode == "success" ? tyr_test_callback_sum
                    : mode == "result-error" ? tyr_test_callback_identity
                    : tyr_test_callback_throw;
  const auto baseline = liveTensors();
  for (unsigned i = 0; i < 10; ++i) {
    auto* closure = lean_alloc_closure(reinterpret_cast<void*>(callback), 2, 1);
    lean_inc(input.get());
    lean_closure_set(closure, 0, input.get());
    Object result(lean_torch_functional_grad(inputs.get(), closure), lean_dec);
    require(mode != "callback-error", "Throwing callback returned instead of terminating");
    if (mode == "success") {
      require(lean_obj_tag(result.get()) == 1, "Valid callback did not produce Except.ok");
      auto* gradients = lean_ctor_get(result.get(), 0);
      require(lean_array_size(gradients) == 1, "Wrong gradient leaf count");
      Object values(lean_torch_tensor_to_float_array_dynamic(
          lean_array_get_core(gradients, 0)), lean_dec);
      require(lean_io_result_is_ok(values.get()), "Gradient read failed");
      auto* array = lean_io_result_get_value(values.get());
      require(lean_array_size(array) == 2, "Wrong gradient size");
      for (size_t j = 0; j < 2; ++j)
        require(lean_unbox_float(lean_array_get_core(array, j)) == 1.0, "Wrong gradient value");
    } else {
      require(lean_obj_tag(result.get()) == 0, "Setup/result failure did not produce Except.error");
    }
    result.reset();
    require(liveTensors() == baseline, "Normal callback or setup/result failure leaked tensors");
  }
}

int main(int argc, char** argv) {
  try {
    require(argc == 2, "Expected success, setup-error, result-error, or callback-error");
    const std::string mode = argv[1];
    require(mode == "success" || mode == "setup-error" || mode == "result-error" ||
            mode == "callback-error", "Unknown mode");
    lean_initialize();
    Object initialized(initialize_FunctionalGradCallback(1), lean_dec);
    require(lean_io_result_is_ok(initialized.get()), "Callback initialization failed");
    const auto baseline = liveTensors();
    run(mode);
    require(liveTensors() == baseline, "Fixture cleanup leaked tensors");
    std::cout << mode << " passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
