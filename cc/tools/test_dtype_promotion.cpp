#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <torch/version.h>

#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::string oneLine(std::string value) {
  for (char& character : value)
    if (character == '\t' || character == '\n' || character == '\r') character = ' ';
  return value;
}

template <typename Operation>
void emitResult(const std::string& prefix, Operation operation) {
  try {
    const auto result = operation();
    if (result == c10::ScalarType::Undefined)
      std::cout << prefix << "\tundefined\tUndefined\n";
    else
      std::cout << prefix << "\tok\t" << c10::toString(result) << '\n';
  } catch (const c10::Error& error) {
    std::cout << prefix << "\terror\t" << oneLine(error.what_without_backtrace()) << '\n';
  } catch (const std::exception& error) {
    std::cout << prefix << "\terror\t" << oneLine(error.what()) << '\n';
  } catch (...) {
    std::cout << prefix << "\terror\tunknown native exception\n";
  }
}

struct Operands {
  c10::ScalarType dtype;
  at::Tensor scalar;
  at::Tensor vector;
  bool available = false;
};

void emitArithmetic(const char* operation, const char* ranks,
                    const at::Tensor& left, const at::Tensor& right) {
  const std::string prefix = std::string("operation\t") + operation + '\t' + ranks + '\t' +
      c10::toString(left.scalar_type()) + '\t' + c10::toString(right.scalar_type());
  try {
    at::Tensor result;
    if (std::string(operation) == "add") result = at::add(left, right);
    else if (std::string(operation) == "sub") result = at::sub(left, right);
    else if (std::string(operation) == "mul") result = at::mul(left, right);
    else result = at::div(left, right);
    std::cout << prefix << "\tok\t" << c10::toString(result.scalar_type()) << '\n';
  } catch (const c10::Error& error) {
    const std::string message = oneLine(error.what_without_backtrace());
    const bool bool_subtraction = std::string(operation) == "sub" &&
        (left.scalar_type() == c10::ScalarType::Bool || right.scalar_type() == c10::ScalarType::Bool);
    const bool unavailable = message.find("not implemented") != std::string::npos ||
        message.find("not supported") != std::string::npos;
    std::cout << prefix << '\t' << (bool_subtraction ? "rejected" : unavailable ? "unavailable" : "unexpected")
              << '\t' << message << '\n';
  } catch (const std::exception& error) {
    std::cout << prefix << "\tunexpected\t" << oneLine(error.what()) << '\n';
  } catch (...) {
    std::cout << prefix << "\tunexpected\tunknown native exception\n";
  }
}

size_t emitArithmeticMatrix() {
  // Runtime kernel coverage is deliberately separate from promoteTypes: a
  // valid result dtype does not promise a kernel for every device/operation.
  const std::vector<c10::ScalarType> ordinary = {
      c10::ScalarType::Bool, c10::ScalarType::Byte, c10::ScalarType::Char,
      c10::ScalarType::Short, c10::ScalarType::Int, c10::ScalarType::Long,
      c10::ScalarType::Half, c10::ScalarType::BFloat16, c10::ScalarType::Float,
      c10::ScalarType::Double, c10::ScalarType::ComplexHalf,
      c10::ScalarType::ComplexFloat, c10::ScalarType::ComplexDouble};
  size_t cases = 0;
  for (const auto left : ordinary) {
    for (const auto right : ordinary) {
      for (const bool left_scalar : {false, true}) {
        for (const bool right_scalar : {false, true}) {
          const auto lhs = at::ones(left_scalar ? at::IntArrayRef{} : at::IntArrayRef{2},
                                    at::TensorOptions().dtype(left).device(at::kCPU));
          const auto rhs = at::ones(right_scalar ? at::IntArrayRef{} : at::IntArrayRef{2},
                                    at::TensorOptions().dtype(right).device(at::kCPU));
          const std::string ranks = std::string(left_scalar ? "s" : "v") + (right_scalar ? "s" : "v");
          for (const char* operation : {"add", "sub", "mul", "div"}) {
            emitArithmetic(operation, ranks.c_str(), lhs, rhs);
            ++cases;
          }
        }
      }
    }
  }
  return cases;
}

Operands makeOperands(c10::ScalarType dtype) {
  Operands result{dtype, {}, {}, false};
  std::string errors;
  // Meta tensors need no data or kernels. Some storage-only types require the
  // CPU constructor instead; result_type only inspects dtype and dimension.
  for (const auto device : {c10::DeviceType::Meta, c10::DeviceType::CPU}) {
    try {
      const auto options = at::TensorOptions().dtype(dtype).device(device);
      result.scalar = at::empty({}, options);
      result.vector = at::empty({2}, options);
      result.available = true;
      std::cout << "tensor\t" << c10::toString(dtype) << "\tok\t"
                << (device == c10::DeviceType::Meta ? "meta" : "cpu") << '\n';
      return result;
    } catch (const c10::Error& error) {
      errors += oneLine(error.what_without_backtrace()) + "; ";
    } catch (const std::exception& error) {
      errors += oneLine(error.what()) + "; ";
    } catch (...) {
      errors += "unknown native exception; ";
    }
  }
  std::cout << "tensor\t" << c10::toString(dtype) << "\tskip\t" << errors << '\n';
  return result;
}
} // namespace

int main() {
  try {
    std::cout << "version\t" << TORCH_VERSION << '\n';
    std::vector<c10::ScalarType> dtypes;
    std::set<std::string> names;
    // The native enum, not Lean's supported list, determines the inventory.
    for (int ordinal = 0; ordinal < static_cast<int>(c10::ScalarType::NumOptions); ++ordinal) {
      const auto dtype = static_cast<c10::ScalarType>(ordinal);
      if (dtype == c10::ScalarType::Undefined) continue;
      const std::string name = c10::toString(dtype);
      if (name == "UNKNOWN_SCALAR" || !names.insert(name).second)
        throw std::runtime_error("Unrecognized or duplicate native ScalarType at " + std::to_string(ordinal));
      dtypes.push_back(dtype);
      std::cout << "dtype\t" << ordinal << '\t' << name << '\n';
    }
    for (const auto left : dtypes)
      for (const auto right : dtypes)
        emitResult(std::string("promote\t") + c10::toString(left) + '\t' + c10::toString(right),
                   [&] { return c10::promoteTypes(left, right); });

    std::vector<Operands> operands;
    size_t unavailable = 0;
    for (const auto dtype : dtypes) {
      operands.push_back(makeOperands(dtype));
      if (!operands.back().available) ++unavailable;
    }
    size_t result_cases = 0;
    for (const auto& left : operands) {
      for (const auto& right : operands) {
        if (!left.available || !right.available) continue;
        for (const bool left_scalar : {false, true}) {
          for (const bool right_scalar : {false, true}) {
            const std::string ranks = std::string(left_scalar ? "s" : "v") + (right_scalar ? "s" : "v");
            emitResult("result\t" + ranks + '\t' + c10::toString(left.dtype) + '\t' + c10::toString(right.dtype),
                       [&] { return at::result_type(left_scalar ? left.scalar : left.vector,
                                                   right_scalar ? right.scalar : right.vector); });
            ++result_cases;
          }
        }
      }
    }
    const auto arithmetic_cases = emitArithmeticMatrix();
    std::cout << "operations\t" << arithmetic_cases << '\n';
    std::cout << "summary\t" << dtypes.size() << '\t' << dtypes.size() * dtypes.size()
              << '\t' << result_cases << '\t' << unavailable << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Native promotion inventory failed: " << error.what() << '\n';
    return 1;
  } catch (...) {
    std::cerr << "Native promotion inventory failed: unknown native exception\n";
    return 1;
  }
}
