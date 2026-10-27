/**
 * tyr_polar.cpp - Polar Express kernels for NorMuon optimizer
 *
 * Implements efficient matrix operations for the Polar Express sign method:
 * - XXT: Symmetric matrix multiplication (A @ A.T)
 * - ba_plus_cAA: Fused operation (beta * A + alpha * A @ A.T)
 *
 * These operations are used in the Newton-Schulz iteration for orthogonalization.
 */

#include <lean/lean.h>
#include "generated/tyr_ffi.h"
#include "tyr_checked_result.h"
#include <torch/torch.h>

// Forward declarations from tyr.cpp
extern lean_object* fromTorchTensor(torch::Tensor t);
extern torch::Tensor borrowTensor(b_lean_obj_arg o);

extern "C" {

/**
 * Symmetric matrix multiplication: C = A @ A.T
 *
 * For batched input [batch, rows, cols], output is [batch, rows, rows]
 * For 2D input [rows, cols], output is [rows, rows]
 *
 * This is optimized for symmetric output - only computes upper triangle
 * and mirrors it (when possible).
 *
 * @param A - Input tensor [batch?, rows, cols]
 * @return Output tensor [batch?, rows, rows]
 */
lean_object* lean_torch_xxt(
    b_lean_obj_arg A_obj) {
  return tyr::checkedIoCall("lean_torch_xxt", [&] {
      try {
          auto A = borrowTensor(A_obj);

          torch::Tensor result;
          if (A.dim() == 3) {
              // Batched: [batch, rows, cols] @ [batch, cols, rows] -> [batch, rows, rows]
              result = torch::bmm(A, A.transpose(-1, -2));
          } else if (A.dim() == 2) {
              // 2D: [rows, cols] @ [cols, rows] -> [rows, rows]
              result = torch::mm(A, A.t());
          } else {
              return lean_io_result_mk_error(lean_mk_io_user_error(
                  lean_mk_string("XXT requires 2D or 3D tensor")));
          }

          return lean_io_result_mk_ok(fromTorchTensor(result));
      } catch (const std::exception& e) {
          return lean_io_result_mk_error(lean_mk_io_user_error(
              lean_mk_string(("XXT failed: " + std::string(e.what())).c_str())));
      }
  });
}

/**
 * Fused operation: C = beta * A + alpha * (A @ A.T)
 *
 * Used in Newton-Schulz iteration for Polar Express orthogonalization.
 * This fuses the symmetric matmul with the linear combination to reduce
 * memory bandwidth and improve performance.
 *
 * For batched input [batch, rows, cols], output is [batch, rows, rows]
 * (requires rows == cols for the addition to work)
 *
 * @param A - Input tensor [batch?, n, n] (must be square for addition)
 * @param alpha - Scalar multiplier for A @ A.T
 * @param beta - Scalar multiplier for A
 * @return Output tensor [batch?, n, n]
 */
lean_object* lean_torch_ba_plus_cAA(
    b_lean_obj_arg A_obj,
    double alpha,
    double beta) {
  return tyr::checkedIoCall("lean_torch_ba_plus_cAA", [&] {
      try {
          auto A = borrowTensor(A_obj);

          // Validate input is square (last two dims must match for the addition)
          auto sizes = A.sizes();
          if (sizes.size() < 2) {
              return lean_io_result_mk_error(lean_mk_io_user_error(
                  lean_mk_string("ba_plus_cAA requires at least 2D tensor")));
          }
          if (sizes[sizes.size()-1] != sizes[sizes.size()-2]) {
              return lean_io_result_mk_error(lean_mk_io_user_error(
                  lean_mk_string("ba_plus_cAA requires square matrices (last two dims must match)")));
          }

          torch::Tensor AA;
          if (A.dim() == 3) {
              // Batched: [batch, n, n] @ [batch, n, n]^T -> [batch, n, n]
              AA = torch::bmm(A, A.transpose(-1, -2));
          } else if (A.dim() == 2) {
              // 2D: [n, n] @ [n, n]^T -> [n, n]
              AA = torch::mm(A, A.t());
          } else {
              return lean_io_result_mk_error(lean_mk_io_user_error(
                  lean_mk_string("ba_plus_cAA requires 2D or 3D tensor")));
          }

          // result = beta * A + alpha * AA
          auto result = torch::add(A * beta, AA, alpha);

          return lean_io_result_mk_ok(fromTorchTensor(result));
      } catch (const std::exception& e) {
          return lean_io_result_mk_error(lean_mk_io_user_error(
              lean_mk_string(("ba_plus_cAA failed: " + std::string(e.what())).c_str())));
      }
  });
}

/**
 * Newton-Schulz iteration step for matrix sign approximation.
 *
 * Computes: Y = a*X + b*X@X.T@X + c*X@X.T@X@X.T@X
 *
 * This is the core iteration used in Polar Express for orthogonalization.
 * The coefficients (a, b, c) are precomputed for stability.
 *
 * @param X - Input matrix [batch?, m, n]
 * @param a, b, c - Iteration coefficients
 * @return Output matrix [batch?, m, n]
 */
lean_object* lean_torch_newton_schulz_step(
    b_lean_obj_arg X_obj,
    double a,
    double b,
    double c) {
  return tyr::checkedIoCall("lean_torch_newton_schulz_step", [&] {
      try {
          auto X = borrowTensor(X_obj);

          torch::Tensor XXT;
          if (X.dim() == 3) {
              XXT = torch::bmm(X, X.transpose(-1, -2));
          } else if (X.dim() == 2) {
              XXT = torch::mm(X, X.t());
          } else {
              return lean_io_result_mk_error(lean_mk_io_user_error(
                  lean_mk_string("newton_schulz_step requires 2D or 3D tensor")));
          }

          // term1 = a * X
          auto term1 = a * X;

          // XXT_X = XXT @ X (for b and c terms)
          torch::Tensor XXT_X;
          if (X.dim() == 3) {
              XXT_X = torch::bmm(XXT, X);
          } else {
              XXT_X = torch::mm(XXT, X);
          }

          // term2 = b * XXT @ X
          auto term2 = b * XXT_X;

          // term3 = c * XXT @ XXT @ X = c * XXT @ (XXT @ X)
          torch::Tensor XXT_XXT_X;
          if (X.dim() == 3) {
              XXT_XXT_X = torch::bmm(XXT, XXT_X);
          } else {
              XXT_XXT_X = torch::mm(XXT, XXT_X);
          }
          auto term3 = c * XXT_XXT_X;

          auto result = term1 + term2 + term3;

          return lean_io_result_mk_ok(fromTorchTensor(result));
      } catch (const std::exception& e) {
          return lean_io_result_mk_error(lean_mk_io_user_error(
              lean_mk_string(("newton_schulz_step failed: " + std::string(e.what())).c_str())));
      }
  });
}

/**
 * Polar Express: Full orthogonalization via Newton-Schulz iterations.
 *
 * Approximates the matrix sign function using iterative refinement.
 * Starting with X = G / ||G||, iterates:
 *   X = sum_i (a_i * X + b_i * X@X.T@X + c_i * X@X.T@X@X.T@X)
 *
 * @param G - Input gradient matrix [batch?, m, n]
 * @param num_iters - Number of iterations (typically 5-10)
 * @return Orthogonalized matrix [batch?, m, n]
 */
lean_object* lean_torch_polar_express(
    b_lean_obj_arg G_obj,
    uint64_t num_iters) {
  return tyr::checkedIoCall("lean_torch_polar_express", [&] {
      try {
          auto G = borrowTensor(G_obj);

          // Precomputed coefficients for 5 iterations (from modded-nanogpt)
          // These coefficients are optimized for stability with safety_factor=2e-2, cushion=2
          static const std::vector<std::tuple<double, double, double>> coeffs = {
              {8.156554524902461, -22.48329292557795, 15.878769915207462},
              {4.042929935166739, -2.808917465908714, 0.5000178451051316},
              {3.241553795498743, -1.758787407092089, 0.35970315917498755},
              {2.866073605966538, -1.394706279118519, 0.3190251879692427},
              {2.6379268658498756, -1.1816678706889182, 0.2925389239509298}
          };

          // Normalize input: X = G / ||G||_F
          auto norm = G.norm();
          auto X = G / norm;

          // Run Newton-Schulz iterations
          size_t iters = std::min(static_cast<size_t>(num_iters), coeffs.size());
          for (size_t i = 0; i < iters; i++) {
              auto [a, b, c] = coeffs[i];

              torch::Tensor XXT;
              if (X.dim() == 3) {
                  XXT = torch::bmm(X, X.transpose(-1, -2));
              } else {
                  XXT = torch::mm(X, X.t());
              }

              // Y = a*X + b*XXT@X + c*XXT@XXT@X
              torch::Tensor XXT_X;
              if (X.dim() == 3) {
                  XXT_X = torch::bmm(XXT, X);
              } else {
                  XXT_X = torch::mm(XXT, X);
              }

              torch::Tensor XXT_XXT_X;
              if (X.dim() == 3) {
                  XXT_XXT_X = torch::bmm(XXT, XXT_X);
              } else {
                  XXT_XXT_X = torch::mm(XXT, XXT_X);
              }

              X = a * X + b * XXT_X + c * XXT_XXT_X;
          }

          return lean_io_result_mk_ok(fromTorchTensor(X));
      } catch (const std::exception& e) {
          return lean_io_result_mk_error(lean_mk_io_user_error(
              lean_mk_string(("polar_express failed: " + std::string(e.what())).c_str())));
      }
  });
}

/**
 * Muon-style orthogonalized gradient update.
 *
 * This follows nanochat's zeropower_via_newtonschulz5 implementation:
 * - Cast to bfloat16 for the NS iteration.
 * - If tall matrix, transpose first so the short dimension is orthogonalized.
 * - Normalize by Frobenius norm with epsilon (1e-7) for numerical stability.
 * - Iterate: X = a*X + (b*A + c*A*A) @ X, where A = X @ X^T.
 * - Transpose back for tall inputs.
 *
 * @param G - Gradient tensor [out, in] or [batch, out, in]
 * @param num_iters - Number of Newton-Schulz iterations
 * @return Orthogonalized gradient
 */
lean_object* lean_torch_muon_orthogonalize(
    b_lean_obj_arg G_obj,
    uint64_t num_iters) {
  return tyr::checkedIoCall("lean_torch_muon_orthogonalize", [&] {
      try {
          auto G = borrowTensor(G_obj);
          if (G.dim() < 2) {
              return lean_io_result_mk_error(lean_mk_io_user_error(
                  lean_mk_string("muon_orthogonalize requires tensor rank >= 2")));
          }

          auto sizes = G.sizes();
          int64_t out_dim = sizes[sizes.size() - 2];
          int64_t in_dim = sizes[sizes.size() - 1];

          // Match nanochat: run Newton-Schulz in bf16.
          auto X = G.to(torch::kBFloat16);
          bool transposed = false;
          if (out_dim > in_dim) {
              X = X.transpose(-1, -2);
              transposed = true;
          }

          // Stable normalization: divide by Frobenius norm + epsilon.
          auto norm = X.norm(2, {-2, -1}, /*keepdim=*/true);
          X = X / (norm + 1e-7);

          // Same coefficients as nanochat's zeropower_via_newtonschulz5.
          constexpr double a = 3.4445;
          constexpr double b = -4.7750;
          constexpr double c = 2.0315;
          size_t iters = static_cast<size_t>(num_iters);
          for (size_t i = 0; i < iters; i++) {
              auto A = torch::matmul(X, X.transpose(-1, -2));
              auto B = b * A + c * torch::matmul(A, A);
              X = a * X + torch::matmul(B, X);
          }

          if (transposed) {
              X = X.transpose(-1, -2);
          }

          // Keep reference behavior: output dtype stays bf16.
          return lean_io_result_mk_ok(fromTorchTensor(X));
      } catch (const std::exception& e) {
          return lean_io_result_mk_error(lean_mk_io_user_error(
              lean_mk_string(("muon_orthogonalize failed: " + std::string(e.what())).c_str())));
      }
  });
}

/**
 * Cautious weight decay update.
 *
 * Only applies weight decay when the update and parameter have the same sign:
 *   mask = sign(update) == sign(param)
 *   param = param - lr * (update + wd * mask * param)
 *
 * @param param - Parameter tensor (modified in-place conceptually, but returns new tensor)
 * @param update - Update/gradient tensor
 * @param lr - Learning rate
 * @param wd - Weight decay
 * @return Updated parameter tensor
 */
lean_object* lean_torch_cautious_update(
    b_lean_obj_arg param_obj,
    b_lean_obj_arg update_obj,
    double lr,
    double wd) {
  return tyr::checkedIoCall("lean_torch_cautious_update", [&] {
      try {
          auto param = borrowTensor(param_obj);
          auto update = borrowTensor(update_obj);

          // mask = (sign(update) * sign(param)) > 0
          // This is true when both have the same sign
          auto mask = (torch::sign(update) * torch::sign(param)) > 0;
          mask = mask.to(param.dtype());

          // Cautious weight decay: only apply when signs match
          auto wd_term = wd * mask * param;
          auto new_param = param - lr * (update + wd_term);

          return lean_io_result_mk_ok(fromTorchTensor(new_param));
      } catch (const std::exception& e) {
          return lean_io_result_mk_error(lean_mk_io_user_error(
              lean_mk_string(("cautious_update failed: " + std::string(e.what())).c_str())));
      }
  });
}

} // extern "C"
