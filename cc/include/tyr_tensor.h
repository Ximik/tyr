#pragma once

#include <lean/lean.h>
#include <torch/torch.h>

// Borrow a tensor from Lean with proper reference counting.
// Returns an owning tensor handle reconstructed from an owning intrusive ref.
torch::Tensor borrowTensor(b_lean_obj_arg o);

// Transfer ownership of a new tensor to Lean.
// The tensor object is moved to a heap allocation owned by Lean.
lean_object* giveTensor(torch::Tensor t);
