# Tyr Code Review & Technical Debt

This document tracks architectural, performance, and correctness issues identified during deep semantic review.

## 🚀 Performance Bottlenecks

### P01: Recursive KV-Cache Prefill
- **Issue**: `prefillCachesFromEmbeds` in `Qwen35/Model.lean` was performing $N$ separate forward passes for a prompt of length $N$.
- **Status**: Partially addressed via Batch Prefill refactor. Needs verification for other models.
- **Impact**: Drastic first-token latency reduction.

### P02: Sequential SafeTensors Loading
- **Issue**: Sharded model loading is fully sequential. Each tensor load involves re-parsing metadata and re-opening file handles.
- **Status**: Standard handle caching implemented.
- **Recommendation**: Implement `mmap`-backed loading in `cc/src/tyr.cpp` to eliminate copies and allow zero-copy weight access. Parallelize the Lean `Array.mapM` calls during layer loading.

### P03: FFI Boundary Overhead
- **Issue**: High-level Lean code calls Torch for every atomic op (e.g. `add`, `mul`, `relu`). In a 32-layer GPT, this incurs thousands of FFI context switches.
- **Recommendation**: Implement **Kernel Fusion** for hot blocks (e.g., Fused Attention, Fused MLP) where a single C++ call handles the entire block.

## 🏛️ Architectural Refactoring

### A01: Monolithic `Tyr/Torch.lean`
- **Issue**: `Torch.lean` is 1450+ lines, mixing low-level FFI, high-level wrappers, and domain-specific submodules (`rotary`, `nn`, `linalg`).
- **Recommendation**: Split into:
  - `Tyr/FFI.lean`: Raw `@extern` declarations.
  - `Tyr/Ops.lean`: Safe Lean wrappers with shape logic.
  - `Tyr/NN/*.lean`: Move functional primitives to dedicated files.

### A02: TensorStruct Higher-Kinded Data (HKD)
- **Issue**: `TensorStruct` traverses indexed `Tensor σ` leaves. This makes generic `tree_transpose` (e.g., `Model (Array α) -> Array (Model α)`) nearly impossible to express cleanly.
- **Recommendation**: Refactor model structures to be parameterized by a leaf provider: `Linear (f : Shape -> Type)`.

### A03: Redundant Linear/Affine Wrappers
- **Issue**: `lean_torch_linear`, `lean_torch_linear3d`, `lean_torch_affine`, and `lean_torch_affine3d` all map to the same `torch::linear` call in C++.
- **Recommendation**: Consolidate into a single polymorphic `torch.linear` in Lean that handles rank-2 and rank-3 via dependent types.

## ⚖️ Correctness & Type Safety

### C01: Shape Inference Runtime Checks
- **Issue**: Complex shapes (e.g. `matmulShape`) are computed at runtime and then "cast" via `reshape`. This deferment hides shape errors until execution.
- **Recommendation**: Move more shape arithmetic to the type-level using Lean 4's macro system and typeclasses.

### C02: Device Consistency
- **Issue**: Some FFI calls (like `rand`, `full`) take an optional `device`, but most ops assume all inputs are on the same device. Mismatches lead to hard Torch crashes.
- **Recommendation**: Add a `Device` index to the `T` type: `T (s : Shape) (d : Device)`. This would provide compile-time guarantees for device-locality.
