# Tensor specifications and checked boundaries

`torch.Tensor σ` is Tyr's public tensor type. Its `StaticSpec` index survives
Lean typechecking and records shape, dtype, and device policy. The underlying
`RawTensor` is a native LibTorch handle used by implementation adapters.

## Imports and implementation layers

`import Tyr.Typed` provides specifications, tensor constructors, core operations,
and typed layer helpers. Add `import Tyr.Typed.Neural` for additional neural,
data, and persistence operations. `import Tyr` aggregates the library.

| Module | Responsibility |
| --- | --- |
| `Tyr.Basic.Types` | Tensor carrier, specifications, dtype/device types, shape helpers |
| `Tyr.Basic` | Native metadata extraction |
| `Tyr.Typed.DType` | Dtype promotion and accumulation rules |
| `Tyr.Typed.Spec` | Shape and result-spec algebra |
| `Tyr.Typed.Device` | Placement policies and tensor contracts |
| `Tyr.Typed.Tensor` | Checked constructors, validation, existential packaging |
| `Tyr.Typed.Ops` | Arithmetic, shape transforms, reductions, autograd, host arrays |
| `Tyr.Typed.Neural` | Neural, data, storage, and dynamic-shape adapters |
| `Tyr.Typed.Layers` | Parameter structures and typed layer operations |
| `Tyr.RawTensor` and submodules | Native declarations without static guarantees |

## Graded claims

A specification can make strong or partial claims:

```lean
import Tyr.Typed
open torch

abbrev ShapeOnly := STensor #[2, 3]
abbrev ShapeAndDType := DTensor #[2, 3] .Float32
abbrev FullySpecified := Tensor {
  shape := #[2, 3], dtype := .Float32, device := .exact .CPU
}
```

`STensor` uses `.Unknown "dynamic"` for dtype and `.any` for device policy.
`DTensor` fixes dtype and uses `.any` placement. These dynamic components do not
assert that a tensor has any particular native dtype or location.

Use `tensor.staticSpec` for declared claims, `tensor.actualSpec` for native shape
and dtype, and `tensor.device` for native placement. `forgetDevice` weakens a
placement policy without moving or copying storage.

## Shape and dtype rules

`TensorSpec.broadcastShapes?` and `matmulShape?` return `Option Shape`.
Broadcasting and generic matrix multiplication require compatibility evidence;
concrete valid dimensions are usually discharged automatically. Specialized
operations such as `linear`, `linear3d`, and `bmm4d` encode their dimension
relationships directly in their argument types.

Dtype promotion is explicit in result specifications. For example, UInt8 with
Int8 promotes to Int16, Float16 with BFloat16 promotes to Float32, and integer
sums use their specified accumulation dtype. Floating-only math requires a
floating dtype; a caller with a dynamic dtype may need an explicit cast or a
metadata-checked preserving adapter.

Matrix multiplication requires operands with matching dtype and device policy.
Matching `.any` policies do not prove matching actual devices: native device
compatibility still matters. Static shape and dtype rules complement runtime
checks; they do not verify the implementation of an extern.

## Boundary validation

```lean
import Tyr.Typed
open torch

def checkRaw (raw : RawTensor) : Except String (DTensor #[2, 3] .Float32) :=
  Tensor.ofRaw raw
```

`Tensor.ofRaw` checks all claims in the destination index. `Tensor.ofTensor` and
`ofTensor?` check a requested shape and dtype; `ofTensorWithContract` also checks
a `TensorContract` placement policy. `Tensor.validate` audits an existing indexed
value against its current native metadata.

`TensorContract` contains a `TensorSpec`, a device policy, and a descriptive
`TensorRole`. The role does not itself enforce a runtime invariant.

`Tensor.assumeSpec` and `assumeDType` wrap a raw handle without validation. Their
use is a trust assumption about the native result, not a proof that its metadata
matches the index. Native adapter parity tests and ABI audits are necessary to
support those assumptions. Checked replacement adapters validate metadata when
an operation's generic dtype might otherwise change unexpectedly.

## Runtime-dependent shapes

A shape may depend on a Lean runtime value and still occur in a dependent result
type. If the native computation determines it, package it existentially:

```lean
import Tyr.Typed
import Tyr.Typed.Neural
open torch

def matchingPositions (tokens : DTensor #[4] .Int64) :=
  Tensor.findTokenPositions tokens 7
```

This returns `IO (Sigma fun count => DTensor #[count] .Int64)`.
`Tensor.packShape` packages an existing indexed tensor. `catMany?` accepts an array
of such packages and validates a requested result shape. A scalar is `#[]`;
it is never a substitute for an unknown shape.

`reshape` requires equal element counts. `reshapeChecked` and `sliceChecked`
validate runtime-dependent results. Exact file loading checks the stored layout
before returning a tensor; it does not repair a mismatch by reshaping.

## Models, differentiation, and persistence

`TensorStruct` traversals preserve each leaf's complete specification. Typed
parameters therefore retain shape, dtype, and device-policy information through
mapping, folding, flattening, and generated AD boundaries.

The indexed autograd API includes `setRequiresGrad`, `detach`, `grad`, `gradOf`,
`backwardLoss`, and explicit-cotangent `backward`. Tensor mutation and operations
that require fresh storage use explicit effectful APIs where provided; indexing
a tensor does not establish unique ownership.

SafeTensors schema/type-provider loaders check shape, dtype, and requested
placement. Checkpoint persistence uses exact tensor loading and preserves the
specifications supplied by its template parameter structure.

## Related guides

- [Tensor operations and examples](tensors.md)
- [Tensor structures](tensorstruct.md)
- [Serialization](../serialization.md)
- [Autodiff](../autodiff.md)
- [Examples and testing](../examples-and-testing.md)
