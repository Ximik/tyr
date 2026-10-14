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
| `Tyr.Basic.Types` | Raw carrier, specifications, dtype/device types, pure shape helpers |
| `Tyr.Basic` | Indexed carrier, checked raw admission, native metadata, native zero factory |
| `Tyr.Typed.DType` | Dtype promotion and accumulation rules |
| `Tyr.Typed.Spec` | Shape and result-spec algebra |
| `Tyr.Typed.Device` | Placement policies and tensor contracts |
| `Tyr.Typed.Tensor` | Checked constructors, validation, existential packaging |
| `Tyr.Typed.Ops` | Arithmetic, shape transforms, reductions, autograd, host arrays |
| `Tyr.Typed.Neural` | Neural, data, storage, and dynamic-shape adapters |
| `Tyr.Typed.Layers` | Parameter structures and typed layer operations |
| `Tyr.TensorStruct.Cast` | Convert a tensor tree to the specifications in a destination type |
| `Tyr.TensorStruct.DeriveCast` | Derive field-wise casts between explicitly selected record types |
| `Tyr.RawTensor` and submodules | Native declarations without static guarantees |

## Graded claims

A specification can make strong or partial claims:

```lean
import Tyr.Typed
open torch

abbrev ShapeOnly := Tensor { shape := #[2, 3] }
abbrev ShapeAndDType := Tensor { shape := #[2, 3], dtype := .Float32 }
abbrev FullySpecified := Tensor {
  shape := #[2, 3], dtype := .Float32, device := .exact .CPU
}
```

Omitting dtype uses `.Unknown "dynamic"`; omitting device policy uses `.any`.
These dynamic components do not assert that a tensor has any particular native
dtype or location. All three examples use the same `Tensor` carrier.
Generic operations should retain a `StaticSpec` or explicit dtype/device
parameters instead of discarding claims through a shape-only specification.
Two shape-only tensors do not necessarily have the same native dtype, even when
their static dtype is the same `.Unknown "dynamic"` value. Only that reserved
token is a wildcard; other unrecognized dtype tokens must match exactly.
Preserving adapters compare actual dtypes, and copying rejects a dtype mismatch
before mutation. Use an explicit cast when conversion is intended.

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

Broadcasting distinguishes zero-dimensional scalar tensors from nonscalars:
within the same dtype category, the nonscalar's precision wins. A Float64 scalar
plus a Float32 vector therefore returns Float32. This is different from adding
two nonscalar Float64/Float32 tensors.

Matrix multiplication requires operands with matching dtype and device policy.
Matching `.any` policies do not prove matching actual devices: native device
compatibility still matters. Static shape and dtype rules complement runtime
checks; they do not verify the implementation of an extern.

## Boundary validation

```lean
import Tyr.Typed
open torch

def checkRaw (raw : RawTensor) : Except String (Tensor { shape := #[2, 3], dtype := .Float32 }) :=
  Tensor.ofRaw raw
```

`Tensor.ofRaw` checks all claims in the destination index. `Tensor.ofTensor` and
`ofTensor?` check a requested shape and dtype; `ofTensorWithContract` also checks
a `TensorContract` placement policy. `Tensor.validate` audits an existing indexed
value against its current native metadata.

`TensorContract` contains a `TensorSpec`, a device policy, and a descriptive
`TensorRole`. The role does not itself enforce a runtime invariant.

There is no public unchecked raw constructor or implicit coercion to
`RawTensor`. The opaque carrier prevents anonymous construction, structural
elimination, and record updates from replacing a checked handle. Native adapters use explicit
`.raw` access and validate returned handles. `checkedReplacement?` additionally
checks that actual dtype and device are unchanged, including for partial specs.
The native `zerosFor` factory bootstraps pure defaults from a requested spec;
it accepts no foreign handle and trusts native allocation/casting. Native
metadata, allocation, and the C ABI remain trusted and covered by parity tests
and ABI audits.
Pure factory calls may share allocations; distinct storage requires an effectful
allocation boundary such as `data.cloneInferenceIO`.
Model and example code should compose typed operations instead of rebuilding
indexed tensors from their raw handles. Raw input adapters should use `ofRaw`;
IO APIs can turn its error into a contextual `IO.userError`.

## Runtime-dependent shapes

A shape may depend on a Lean runtime value and still occur in a dependent result
type. If the native computation determines it, package it existentially:

```lean
import Tyr.Typed
import Tyr.Typed.Neural
open torch

def matchingPositions (tokens : Tensor { shape := #[4], dtype := .Int64 }) :=
  Tensor.findTokenPositions tokens 7
```

This returns `IO (Sigma fun count => Tensor { shape := #[count], dtype := .Int64 })`.
`Tensor.packShape` packages an existing indexed tensor's shape while retaining
its dtype and device-policy parameters. `catMany?` accepts an array of such
packages and validates a requested result shape. `Tensor.toSome` instead returns
`Sigma Tensor`, retaining the complete specification as an existential witness.
A scalar is `#[]`; it is never a substitute for an unknown shape.

`reshape` requires equal element counts. `reshapeChecked?` checks runtime
element counts and rejects wrapped negative extents before invoking the native
reshape. `castDType?` and `castLike?` perform real dtype conversions and validate
the result. Their pure convenience wrappers, along with `sliceChecked`, use
Lean's panic fallback on metadata failure; use the `Except` variants when an
error must be handled. Exact file loading checks the stored layout
before returning a tensor; it does not repair a mismatch by reshaping.

## Models, differentiation, and persistence

`TensorStruct` traversals preserve each leaf's complete specification. Typed
parameters therefore retain shape, dtype, and device-policy information through
mapping, folding, flattening, and generated AD boundaries.

Use `TensorStruct.castTo TargetType source` to choose each leaf's dtype and
placement in the destination Lean type. For example, a model can request BF16
weights and retain Int64 indices:

```lean
import Tyr.TensorStruct.DeriveCast
open torch

structure StoredWeights where
  weight : Tensor { shape := #[2, 3], dtype := .Float32 }
  indices : Frozen { shape := #[2], dtype := .Int64 }
  config : Static String

structure RuntimeWeights where
  weight : Tensor {
    shape := #[2, 3], dtype := .BFloat16, device := .exact .CPU
  }
  indices : Frozen { shape := #[2], dtype := .Int64 }
  config : Static String

derive_tensor_tree_cast StoredWeights => RuntimeWeights

def prepareWeights (source : StoredWeights) : IO RuntimeWeights :=
  TensorStruct.castTo RuntimeWeights source
```

The destination type supplies every leaf specification. A concrete dtype
requests a native cast, and `.exact device` requests placement on that device.
`castTo` requests dtype and placement together in one native conversion, so an
MPS Float32 tensor can request CPU Float64 directly.
A dynamic target dtype retains the source's actual dtype; `.any` retains its
actual placement. Every leaf keeps its shape, so callers perform any reshape
explicitly beforehand.

`TensorTreeCast SourceType TargetType` supplies the traversal. Products, arrays,
lists, options, and fixed-length vectors compose supported leaf casts. Array and
list lengths, option branches, vector lengths, `Frozen` roles, and `Static`
values are preserved. The record command requires matching field names in the
same declaration order and delegates each field to its cast instance. For
`castTo`, metadata validation failures and caught native conversion exceptions
produce IO errors containing the field or element path.

`TensorStruct.castDType dtype params` is a uniform convenience operation for
the same declared structure type. It validates each converted leaf against its
original specification, allowing dynamic dtype leaves to change and reporting
an error for a conflicting concrete dtype.

The indexed autograd API includes effectful `parameter` and `setRequiresGrad`,
pure `detach` and `grad`, plus `gradOf`,
`backwardLoss`, and explicit-cotangent `backward`. `gradOf` reads an independent
gradient snapshot in IO; `zeroGrad` clears accumulated gradients in IO. Backward
retains the graph for surviving forward values. Inference explicitly detaches model
parameters and floating inputs. Value
updates consume a tensor and return its replacement, preserving retained aliases.
Typed adapters capture actual dtype/device before the native call and validate
the replacement without retaining the consumed tensor. Indexing a tensor does
not establish unique storage ownership.

SafeTensors schema/type-provider loaders check shape, dtype, and requested
placement. Checkpoint persistence uses exact tensor loading and preserves the
specifications supplied by its template parameter structure.

## Related guides

- [Tensor operations and examples](tensors.md)
- [Tensor structures](tensorstruct.md)
- [Serialization](../serialization.md)
- [Autodiff](../autodiff.md)
- [Examples and testing](../examples-and-testing.md)
