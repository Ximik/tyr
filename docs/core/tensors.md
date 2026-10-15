# Indexed tensors and the native boundary

Use `Tyr.Typed` for tensor operations and `Tyr.Typed.Neural` for additional
neural, storage, and data adapters. The public type is `torch.Tensor σ`, where
`σ : StaticSpec` records shape, dtype, and device policy. `Tyr.Torch` supplies
runtime services such as device discovery, CUDA streams/events, and RNG seeding.
Inference uses explicitly detached tensors rather than ambient gradient-mode scopes.

## Tensor specifications

```lean
import Tyr.Typed
import Tyr.Typed.Neural
open torch

abbrev Activations := Tensor { shape := #[32, 128], dtype := .Float32 }
abbrev PinnedActivations := Tensor {
  shape := #[32, 128]
  dtype := .BFloat16
  device := .exact (.CUDA 0)
}
```

`Tensor { shape := shape, dtype := dtype }` leaves placement unrestricted.
`Tensor { shape := shape }` also leaves dtype dynamic. A dynamic dtype or `.any` placement makes no claim about that
component; `actualSpec` and `device` report native metadata. `shape`, `dtype`, and
`staticSpec` describe the declared index. Different shape/dtype indices are not
interchangeable.

`RawTensor` is the native handle used by adapters in `Tyr.RawTensor` and its
submodules. It is not the model-facing tensor type. `Tensor.ofRaw` validates
shape, dtype, and placement and returns `Except String (Tensor σ)`.
There is no public unchecked raw constructor or implicit raw coercion. Native
adapters use explicit `.raw` access and validate returned handles. Static
indices alone do not prove a native implementation correct: metadata extraction
and the native zero factory remain trusted native boundaries.

## Creation and conversion

Factories live in `torch.Tensor`:

- `zeros`, `ones`, and `full` create Float32 tensors.
- `fullInt` creates Int64 tensors.
- `fromFloatArray` and `fromInt64Array` retain the host array length in the shape.
- `randn` and `uniformLike` return fresh random tensors in `IO`.
- `zerosFor` and `onesFor` accept a complete specification.
- `toFloat32`, `toBFloat16`, and `castDType` change the dtype index.

Pure factories are suitable for values. When independent mutable storage is
required, use an allocation/copy API with an explicit ownership discipline;
repeating a pure expression does not establish independent storage.

## Arithmetic, transforms, and reductions

`Tensor.add`, `sub`, `mul`, and `div` take equal shapes and require a
`DType.Promotes` witness for the result dtype. Their `addB`, `subB`, `mulB`, and
`divB` variants additionally support broadcasting through `TensorSpec.Promotes`.
These erased proofs establish the promotion rule; they do not establish that
the operator has a kernel for that dtype and device. Operator-specific domain
constraints reject inputs such as complex ReLU and integral mean. Matrix multiplication
requires compatible dimensions and matching dtype and device policy, and checks
actual dtype/device equality before execution. `linear` consumes an `[outDim, inDim]` weight matrix;
`linear3d` handles batched sequences.

Broadcasting arithmetic uses `DevicePolicy.PointwiseCompatible` to reject known
placement conflicts. A rank-zero CPU operand can participate on another device;
a size-one vector cannot. Even matching `.any` policies require actual-device
validation. Selection has its own scalar relocation rule; see
[device compatibility](typed.md#device-compatibility).

`reshape` requires equal element counts. `reshapeChecked` validates a requested
runtime shape. In both cases `#[]` means a scalar. `sliceChecked` validates the
claimed result of a runtime slice. Other transforms include `transpose`,
`permute`, `unsqueeze`, `squeeze`, and `unbind`.

Use `Tensor.packShape` to package a runtime-dependent shape existentially.
`catMany?` accepts indexed existential tensors and checks the requested output
specification; `catManyChecked` reports a mismatch through Lean's panic mechanism.
A dynamic shape does not require pretending that a tensor is a scalar.

Reduction indices follow native accumulation rules. For example, integer sums
may accumulate in Int64. `Tensor.Preserving` provides explicit runtime-checked
adapters for generic or shape-only tensors. Its binary arithmetic checks actual
operand promotion and rejects a change to the left operand's dtype, then checks
result metadata. Use the `?` variants to handle these rejections; convenience
wrappers use Lean's panic fallback. Deliberate mixed-precision computation needs
explicit input conversions or a witnessed promoting operation.
`Tensor.Promoting` provides the latter when dtypes are known only at runtime:
its checked variants return the computed dtype with a promotion witness, and
its convenience wrappers retain that native result under a dynamic dtype index.
They do not coerce operands to the left operand's dtype.
The arithmetic `?` variants also return native kernel failures through `Except`.
The checked matrix, expansion, concatenation, and reduction APIs provide
recoverable errors for their respective checks and native calls; this does not
extend to every raw native operation. See [operator checks](typed.md#operator-checks-and-failures).

## Autograd and mutation

`setRequiresGrad`, `detach`, and `gradOf` retain the tensor specification.
`setRequiresGrad : Tensor σ → Bool → IO (Tensor σ)` returns the requested flag
without changing retained aliases. `parameter : Tensor σ → IO (Tensor σ)` always
creates an independent detached trainable leaf. `gradOf : IO (Tensor σ)` returns a detached, independent snapshot of
accumulated gradients. Later backward passes, clipping, and `zeroGrad : IO Unit`
do not change a previously returned gradient snapshot.
`backwardLoss` takes a scalar loss; `backward` takes an output and an explicit
cotangent with the same specification. Both retain the graph for reuse; its saved
storage remains live while outputs are reachable. `hasGradFn` inspects graph ownership.
`copyInto : Tensor σ → Tensor σ → IO (Tensor σ)` returns the updated destination.
Always use its result: retained aliases keep their original values. Native code
consumes the destination reference and reuses storage only after proving native
ownership permits it; otherwise it copies. Inference `data.copySliceIO` follows
the same returned-value contract. Reuse currently covers ordinary uniquely owned
CPU storage; accelerator, view, autograd, and externally backed storage take the
copying path.

```lean
import Tyr.Typed
import Tyr.Typed.Neural
open torch

def classifierStep {batch inDim classes : UInt64}
    (weight : Tensor { shape := #[classes, inDim], dtype := .Float32 })
    (input : Tensor { shape := #[batch, inDim], dtype := .Float32 })
    (targets : Tensor { shape := #[batch], dtype := .Int64 }) : IO Float := do
  let logits := Tensor.linear input weight
  let loss := Tensor.crossEntropy logits targets
  Tensor.backwardLoss loss
  pure loss.item
```

The caller is responsible for creating trainable leaf parameters and managing
accumulated gradients. See [autodiff](../autodiff.md) for the separate staged AD
system and [tensor structures](tensorstruct.md) for parameter-tree traversal.

## Neural operations and persistence

`Tyr.Typed.Neural` includes attention/GQA, checked native flash-attention dispatch,
rotary embeddings, spectral/audio helpers, and file adapters. Linear algebra and
norm operations are available under `Tensor`, rather than separate legacy
functional namespaces.

`Tensor.saveTensor` writes a tensor. `Tensor.loadTensorExact σ path` requires the
stored shape to match and validates dtype and placement. SafeTensors loaders in
`Tyr.SafeTensors` validate stored metadata and can generate indexed parameter
structures; see [serialization](../serialization.md).

## Verification

Native adapters contain `@[extern]` declarations over `RawTensor`. Their ABI must
match the C++ definitions, including boxed versus unboxed arguments and ownership.
After building the adapters, run `python3 scripts/check_ffi_abi.py`. Additional
per-module ABI audits are listed in the CPU CI workflow and qualification script.

`Tests.TestTyped`, `Tests.TestTypedNeural`, and `Tests.TestFFIBoundary` cover static
rejections, native metadata, numerical behavior, and ABI-sensitive calls. CPU
parity and GPU target compilation do not establish runtime GPU qualification.

## Related guides

- [Typed specifications](typed.md)
- [Tensor structures](tensorstruct.md)
- [Modules](../modules.md)
- [FFI and build](../ffi-and-build.md)
- [Examples and testing](../examples-and-testing.md)
