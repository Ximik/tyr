# Indexed tensors and the native boundary

Use `Tyr.Typed` for tensor operations and `Tyr.Typed.Neural` for additional
neural, storage, and data adapters. The public type is `torch.Tensor σ`, where
`σ : StaticSpec` records shape, dtype, and concrete device. `Tyr.Torch` supplies
runtime services such as device discovery, CUDA streams/events, and RNG seeding.
Inference uses explicitly detached tensors rather than ambient gradient-mode scopes.

## Tensor specifications

```lean
import Tyr.Typed
import Tyr.Typed.Neural
import Tyr.Torch
open torch

abbrev Activations := Tensor { shape := #[32, 128], dtype := .Float32, device := .CPU }
abbrev PinnedActivations (device : Device) := Tensor {
  shape := #[32, 128]
  dtype := .BFloat16
  device := device
}

def allocateCUDAActivations : IO (Σ device : Device, PinnedActivations device) := do
  let device ← Device.cuda 0
  let support ← DType.BFloat16.acquireConstantSupport device
  let _ := support.down
  pure ⟨device, Tensor.zerosFor ⟨#[32, 128], .BFloat16, device⟩⟩
```

Every specification supplies exact shape, dtype, and device indices. There is
no wildcard default. `SomeDTypeTensor` packages a runtime dtype without erasing
it. `shape`, `dtype`,
`device`, and `staticSpec` describe the index; `actualSpec` and `runtimeDevice`
inspect native metadata. Runtime-selected devices can appear directly in
result types, or be packaged as `Sigma fun device => Tensor { shape := shape,
dtype := dtype, device := device }` when the caller does not select placement.

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
- `zerosFor`, `onesFor`, and `eyeFor` require exact dtype/device constant support.
- `toFloat32`, `toBFloat16`, and `castDType` change the dtype index.

Pure factories are suitable for values. When independent mutable storage is
required, use an allocation/copy API with an explicit ownership discipline;
repeating a pure expression does not establish independent storage.

`Device.CPU` is always available. Acquire accelerators with `Device.cuda index`
or `Device.mps` in IO before constructing tensors. The result carries the exact
placement in tensor indices. A raw `DeviceKind` description cannot be passed in
its place.

The erased class `DType.ConstantSupported dtype device` establishes that the
constant factory paths support this exact pair and represent zero and one.
Common supported combinations have static instances. For a runtime-selected
pair, acquire evidence once with `dtype.acquireConstantSupport device` and
install `support.down` as a local instance. `Inhabited (Tensor spec)` requires
the same evidence; `default` cannot allocate an unsupported specification.
This capability does not imply support for arbitrary arithmetic kernels.

## Arithmetic, transforms, and reductions

`Tensor.add`, `sub`, `mul`, and `div` take equal shapes and require a
`DType.Promotes` witness for the result dtype. Their `addB`, `subB`, `mulB`, and
`divB` variants additionally support broadcasting through `TensorSpec.Promotes`.
These erased proofs establish the promotion rule; they do not establish that
the operator has a kernel for that dtype and device. Operator-specific domain
constraints reject inputs such as complex ReLU and integral mean. Matrix multiplication
requires compatible dimensions and matching dtype and concrete device, and checks
actual dtype/device equality before execution. `linear` consumes an `[outDim, inDim]` weight matrix;
`linear3d` handles batched sequences.

Broadcasting arithmetic uses `Device.PointwiseCompatible` to reject known
placement conflicts. A rank-zero CPU operand can participate on another device;
a size-one vector cannot. Native metadata is validated at checked boundaries.
Selection has its own scalar relocation rule; see
[device compatibility](typed.md#device-compatibility).

`reshape` requires equal element counts. `reshapeChecked` validates a requested
runtime shape. In both cases `#[]` means a scalar. `sliceChecked` validates the
claimed result of a runtime slice. Other transforms include `transpose`,
`permute`, `unsqueeze`, `squeeze`, and `unbind`.

Use `Tensor.packShape` to package a runtime-dependent shape existentially.
`catMany?` accepts indexed existential tensors and checks the requested output
specification; `catManyChecked` prints a diagnostic and aborts on mismatch.
A dynamic shape does not require pretending that a tensor is a scalar.

Reduction indices follow native accumulation rules. For example, integer sums
may accumulate in Int64 and integer division produces floating output. Ordinary
operations expose those result dtypes in their types. Generic real-floating
algorithms accept `DType.RealFloating d`; its proof instances establish valid
promotion, and its equality lemmas simplify result indices without converting
native data. `FloatingTensorStruct` supplies that evidence at each leaf of a
parameter tree. Model loaders validate compute dtypes and retain the evidence
in `RealFloatingDType`; generated SafeTensors records retain concrete indices
and derived tree constraints. There are no same-dtype adapters that reject promotion and
return an input as a fallback.
Deliberate mixed-precision computation needs explicit input conversions or a
witnessed promoting operation.
`Tensor.Promoting` provides the latter when dtypes are known only at runtime:
its checked binary and host-scalar variants return the computed dtype with a
promotion witness. Its unary convenience wrappers express their computed dtype
directly in the result index. These operations preserve the promoted native
dtype without coercing it to an input's dtype.
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
`zeroGrad` and `TensorStruct.zeroGrads` default to `setToNone := true`, releasing
the gradient buffer and breaking cycles created by `backward(createGraph := true)`.
Pass `(setToNone := false)` to zero an existing buffer instead. `Tensor.grad`
is preferable for higher derivatives because it does not accumulate into `.grad`.
`backwardLoss` takes a scalar loss; `backward` takes an output and an explicit
cotangent with the same specification. Both release saved backward intermediates
by default. Pass `(retainGraph := some true)` for intentional reuse, then release
on the final pass. Omitted retention follows `createGraph`, whose default is
`false`, matching PyTorch. `Tensor.grad` returns a cotangent in IO without
accumulating parameter gradients and follows the same lifetime options.
The tensor values remain readable after backward, but another backward through
consumed saved state returns an IO error. Pure forward expressions may be shared;
repeating an equal expression does not promise a fresh graph. `hasGradFn` reports
the presence of a gradient function, not whether saved intermediates remain usable.
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
    (weight : Tensor { shape := #[classes, inDim], dtype := .Float32, device := .CPU })
    (input : Tensor { shape := #[batch, inDim], dtype := .Float32, device := .CPU })
    (targets : Tensor { shape := #[batch], dtype := .Int64, device := .CPU }) : IO Float := do
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
The `libtyr` build generates `cc/src/generated/tyr_ffi.h` from Lean's compiled
extern declarations (`TyrFFI`), and `libTyrC`'s sources include it, so a
signature mismatch fails to compile. Ownership is not part of the C types.

`Tests.TestTyped`, `Tests.TestTypedNeural`, and `Tests.TestFFIBoundary` cover static
rejections, native metadata, numerical behavior, and ABI-sensitive calls. CPU
parity and GPU target compilation do not establish runtime GPU qualification.

## Related guides

- [Typed specifications](typed.md)
- [Tensor structures](tensorstruct.md)
- [Modules](../modules.md)
- [FFI and build](../ffi-and-build.md)
- [Examples and testing](../examples-and-testing.md)
