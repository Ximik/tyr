# Indexed tensors and the native boundary

Use `Tyr.Typed` for tensor operations and `Tyr.Typed.Neural` for additional
neural, storage, and data adapters. The public type is `torch.Tensor σ`, where
`σ : StaticSpec` records shape, dtype, and device policy. `Tyr.Torch` supplies
runtime services such as device discovery, CUDA streams/events, RNG seeding,
and global gradient-mode control.

## Tensor specifications

```lean
import Tyr.Typed
import Tyr.Typed.Neural
open torch

abbrev Activations := DTensor #[32, 128] .Float32
abbrev PinnedActivations := Tensor {
  shape := #[32, 128]
  dtype := .BFloat16
  device := .exact (.CUDA 0)
}
```

`DTensor shape dtype` leaves placement unrestricted. `STensor shape` also leaves
dtype dynamic. A dynamic dtype or `.any` placement makes no claim about that
component; `actualSpec` and `device` report native metadata. `shape`, `dtype`, and
`staticSpec` describe the declared index. Different shape/dtype indices are not
interchangeable.

`RawTensor` is the native handle used by adapters in `Tyr.RawTensor` and its
submodules. It is not the model-facing tensor type. `Tensor.ofRaw` validates
shape, dtype, and placement and returns `Except String (Tensor σ)`.
`Tensor.assumeSpec` bypasses those checks and belongs only at a justified trust
boundary. Static indices alone do not prove a native implementation correct.

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

`Tensor.add`, `sub`, `mul`, and `div` compute result specifications using shape
broadcasting and dtype promotion. Floating-only operations require a floating
dtype. Matrix multiplication requires compatible dimensions and matching dtype
and device policy. `linear` consumes an `[outDim, inDim]` weight matrix;
`linear3d` handles batched sequences.

`reshape` requires equal element counts. `reshapeChecked` validates a requested
runtime shape. In both cases `#[]` means a scalar. `sliceChecked` validates the
claimed result of a runtime slice. Other transforms include `transpose`,
`permute`, `unsqueeze`, `squeeze`, and `unbind`.

Use `Tensor.packShape` to package a runtime-dependent shape existentially.
`catMany?` accepts indexed existential tensors and checks the requested output
specification; `catManyChecked` reports a mismatch through Lean's panic mechanism.
A dynamic shape does not require pretending that a tensor is a scalar.

Reduction indices follow native accumulation rules. For example, integer sums
may accumulate in Int64. `Tensor.Preserving` contains metadata-checked adapters
for operations where preserving an input's generic dtype requires validation.

## Autograd and mutation

`setRequiresGrad`, `detach`, and `gradOf` retain the tensor specification.
`backwardLoss` takes a scalar loss; `backward` takes an output and an explicit
cotangent with the same specification. `hasGradFn` inspects graph ownership.
`copyInto` is an `IO` operation that copies into existing storage.

```lean
import Tyr.Typed
import Tyr.Typed.Neural
open torch

def classifierStep {batch inDim classes : UInt64}
    (weight : DTensor #[classes, inDim] .Float32)
    (input : DTensor #[batch, inDim] .Float32)
    (targets : DTensor #[batch] .Int64) : IO Float := do
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
