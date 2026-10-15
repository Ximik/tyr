# Tensor specifications and checked boundaries

`torch.Tensor σ` is Tyr's public tensor type. Its `StaticSpec` index survives
Lean typechecking and records shape, dtype, and concrete device. The underlying
`RawTensor` is a native LibTorch handle used by implementation adapters.

## Imports and implementation layers

`import Tyr.Typed` provides specifications, tensor constructors, core operations,
and typed layer helpers. Add `import Tyr.Typed.Neural` for additional neural,
data, and persistence operations. `import Tyr` aggregates the library.

| Module | Responsibility |
| --- | --- |
| `Tyr.Basic.Types` | Raw carrier, specifications, dtype/device types, pure shape helpers |
| `Tyr.Basic` | Indexed carrier, checked raw admission, native metadata, native zero factory |
| `Tyr.Typed.DType` | Partial dtype promotion and accumulation rules |
| `Tyr.Typed.Promotion` | Erased proof classes for successful dtype promotion |
| `Tyr.Typed.Spec` | Shape and result-spec algebra |
| `Tyr.Typed.Device` | Static placement eligibility, native-device checks, and tensor contracts |
| `Tyr.Typed.Tensor` | Checked constructors, validation, existential packaging |
| `Tyr.Typed.Ops` | Arithmetic, shape transforms, reductions, autograd, host arrays |
| `Tyr.Typed.Neural` | Neural, data, storage, and dynamic-shape adapters |
| `Tyr.Typed.Layers` | Parameter structures and typed layer operations |
| `Tyr.TensorStruct.Cast` | Convert a tensor tree to the specifications in a destination type |
| `Tyr.TensorStruct.DeriveCast` | Derive field-wise casts between explicitly selected record types |
| `Tyr.RawTensor` and submodules | Native declarations without static guarantees |

## Concrete placement and dtype claims

Every `StaticSpec` names an actual device. Dtype may remain dynamic:

```lean
import Tyr.Typed
open torch

abbrev DynamicDType := Tensor { shape := #[2, 3], device := .CPU }
abbrev FloatWeights := Tensor {
  shape := #[2, 3], dtype := .Float32, device := .CPU
}
abbrev PlacedWeights (device : Device) := Tensor {
  shape := #[2, 3], dtype := .Float32, device := device
}
```

The device field is mandatory; there is no wildcard placement policy or
`forgetDevice`. A runtime-selected `device : Device` can occur in the result
type of a factory or loader. Models share that parameter across their tensor
fields and forward inputs. An unknown placement is packaged existentially,
for example `Sigma fun device => PlacedWeights device`; unpacking it exposes
the same device index for subsequent operations without copying storage.

Omitting dtype uses `.Unknown "dynamic"`. Two such tensors may have different
native dtypes. Only that reserved token is a wildcard; other unrecognized dtype
tokens must match exactly. Preserving adapters inspect actual dtypes and reject
unintended promotion. Copying rejects a dtype mismatch before mutation.

For runtime-selected dtypes, use `SomeDTypeTensor shape device` to retain an
exact existential index. Its `ExactDType` witness excludes the reserved wildcard:

```lean
def packageWeights (tensor : FloatWeights) : SomeDTypeTensor #[2, 3] .CPU :=
  tensor.packDType (by decide)
```

`packDType` packages a statically exact index using an explicit exclusion proof.
`tensor.reifyDType` reads the native dtype and returns a checked exact package;
it also accepts legacy wildcard inputs. `SomeDTypeTensor.ofRaw` validates the
requested shape and device at a native boundary. These operations retain values
and autograd identity without casting or cloning the tensor. The existential
itself is an ordinary Lean wrapper.

Unpacking exposes the exact dtype through `packed.1.val` and the tensor through
`packed.2`. `sameDType?` supplies checked equality evidence before combining two
packages in an operation requiring a shared dtype. `atDType?` checks a requested
exact index; it performs no conversion. Other unknown dtype tokens remain exact
metadata claims and do not promise native kernel support.

Models should retain shared dtype parameters across their related leaves. For
example, `Sigma fun dtype : ExactDType => Qwen3Model cfg dtype.val` preserves
agreement across a Qwen model. Mixed precision models need an explicit tree of
dtype parameters for their distinct tensor roles. The wildcard default,
`forgetDType`, and legacy promoting conveniences remain migration work; the
exact package has no implicit coercion back to those wildcard tensors.

`tensor.staticSpec`, `shape`, `dtype`, and `device` describe the declared index.
`tensor.actualSpec` and `runtimeDevice` inspect native metadata. `tensor.to target`
performs a device conversion and returns a tensor indexed by `target`.
`atDevice?` checks equality with a requested device without transferring storage;
`sameDevice?` exposes checked equality evidence for two device indices.

### Sharing a device across a model

Section parameters become implicit record parameters when tensor fields use them.
The deriving handlers retain those parameters:

```lean
import Tyr.Module.Derive
open torch

variable {device : Device}

structure Projection (inputDim outputDim : UInt64) where
  weight : Tensor {
    shape := #[outputDim, inputDim], dtype := .Float32, device := device
  }
  deriving TensorStruct

def Projection.init (inputDim outputDim : UInt64) (device : Device) :
    IO (Projection (device := device) inputDim outputDim) := do
  let weight ← Tensor.randn #[outputDim, inputDim] (device := device)
  return { weight := ← Tensor.parameter weight }

def Projection.forward {inputDim outputDim batch : UInt64}
    (model : Projection (device := device) inputDim outputDim)
    (input : Tensor ⟨#[batch, inputDim], .Float32, device⟩) :
    Tensor ⟨#[batch, outputDim], .Float32, device⟩ :=
  Tensor.linear input model.weight

def packProjection {inputDim outputDim : UInt64}
    (model : Projection (device := device) inputDim outputDim) :
    Sigma (fun placement => Projection (device := placement) inputDim outputDim) :=
  ⟨device, model⟩
```

The existential package changes only how the device is exposed to the caller;
it does not transfer or clone tensor storage. A requested-device loader can
return `Projection (device := device) ...` directly instead.

## Shape and dtype rules

`TensorSpec.broadcastShapes?` and `matmulShape?` return `Option Shape`.
Broadcasting and generic matrix multiplication require compatibility evidence;
concrete valid dimensions are usually discharged automatically. Specialized
operations such as `linear`, `linear3d`, and `bmm4d` encode their dimension
relationships directly in their argument types.

`DType.all` contains all 46 concrete scalar types in the pinned PyTorch 2.10
runtime: ordinary numeric types, Complex32/64/128, quantized types, packed bits,
low-bit integers, and Float8/Float4 formats. Complex widths count both real and
imaginary components. `DType.ofString?` accepts canonical, native, and Python
names such as `Complex64`, `ComplexFloat`, and `torch.complex64`. Recognition
does not promise allocation, conversion, or arithmetic kernels on every backend.
SafeTensors support remains limited to the formats returned by `safeTensorTag?`.

Dtype promotion is partial. `DType.promote? l r` returns `some result` only when
PyTorch defines the pair. Typed arithmetic carries this result as an erased
`Prop` witness, `DType.Promotes l r result`, whose `valid` field proves
`DType.promote? l r = some result`. The result parameter is inferred for concrete
dtypes. Generic forward functions carry the constraint explicitly:

```lean
import Tyr.Typed
open torch

-- Tensor.add has this same shape/dtype/device signature.
def combine {s : Shape} {l r result : DType} {dev : Device}
    [DType.Promotes l r result]
    (left : Tensor ⟨s, l, dev⟩) (right : Tensor ⟨s, r, dev⟩) :
    Tensor ⟨s, result, dev⟩ :=
  Tensor.add left right

example : DType.Promotes .Float16 .BFloat16 .Float32 := inferInstance
example : DType.Promotes .Complex32 .BFloat16 .Complex64 := inferInstance
example {l r result : DType} [h : DType.Promotes l r result] :
    DType.promote? l r = some result := h.valid

example : DType.ScalarPromotes .Float32 .Float64 .Float32 := inferInstance
example : DType.NonBool .Float32 := inferInstance
```

Unknown dtypes, including `.Unknown "dynamic"`, have no static promotion
witness. Unsupported mixed shell-type pairs, such as Float8E4M3FN with Float32,
also have no witness; callers must choose an explicit conversion. A promotion
witness proves that Tyr's promotion function returns that dtype; it is not a
proof of the native implementation or kernel availability. Operator conditions
are separate: `Tensor.sub` additionally requires `DType.NonBool` for each operand.
Integer sums use their specified accumulation dtype. Operator-specific domain
witnesses reject invalid static inputs, such as complex ReLU or integral mean;
checked variants inspect actual dtypes when those indices are dynamic.

Broadcasting distinguishes zero-dimensional scalar tensors from nonscalars:
within the same dtype category, the nonscalar's precision wins. A Float64 scalar
plus a Float32 vector therefore returns Float32. This is different from adding
two nonscalar Float64/Float32 tensors.
`DType.ScalarPromotes` expresses this rule; host `Float` arithmetic uses the
Float32 default scalar category. Broadcasting uses `TensorSpec.Promotes`, which
selects the appropriate rule from the operand ranks.

Runtime-typed parameter trees can use the explicit `Tensor.Preserving.add?`,
`sub?`, and `mul?` adapters. They inspect actual dtypes before arithmetic and
reject undefined promotion or a result that would change the left operand's
dtype. They then validate the result's shape, dtype, and device. The host-scalar
adapters `addScalar?`, `mulScalar?`, and `divScalar?` likewise check the actual
dtype against weak scalar promotion. These adapters do not
introduce a static promotion instance or conceal a change behind a dynamic
index; the corresponding convenience wrappers use Lean's panic fallback.
For example, a dynamic BF16 left operand and Float32 right operand are rejected
because their result would be Float32. Mixed-precision algorithms must make an
intended conversion explicit before using a preserving adapter, or expose the
promoted result through a witness. The checked arithmetic adapters also return
native kernel failures through `Except.error`.

Use `Tensor.Promoting.add?`, `sub?`, `mul?`, and `div?` when input dtype indices
are dynamic and the operation should follow native promotion. They inspect the
actual operand dtypes, construct a promotion witness, and validate the result
at its computed dtype. The returned `Promoted` package exposes `dtype`,
`witness`, and `tensor`; division additionally applies its integer-to-floating
result rule. Host-scalar variants such as `mulScalar?` return `ScalarPromoted`
with a `ScalarPromotes` witness. No input
cast is needed. Their convenience wrappers, such as `Tensor.Promoting.mul`,
return a shape-indexed tensor with dynamic dtype, retaining the promoted native
value. For example, BF16 activations multiplied by Float32 routing weights
produce Float32, while BF16 activations multiplied by an Int64 mask remain BF16.
The `?` arithmetic variants report unsupported promotion, placement mismatches,
and native kernel failures. A valid promotion witness still does not guarantee
that a kernel exists.

Runtime broadcasting uses `Tensor.Promoting.addB?`, `subB?`, `mulB?`, and `divB?`.
They take an explicit output shape and check it against the broadcast result
before native execution. Their `PointwisePromoted` package carries
`TensorSpec.Promotes` evidence tied to the operands' actual ranks and dtypes.
`Tensor.whereSelect` promotes its two value operands and requires a Bool
condition; `Tensor.Promoting.whereSelect?` provides the corresponding checked
runtime witness. Selection does not cast the result back to the left dtype.

Matrix multiplication requires operands with matching dtype and concrete device.
The checked `matmul?`, `mm?`, `linear?`, `linear3d?`, `matmul3d?`, and `bmm4d?`
adapters verify actual dtype and device equality before calling native code.
Elementwise promotion never supplies an implicit matrix cast.

`Tensor.matrixLog?` accepts standard real floating matrices, including supported
nonsymmetric matrices and real rotations. Its eigendecomposition-based solver
rejects non-real or nonfinite outputs and checks that exponentiating the result
reconstructs the input with a dtype-scaled relative tolerance. Defective or
numerically unstable inputs can return an error even when a real logarithm
exists; this is a solver limitation, not a restriction to positive definite
matrices. Use the checked variant when those numerical failures must be handled.

## Device compatibility

A tensor's concrete device is part of its type. Equal device indices tie inputs
to the same placement; independently loaded existential tensors must establish
equality before a shared-device operation. Checked adapters also inspect native
metadata at the FFI boundary. Matrix multiplication and concatenation require
equal actual devices.

Broadcasting arithmetic requires `Device.PointwiseCompatible leftSpec rightSpec`.
Its instances reject CPU/CUDA vectors and CUDA:0/CUDA:1 vectors. The binary
pointwise exception permits a rank-zero CPU tensor on the other operand's
device; a size-one vector and an accelerator scalar do not get that exception.
The result follows the non-CPU operand when this exception applies.

```lean
import Tyr.Typed
open torch

example (scalar : Tensor ⟨#[], .Float32, .CPU⟩)
    (vector : Tensor ⟨#[2], .Float32, .CUDA 0⟩) :
    Tensor ⟨#[2], .Float32, .CUDA 0⟩ := scalar.addB vector
```

The same-shape `add`/`sub`/`mul`/`div` APIs require one shared device index;
use their broadcasting variants when the devices differ, including for two
scalar tensors. `whereSelect` follows a separate native rule: it selects
the first non-CPU device in condition, left, right order, and relocates rank-zero
operands to that device. This includes accelerator scalars. Non-scalars must
already occupy the selected device. Tyr checks this rule before native execution,
matching the [pinned PyTorch implementation](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/TensorCompare.cpp#L541-L594).

`Tests.TestDeviceCompatibility` covers static rejection, metadata rules for
CPU/MPS/CUDA indices, and CPU numerical execution. The accelerator metadata cases
do not allocate accelerator tensors or establish hardware qualification.

## Operator checks and failures

Static promotion, domain, and shape evidence is separate from backend kernel
availability. `relu?`, `maxAll?`, `sumDim?`, and `meanAll?` check their supported
domains, including nonempty maximum inputs and valid reduction axes. `expand?`
checks the requested broadcast, while `cat?` and `catMany?` validate placement
and the returned shape. These APIs and the arithmetic/matrix `?` families return
native operation failures through `Except String`. Their pure convenience
wrappers use Lean's panic fallback; callers that need to handle failure should
use the checked variants. This contract applies to those adapters, not every
raw native declaration. Static rules and metadata checks do not prove an extern's
implementation correct.

`argmax`, `maxLast3d`, and `topKLast2d` require `DType.Ordered` evidence.
`argmax` also requires a valid, nonempty reduction axis; scalar input permits
axis zero. `maxLast3d` requires a nonempty last axis, while `topKLast2d` requires
`k ≤ cols` and permits `k = 0`. Concrete dimensions usually discharge these
proofs automatically. For runtime dimensions or dynamic dtypes, the matching
`?` variants return domain, range, and native errors through `Except String`.
The `argmaxChecked`, `maxLast3dChecked`, and `topKLast2dChecked` conveniences
perform the same runtime checks and use Lean's panic fallback on failure.
Selected values retain the actual input dtype; indices have dtype Int64.

`crossEntropy` and `crossEntropyNone` require `DType.RealFloating` for logits
of shape `[N, C]` and Int64 targets of shape `[N]` on the same device.
The first returns a scalar mean loss; the second returns one loss per row.
Their checked native boundary validates labels before loss dispatch: each label
must be in `[0, C)` or equal the default ignored label `-100`.
`crossEntropy?` and `crossEntropyNone?` recover dtype, label, and native failures.
For dynamic dtype indices, `crossEntropyChecked` and `crossEntropyNoneChecked`
provide pure runtime-checked conveniences with the same panic fallback.
`ChatSFT.maskedCrossEntropy?` retains errors through reshaping, mask arithmetic,
reduction, and scalar extraction. Its mask must have a real numeric or Bool
dtype; the pure wrapper requires that domain as type-level evidence.

`indexSelect?` validates vector indices before native indexing, including on
accelerators, and preserves the data dtype and device. `TokenBytes` uses it
through IO so invalid token IDs return errors. `sumAll?` and `item?` retain
reduction and host-scalar conversion failures for recoverable callers.

`conv1d?`, `conv2dBias?`, `conv1dGroups?`, and `convTranspose1d?` check actual
dtype and device equality across input, weight, and any bias before invoking
checked native kernels. Convolution does not implicitly promote its operands,
including when their dtype indices are dynamic. Invalid convolution parameters
and unavailable kernels return `Except.error`; successful outputs are checked
against the computed spatial shape and input dtype/device. The corresponding
names without `?` use Lean's panic fallback on failure.
The `conv2dBias` result shape broadcasts singleton stride, padding, and dilation
arrays across both spatial axes, matching the native operation.

After building the dtype modules, run `bash scripts/test_dtype_promotion.sh`
to compare all 46² promotion pairs and all four scalar/vector rank combinations
against an independently enumerated native runtime inventory. The harness also
checks arithmetic result dtypes separately from unavailable kernels.

## Attention and dropout

Attention separates pure inference from stochastic execution. `Tensor.attention`,
the GQA/masked/window variants, and `Tensor.flashAttention` accept no dropout
probability and use zero dropout. Their `?` forms retain validation and native
kernel errors. To train with dropout, bind the corresponding IO operation:

```lean
let output ← Tensor.attentionIO query key value dropout training isCausal
```

The required dropout and training arguments follow the tensor operands;
remaining causal, GQA, mask, window, and scale options retain their operation's
order. Training uses the supplied probability; evaluation uses zero. The IO
adapters first validate the original probability as finite and in `[0, 1]`,
including during evaluation. The operation preserves the native autograd graph,
including the sampled mask.
Pure attention's zero-dropout contract is not a guarantee of bitwise
reproducibility across devices or backend algorithms.

## Boundary validation

```lean
import Tyr.Typed
open torch

def checkRaw (raw : RawTensor) : Except String (Tensor { shape := #[2, 3], dtype := .Float32, device := .CPU }) :=
  Tensor.ofRaw raw
```

`Tensor.ofRaw` checks all claims in the destination index. `Tensor.ofTensor` and
`ofTensor?` check a requested shape and dtype and index the result by
`raw.device`; `ofTensorWithContract` also checks an optional expected placement. `Tensor.validate` audits an existing indexed
value against its current native metadata.

`TensorContract` contains a `TensorSpec`, an optional `expectedDevice`, and a descriptive
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

def matchingPositions (tokens : Tensor { shape := #[4], dtype := .Int64, device := .CPU }) :=
  Tensor.findTokenPositions tokens 7
```

This returns `IO (Sigma fun count => Tensor { shape := #[count], dtype := .Int64, device := .CPU })`.
`Tensor.packShape` packages an existing indexed tensor's shape while retaining
its dtype and device parameters. `catMany?` accepts an array of such
packages and validates a requested result shape. `Tensor.toSome` instead returns
`Sigma Tensor`, retaining the complete specification as an existential witness.
A scalar is `#[]`; it is never a substitute for an unknown shape.

`reshape` requires equal element counts. `reshapeChecked?` checks runtime
element counts and rejects wrapped negative extents before invoking the native
reshape. `castDType?` and `castLike?` perform real dtype conversions and validate
the result, returning native conversion failures through `Except`. Their pure
convenience wrappers, along with `sliceChecked`, use Lean's panic fallback on
failure; use the `Except` variants when an
error must be handled. Exact file loading checks the stored layout
before returning a tensor; it does not repair a mismatch by reshaping.

## Models, differentiation, and persistence

`TensorStruct` traversals preserve each leaf's complete specification. Typed
parameters therefore retain shape, dtype, and device information through
mapping, folding, flattening, and generated AD boundaries.

Use `TensorStruct.castTo TargetType source` to choose each leaf's dtype and
placement in the destination Lean type. For example, a model can request BF16
weights and retain Int64 indices:

```lean
import Tyr.TensorStruct.DeriveCast
open torch

structure StoredWeights where
  weight : Tensor { shape := #[2, 3], dtype := .Float32, device := .CPU }
  indices : Frozen { shape := #[2], dtype := .Int64, device := .CPU }
  config : Static String

structure RuntimeWeights where
  weight : Tensor {
    shape := #[2, 3], dtype := .BFloat16, device := .CPU
  }
  indices : Frozen { shape := #[2], dtype := .Int64, device := .CPU }
  config : Static String

derive_tensor_tree_cast StoredWeights => RuntimeWeights

def prepareWeights (source : StoredWeights) : IO RuntimeWeights :=
  TensorStruct.castTo RuntimeWeights source
```

The destination type supplies every leaf specification. A concrete dtype
requests a native cast, and the destination device requests placement on that device.
`castTo` requests dtype and placement through one checked native entry point.
That entry point chooses the conversion order for the backend: MPS-to-CPU
transfers retain the source dtype until they reach CPU, then apply the requested
cast. This lets an MPS Float32 tensor request CPU Float64 without an unsupported
MPS Float64 intermediate.
A dynamic target dtype retains the source's actual dtype. The target always
names a concrete device. Every leaf keeps its shape, so callers perform any reshape
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
