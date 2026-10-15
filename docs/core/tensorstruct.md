# TensorStruct: traversable indexed tensor trees

`torch.TensorStruct` provides generic traversal of model parameters and state.
Every tensor leaf retains its complete `StaticSpec`: shape, dtype, and concrete
device. Optimizers, checkpointing, and AD can operate over parameter structures
without erasing those indices.

## Traversal contract

The four methods have the following signatures:

```lean
class TensorStruct (α : Type) where
  map : (∀ {σ}, Tensor σ → Tensor σ) → α → α
  mapM {m : Type → Type} [Monad m] : (∀ {σ}, Tensor σ → m (Tensor σ)) → α → m α
  zipWith : (∀ {σ}, Tensor σ → Tensor σ → Tensor σ) → α → α → α
  fold {β : Type} : (∀ {σ}, Tensor σ → β → β) → β → α → β
```

The leaf function is polymorphic over the specification. A traversal can visit
many different shapes and dtypes, but `map` cannot silently change a leaf's
specification. `mapM` supports effectful operations such as checked loading.
`zipWith` combines corresponding leaves, and `fold` summarizes them.

The indexed tensor instance is defined in `Tyr.TensorStruct`. Import
`Tyr.Module.Derive` to use the deriving handler, or import the `Tyr` umbrella.

## Deriving parameter trees

```lean
import Tyr.Module.Derive
open torch

structure LinearParameters (inDim outDim : UInt64) where
  weight : Tensor { shape := #[outDim, inDim], dtype := .Float32, device := .CPU }
  bias : Tensor { shape := #[outDim], dtype := .Float32, device := .CPU }
  name : Static String
  deriving TensorStruct

def initialized : LinearParameters 3 2 := {
  weight := Tensor.ones #[2, 3]
  bias := Tensor.zeros #[2]
  name := "projection"
}

example : TensorStruct.count initialized = 2 := rfl
```

The handler delegates each field to its own instance. Each field must therefore
have a suitable traversal instance. `deriving Model` additionally derives
`ToTensorStructSchema` and `TensorStructFlatten` for schema-aware AD use.
Generated schemas retain tensor metadata and distinguish parameter participation.

## Containers and traversal policies

| Container | Behavior |
| --- | --- |
| `Tensor σ` | One indexed leaf |
| `ShapePackedTensor dtype device` | Existential shape with a fixed dtype and concrete device; zip checks shape equality |
| `Array α`, `List α` | Elementwise traversal; zip panics on unequal lengths |
| `Vector n α` | Length-indexed traversal and zip |
| `Option α` | Traverses `some`; mismatched zip variants yield `none` |
| `Static α` | Skipped by all tensor traversals |
| `Frozen σ` | Wraps `Tensor σ`; ordinary traversal still visits it |

`Static` is appropriate for names and configuration. Common scalar metadata types
also have no-tensor instances. `Frozen` describes intended participation but does
not make ordinary `TensorStruct.map` skip a tensor. An optimizer must honor its
participation policy; wrapping a value alone does not prevent arbitrary updates.

`Vector n α` stores an array with a proof of its length. Its operations include
`map`, `mapM`, `zipWith`, `foldl`, `replicate`, `get`, `toArray`, and `push`.
Use it when the number of layers or states is known. It checks the outer count;
nested runtime-dependent structures can still need their own checks.

## Utilities

All utilities below require `[TensorStruct α]`:

- `count` counts tensor leaves.
- `grads : α → IO α` obtains an independent gradient snapshot tree;
  `zeroGrads : α → IO Unit` clears the parameters' accumulated gradients.
- `detach : α → α` removes graph participation explicitly.
- `requiresGrad : α → Bool → IO α` changes autograd flags.
- `makeLeafParams : α → IO α` creates independent trainable leaves.
- `functionalGrad` differentiates a scalar function using private temporary leaves
  and returns detached gradients without accumulating parameter gradients.
- `scale`, `add`, and `sub` transform matching parameter trees.
- `all` and `any` evaluate specification-polymorphic leaf predicates.

Generic scalar scaling uses a preserving adapter. Its runtime metadata check
matters for dtypes where native scalar arithmetic could change the result dtype.
A traversal preserves indices, not unique storage ownership; use explicit
allocation/copy APIs when independent mutable storage is required.

## Casting to a destination type tree

`TensorStruct.castTo TargetType source` uses each tensor specification in the
destination Lean type to choose its dtype and device. The result has `TargetType`;
each leaf keeps its shape, and container lengths and `Static`/`Frozen` roles are
preserved. Dtype and device changes happen together in one native conversion.
Validation and native conversion errors report the affected field or index.

Import `Tyr.TensorStruct.Cast` for supported leaf and container conversions. For
records, import `Tyr.TensorStruct.DeriveCast` and declare
`derive_tensor_tree_cast SourceType => TargetType`. This matches fields by name
and order without allocating a destination template. See the
[complete record example](typed.md#models-differentiation-and-persistence).

`TensorStruct.castDType dtype source` is the uniform convenience for dynamic
dtype leaves within the same declared type; concrete leaf specifications still
apply. Ordinary `map` and `mapM` retain each leaf's complete specification.

## Persistence and AD

`torch.checkpoint.saveParams` traverses a tree to write its leaves.
`loadParams` uses a template tree to supply each leaf's full specification and
loads exact stored shapes. Stored tensor-count, dtype, and shape mismatches are
errors. The caller can choose whether loaded tensors become trainable parameters.

Schema and flattening support lives in `Tyr.AD.TensorStructSchema`. The AD
frontend recognizes indexed `Tensor` and `Frozen` parameters and checks declared
dtype metadata against the generated boundaries. See [autodiff](../autodiff.md)
and [serialization](../serialization.md).

## Related guides

- [Tensor operations](tensors.md)
- [Specifications and validation](typed.md)
- [Modules](../modules.md)
- [Optimization](../optimization.md)
