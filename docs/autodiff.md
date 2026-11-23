# Automatic differentiation

Differentiation uses LibTorch autograd: runtime, tape-based reverse mode on `Tensor σ`, exposed through `Tensor.backwardLoss`, `Tensor.backward`, `Tensor.grad`, and `Tensor.gradOf`.

Day-to-day training uses libtorch's tape through the FFI. The idiom is
`Tensor.backwardLoss loss` followed by `let gradient ← Tensor.gradOf param`
(usually `let gradients ← TensorStruct.grads params`). Gradient reads return
independent snapshots; `TensorStruct.zeroGrads params` clears accumulated
gradients in IO without changing those snapshots. Leaf parameters are prepared
with `let parameter ← Tensor.parameter t` or `let params ← TensorStruct.makeLeafParams tree`.
These effectful calls create independent gradient identities. Backward releases saved
intermediates by default. Use `(retainGraph := some true)` when another pass through
the same graph is needed, and release on the final pass. Omitted retention follows
`createGraph`, whose default is false. `Tensor.grad` and `TensorStruct.grad` return
gradients in IO without accumulating parameter gradients and use the same options.
Pure forward computations may be shared; repeating a pure expression does not
guarantee a fresh graph. Evaluation explicitly detaches models
and floating inputs; ambient `no_grad` scopes are no longer exposed.
See [core/tensors.md](core/tensors.md), [core/tensorstruct.md](core/tensorstruct.md),
and [optimization.md](optimization.md).

## Related guides

- [core/tensors.md](core/tensors.md) — `torch.autograd` FFI bindings
- [core/tensorstruct.md](core/tensorstruct.md) — parameter-tree traversal (`TensorStruct.grads`, `zeroGrads`)
- [modules.md](modules.md) — layers and the `deriving Model` pipeline
- [optimization.md](optimization.md) — optimizers and training loops built on the FFI path
