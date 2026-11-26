-- Core infrastructure
import Tyr.Core.FFI.Basic
import Tyr.IO.Log
import Tyr.Core.Util
import Tyr.Core.Torch
import Tyr.Core.Typed
import Tyr.IO.SafeTensors
import Tyr.Core.TensorStruct
import Tyr.Core.TensorStruct.Cast
import Tyr.Core.TensorStruct.DeriveCast
import Tyr.Core.TensorStruct.Indexed
import Tyr.Core.TensorStruct.Schema

-- Neural network modules
import Tyr.Lib.Module
import Tyr.Lib.Quantization.FP8

-- Optimizers
import Tyr.Lib.Optim
import Tyr.Lib.Optim.ManifoldMuon
import Tyr.Lib.Optim.RiemannianSGD
import Tyr.Lib.Optim.RiemannianTreeSGD
import Tyr.Lib.Optim.Scheduler
import Tyr.Lib.Optim.DualOptimizer
import Tyr.Lib.Optim.DistAdam

-- Modular and manifold optimization primitives
import Tyr.Lib.Optim.Modular
import Tyr.Lib.Optim.Manifolds

-- Distributed training
import Tyr.Lib.Distributed
import Tyr.Lib.Distributed.Sharding

-- Data loading and tokenization
import Tyr.Lib.Data
import Tyr.Lib.Data.Loader
import Tyr.Lib.Tokenizer

-- HuggingFace Hub utilities
import Tyr.IO.Hub

-- Shared model utilities and generation infrastructure
import Tyr.Lib.Inference.Generation

-- Checkpointing
import Tyr.IO.Checkpoint

-- Pipeline orchestration
import Tyr.Lib.Train.Pipeline

-- Training loops
import Tyr.Lib.Train.RunLedger
import Tyr.Lib.Train.ChatSFT
import Tyr.Lib.Train.GRPO


/-!
# Tyr

`Tyr` is a dependently typed deep learning library for Lean 4.
Its central goal is to make tensor programs safer by tracking key invariants
(especially shapes) in types while still exposing practical training and inference tooling.

## Major Components

This root module re-exports the core, general-purpose surface area:

- `Tyr.Core.FFI.Basic`: foundational tensor aliases and shared utilities.
- `Tyr.Core.Torch`: low-level tensor operations and libtorch bindings.
- `Tyr.Core.Typed`: tensor wrappers with static shape/dtype checks and checked raw boundaries.
- `Tyr.IO.SafeTensors`: typed SafeTensors schema and loading support.
- `Tyr.Core.TensorStruct`: generic traversal/mapping over tensor-containing structures.
- `Tyr.Core.TensorStruct.Cast`: conversion to a destination tree of tensor types,
  with per-leaf dtype/device specifications and no allocated destination template.
- `Tyr.Core.TensorStruct.Indexed`: `Select`, a value whose type is chosen by a `Bool` index, and
  `IndexedList`, a sequence whose element types depend on the position.
- `Tyr.Lib.Module`: neural network module abstractions and layers.
- `Tyr.Lib.Optim` (+ scheduler/dist variants): optimizers and learning-rate scheduling.
- `Tyr.Lib.Distributed` and `Tyr.Lib.Distributed.Sharding`: distributed training and partitioning utilities.
- `Tyr.Lib.Data`, `Tyr.Lib.Data.Loader`, `Tyr.Lib.Tokenizer`: data ingestion, batching, and tokenization.
- `Tyr.IO.Checkpoint`: model/optimizer checkpoint save and restore.
- `Tyr.Lib.Train.Pipeline`: training/inference orchestration helpers.

## Example

```lean
import Tyr

open torch

def toyForward : Tensor { shape := #[2, 4], dtype := .Float32 } :=
  let x := Tensor.zeros #[2, 8]
  let w := Tensor.zeros #[4, 8]
  x.linear w
```

`import Tyr` re-exports the indexed tensor API and runtime services.
`Tensor σ` retains shape, dtype, and device-policy claims. Native adapters use
`RawTensor`; `Tensor.ofRaw` checks metadata when crossing that boundary.

## Scope

This file is intentionally a stable import surface for common Tyr workflows.
Task- and model-specific code lives in dedicated modules (for example `Examples` and `Tyr.Model`).
-/
