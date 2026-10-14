-- Core infrastructure
import Tyr.Basic
import Tyr.Log
import Tyr.Torch
import Tyr.Typed
import Tyr.SafeTensors
import Tyr.TensorStruct
import Tyr.Mctx
import Tyr.MctxDag

-- AutoGrad (JAX-style AD on IR)
import Tyr.AD

-- Event-skeleton representation for hybrid/stochastic differentiation
import Tyr.EventSkeleton

-- Neural network modules
import Tyr.Module

-- Optimizers
import Tyr.Optim
import Tyr.Optim.ManifoldMuon
import Tyr.Optim.RiemannianSGD
import Tyr.Optim.RiemannianTreeSGD
import Tyr.Optim.Scheduler
import Tyr.Optim.DualOptimizer
import Tyr.Optim.DistAdam

-- Modular and manifold optimization primitives
import Tyr.Modular
import Tyr.Manifolds

-- Distributed training
import Tyr.Distributed
import Tyr.Sharding

-- Data loading and tokenization
import Tyr.Data
import Tyr.DataLoader
import Tyr.Tokenizer
import Tyr.Text

-- HuggingFace Hub utilities
import Tyr.Hub

-- Shared model utilities and generation infrastructure
import Tyr.Model.Utils
import Tyr.Model.Generation

-- Checkpointing
import Tyr.Checkpoint

-- Pipeline orchestration
import Tyr.Pipeline

-- Training loops
import Tyr.Train.RunLedger
import Tyr.Train.ChatSFT
import Tyr.RL.GRPO

-- GPU kernel abstraction
import Tyr.GPU
import Tyr.GPU.AD
import Tyr.GPU.AutoGrad

/-!
# Tyr

`Tyr` is a dependently typed deep learning library for Lean 4.
Its central goal is to make tensor programs safer by tracking key invariants
(especially shapes) in types while still exposing practical training and inference tooling.

## Major Components

This root module re-exports the core, general-purpose surface area:

- `Tyr.Basic`: foundational tensor aliases and shared utilities.
- `Tyr.Torch`: low-level tensor operations and libtorch bindings.
- `Tyr.Typed`: tensor wrappers with static shape/dtype checks and checked raw boundaries.
- `Tyr.SafeTensors`: typed SafeTensors schema and loading support.
- `Tyr.TensorStruct`: generic traversal/mapping over tensor-containing structures.
- `Tyr.Mctx`: Monte Carlo tree search infrastructure.
- `Tyr.MctxDag`: DAG-oriented counterpart to `Tyr.Mctx` with sharing semantics.
- `Tyr.AutoGrad`: JAX-style automatic differentiation over Tyr IR.
- `Tyr.EventSkeleton`: event-skeleton representation for hybrid/stochastic differentiation.
- `Tyr.Module`: neural network module abstractions and layers.
- `Tyr.Optim` (+ scheduler/dist variants): optimizers and learning-rate scheduling.
- `Tyr.Distributed` and `Tyr.Sharding`: distributed training and partitioning utilities.
- `Tyr.Data`, `Tyr.DataLoader`, `Tyr.Tokenizer`: data ingestion, batching, and tokenization.
- `Tyr.Text`: streaming text consensus and voice-activity (VAD) providers.
- `Tyr.Checkpoint`: model/optimizer checkpoint save and restore.
- `Tyr.Pipeline`: training/inference orchestration helpers.
- `Tyr.GPU` (+ `Tyr.GPU.AD`, `Tyr.GPU.AutoGrad`): GPU kernel abstractions and AD support.

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
