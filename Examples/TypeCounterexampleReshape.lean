/-
  TypeCounterexampleReshape.lean

  `reshapeChecked`, which the models use instead of `reshape` (over a thousand
  call sites), takes no side condition: the element count is compared only at
  runtime. Every definition below compiles; the wrong ones abort the process
  when they run. Deliberately not imported by `Examples.lean`. `lake lean`
  builds the imports and loads Tyr's native code, so the `#eval` at the bottom
  actually runs:

    lake lean Examples/TypeCounterexampleReshape.lean
-/
import Tyr.Typed
open torch

/-- `StaticSpec` has no default device; fix it to CPU so the cases
    read as shape + dtype only. -/
abbrev DTensor (shape : Shape) (dtype : DType) : Type :=
  Tensor { shape, dtype, device := .CPU }

-- Correct.
def splitHeads {b : UInt64} (x : DTensor #[b, 512] .Float32) : DTensor #[b, 8, 64] .Float32 :=
  x.reshapeChecked #[b, 8, 64]

-- Correct.
def flatten {b s : UInt64} (x : DTensor #[b, s, 512] .Float32) : DTensor #[b * s, 512] .Float32 :=
  x.reshapeChecked #[b * s, 512]

-- Wrong: 512 ≠ 480 elements per row.
def wrongSplit {b : UInt64} (x : DTensor #[b, 512] .Float32) : DTensor #[b, 8, 60] .Float32 :=
  x.reshapeChecked #[b, 8, 60]

-- Wrong unless b = s.
def wrongVar {b s : UInt64} (x : DTensor #[b, s] .Float32) : DTensor #[b, b] .Float32 :=
  x.reshapeChecked #[b, b]

-- The type claims `#[2, 8, 60]`; running it aborts the process.
#eval (wrongSplit (Tensor.ones #[2, 512])).raw.runtimeShape
