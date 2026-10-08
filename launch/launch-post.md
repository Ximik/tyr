# Launch copy

## Primary announcement

I am making Tyr public as a research preview.

Tyr is an experimental machine-learning and scientific-computing system written
in Lean 4. The project started from a relatively narrow question: which errors
in tensor programs can be turned into type errors without giving up a practical
numerical runtime? Tensor shapes and dtypes are part of the types used by the
core operations, while numerical execution is provided by native backends.

The same question recurs at other boundaries. Tyr contains a SafeTensors type
provider that generates checked declarations from checkpoint headers during
elaboration, ODE and SDE solvers with adjoint methods, an event representation
for differentiating hybrid systems, and a typed GPU language with CUDA code
generation and numerical parity tests.

The accompanying article gives several executable examples:

- an incompatible tensor projection rejected during elaboration;
- a URDF-derived contact trajectory with a saltation update and a derivative
  through the velocity reset;
- checkpoint headers turned into typed loader APIs;
- a Qwen3.6 27B checkpoint executed through Tyr on CUDA; and
- the general GPU compiler suite, with native execution reported separately as
  a named-device measurement.

Tyr is not a production framework, and not every property of a numerical
program is statically checked. The article distinguishes compile-time
contracts from runtime tests and numerical comparison.

I would be interested in talking to people working on compilers, automatic
differentiation, hybrid systems, scientific machine learning, and GPU runtimes.

Repository: https://github.com/ranvier-labs/tyr

Article: [article URL]

## Short version

I am making Tyr public as a research preview.

Tyr is an experimental machine-learning and scientific-computing system in
Lean 4. It explores typed tensor operations, schema-generated checkpoint APIs,
differentiation through hybrid events, and typed GPU programs connected to
practical numerical backends.

The article contains the examples, commands, raw outputs, and present
limitations.

https://github.com/ranvier-labs/tyr

## Thread

**1/** I am making Tyr public as a research preview. It is an experimental
machine-learning and scientific-computing system written in Lean 4.

The starting question is which structural assumptions in an ML program can be
made part of the program itself. https://github.com/ranvier-labs/tyr

**2/** The small example is a tensor projection. Its batch and feature
dimensions occur in the type. A transposed weight is rejected during Lean
elaboration, before numerical execution.

[attach Figure 1]

**3/** Shapes are only one boundary. Tyr also generates typed APIs from
SafeTensors headers, implements ODE/SDE solvers and adjoints, represents
discrete events in hybrid dynamics, and lowers typed GPU programs to CUDA.

**4/** One example differentiates a URDF-derived contact system. The forward
trajectory contains a velocity jump at impact; the reverse pass applies a
saltation update and computes the derivative with respect to restitution.

[attach Figure 2]

**5/** Tyr also has a recorded Qwen3.6 27B CUDA inference run. I use this as a
runtime integration test, not as the main result. GPU execution tests are
reported separately with the device, generated kernel, reference, and
tolerance.

**6/** This is a research codebase. The article states which properties are
checked statically and which are established by runtime tests.

[article URL]
