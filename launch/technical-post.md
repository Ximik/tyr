# Tyr: types as part of the machine-learning system

Tyr is an experimental machine-learning and scientific-computing system written
in Lean 4. It investigates which assumptions can be made explicit in a program
while retaining practical tensor, model, solver, and GPU execution paths.

Machine-learning programs distribute structural assumptions across source
code, checkpoint metadata, compiler configuration, and runtime assertions. A
tensor operation may only be defined for a particular shape, a checkpoint field
may be required by a model, and a generated kernel may only be valid on a
subset of devices. These conditions are often discovered only when the
corresponding path is executed.

Tyr explores a different arrangement. Tensor shapes and dtypes can enter Lean
types; checkpoint headers can generate typed declarations during elaboration;
solver and event structure can be represented explicitly; and GPU capabilities
can constrain kernel programs before code generation. This does not amount to
formal verification of floating-point execution. The system combines static
contracts, runtime validation, and numerical comparison.

## Why dependent types?

Many assumptions in an ML program are relations between values. The output
width of one layer must equal the input width of the next; two operands must
have compatible dtypes and devices; a reshape must preserve the number of
elements. An ordinary tensor type says that a value is a tensor. A dependent
tensor type can also state its shape, dtype, or device policy, because values
such as dimensions occur in the type itself.

This moves a class of consistency checks from execution to program
elaboration. The relationships are compositional: if one function returns
`Tensor { shape := #[batch, hidden] }` and the next requires that shape, Lean checks their
connection even when `batch` and `hidden` are symbolic. The point is not to
encode every property of a learned system in a type. Data-dependent dimensions,
imported checkpoints, device availability, allocation, and floating-point
behavior remain runtime concerns. Tyr uses types for stable structural
relations and runtime checks at external boundaries.

## A shape-dependent operation

The contract for a linear projection makes the relationship explicit:

```lean
def linear {m n b : UInt64} {d : DType} {dev : DevicePolicy}
    (x      : Tensor ⟨#[b, m], d, dev⟩)
    (weight : Tensor ⟨#[n, m], d, dev⟩) :
    Tensor ⟨#[b, n], d, dev⟩ :=
  Tensor.linear x weight
```

The same `m` occurs in both inputs, `b` is preserved, and `n` becomes the
output width. The dtype and device policy must also agree.

A small use site specializes that contract to ordinary model dimensions:

```lean
def project
    (x : Tensor { shape := #[32, 768], dtype := .Float32 })
    (weight : Tensor { shape := #[512, 768], dtype := .Float32 }) :
    Tensor { shape := #[32, 512], dtype := .Float32 } :=
  linear x weight
```

Changing the weight to `#[768, 512]` produces an application type mismatch
during elaboration. The numerical runtime is never entered. Both cases can be
run with:

```bash
./scripts/launch/run_shape_safety.sh
```

## What is compiled, and what runs?

Lean first elaborates `project` and unifies the dimension variables in
`linear`. In the valid case it resolves `b = 32`, `m = 768`, and `n = 512`.
The invalid case cannot assign one value to the two occurrences of `m`, so
compilation stops at this stage.

The accepted program does not carry a second, shape-rich tensor
representation at runtime. Tyr's `Tensor σ` is a typed view whose
`StaticSpec` index is phantom; its only field is the existing raw tensor
handle. Lean compiles the ordinary program and its call to the external symbol
`lean_torch_linear`. The C++ bridge passes the two handles to
`torch::linear`, and LibTorch dispatches according to their actual device
placement.

The type checker therefore establishes the wiring relation, while LibTorch
remains responsible for allocation and numerical execution. Tyr's lower-level
typed GPU language is a separate path: it generates CUDA rather than calling
this LibTorch operator.

## Differentiation through a contact event

Hybrid systems combine continuous dynamics with discrete state changes. The
URDF contact example describes a sphere falling under gravity and colliding
with a plane. Tyr localizes the contact at `τ = 0.2`, applies the velocity
reset, and propagates the adjoint through the event with a saltation update.

The executable exports an analytic reference trajectory at 5 ms intervals and
retains both one-sided states at impact. Position is continuous, while velocity
changes from `-2.962000` to `1.184800`. For terminal adjoint `p+ = (0,1)`, the
reverse event produces

```text
saltation alpha       = 4.636732
p-                    = (4.636732, -0.4)
dL / d restitution    = 2.962000
```

Run it with:

```bash
lake exe RunUrdfContactExample --csv /tmp/contact.csv
```

This example is useful because the discrete event is not removed from the
model. The forward state reset and its reverse contribution remain explicit in
the trace.

## Checkpoint schemas during elaboration

The SafeTensors type provider reads checkpoint metadata while compiling a
module:

```lean
safetensors_type_provider "indexed_dir" as Weights
```

It generates tensor specifications, checked per-tensor loaders, and a nested
record corresponding to the checkpoint hierarchy. In the launch fixture this
includes a declaration of the form

```lean
Weights.load_embed_weightTyped :
  IO (Tensor { shape := #[2, 2], dtype := DType.Float32 })
```

The focused example is:

```bash
./scripts/launch/run_safetensors_schema.sh
```

## Model execution

Tyr contains model, tokenizer, sharded-checkpoint, and generation code for
current transformer families. This tests whether the typed surface remains
connected to a practical runtime.

One recorded run loaded a cached Qwen3.6 27B checkpoint and generated 32 tokens
through the ordinary `Device.CUDA 0` path. The machine used an NVIDIA GB10
because that was the available device; neither the model implementation nor the
device abstraction is specific to it. The raw output is stored in
`launch/generated/model-inference/qwen36-27b-gb10.txt`. This is an integration
example rather than a model-quality or performance result.

## GPU programs

There are two relevant layers. Tensor programs carry an explicit device and
the runtime represents CPU, CUDA, and MPS targets. The lower-level kernel
compiler currently targets NVIDIA CUDA. It contains a typed tile language,
architecture capability classes, code generation, native dispatch, and parity
tests against numerical references. This path is not tied to the GB10 used for
recorded launch workloads. The local compiler-side suite contains 49 tests
covering constrained matrix multiplication, reductions, online softmax,
barriers, attention structure, Brownian sampling, and Runge-Kutta code
generation.

In the current checkout, `lake exe TestGPUDSL` passes all 49 tests. The local
machine has no NVCC and builds CPU kernel stubs, so this result verifies the
language, capability constraints, and code-generation checks rather than native
CUDA execution.

A separate hardware run generated five CUDA modules and executed ten tests for
attention, normalization, Brownian sampling, and Runge–Kutta operations. All
ten matched their numerical references on the recorded tolerances. The run used
an NVIDIA GB10 with CUDA 13.0; this identifies the measurement rather than the
scope of the library. The raw log and companion manifest are under
`launch/generated/gpu/`.

Hardware execution is a separate boundary. A useful report names the generated
kernel, device, reference implementation, numerical tolerance, and actual
dispatch route. A passing test on one device is evidence for that path and is
not generalized to every accelerator.

## Current direction

Tyr is an attempt to use Lean as part of an ML system rather than only as a
language in which isolated tensor identities are proved. The useful question is
not whether every numerical property can be encoded in a type. It is which
contracts become clearer or cheaper to enforce when the model, compiler,
solver, checkpoint schema, and kernel description share one language.

The repository includes the examples, focused tests, and raw execution
artifacts used here. Experimental application work is kept separate from this
account of the core system.
