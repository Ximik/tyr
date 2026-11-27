# Architecture

`lake exe CheckLayers` checks the import and `@[extern]` rules under Layers and FFI.

## Layers
- `Tyr/Core` ← `Tyr/IO` ← `Tyr/Lib` ← `Tyr/Model`: a module imports only from its own layer or those to its left.
- Every module under `Tyr/` is in one of the four layers.
- `import Tyr` gives every module of `Tyr/Core`, `Tyr/IO` and `Tyr/Lib` outside `Tyr/Core/FFI/`, and no `Tyr/Model` module. A model is imported by its own module.
- `lake build Tyr` builds `Tyr.lean` and every `Tyr/Model` module (`globs` in `lakefile.lean`).

## FFI
- `@[extern]` only in `Tyr/Core/FFI/`.
- `Tyr/Core/FFI/` imports nothing else from `Tyr`.
- TO BE DONE: `!` operations only in `Tyr/Core/FFI/`.

## Shape
- Shape arithmetic only in `Tyr/Core/Shape/`.
- `Shape.foo` gives the result shape; its admission is a decidable `Shape.FooCompatible : … → Prop`.
