# Architecture

## Layers
- `Tyr/Core` ← `Tyr/IO` ← `Tyr/Lib` ← `Tyr/Model`: a module imports only from its own layer or those to its left (`lake exe CheckLayers`).

## FFI
- `@[extern]` only in `Tyr/Core/FFI/` (`lake exe CheckLayers`).
- `Tyr/Core/FFI/` imports nothing else from `Tyr` (`lake exe CheckLayers`).
- TO BE DONE: `!` operations only in `Tyr/Core/FFI/`.

## Shape
- Shape arithmetic only in `Tyr/Core/Shape/`.
- `Shape.foo` gives the result shape; its admission is a decidable `Shape.FooCompatible : … → Prop`.
