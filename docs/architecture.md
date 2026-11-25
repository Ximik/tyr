# Architecture

## FFI
- `@[extern]` only in `Tyr/FFI/`.
- `Tyr/FFI/` imports nothing else from `Tyr`.
- TO BE DONE: `!` operations only in `Tyr/FFI/`.

## Shape
- Shape arithmetic only in `Tyr/Shape/`.
- `Shape.foo` gives the result shape; its admission is a decidable `Shape.FooCompatible : … → Prop`.
