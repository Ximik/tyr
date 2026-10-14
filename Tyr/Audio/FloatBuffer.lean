/-
  Tyr/Audio/FloatBuffer.lean

  Contiguous unboxed Float64 buffer backed by native memory.
  Analogous to Haskell's `Data.Vector.Storable Double` —
  avoids the per-element boxing overhead of `Array Float`.
-/

/-- Contiguous unboxed buffer of Float64 values, backed by native `double*`.
    Mutating operations are exposed through `IO`. -/
opaque FloatBuffer : Type := Unit

namespace FloatBuffer

/-- Create an empty buffer with the given initial capacity (in elements).
    Unaddressable capacities and allocation failures return an IO error. -/
@[extern "lean_float_buffer_mk_empty"]
opaque mkEmpty (capacity : @& Nat) : IO FloatBuffer

/-- Number of elements in the buffer. Pure — O(1). -/
@[extern "lean_float_buffer_size"]
opaque size (buf : @& FloatBuffer) : Nat

/-- Read the element at index `i`. The caller must ensure `i < size`;
    an invalid index is a fatal bounds error. -/
@[extern "lean_float_buffer_get"]
opaque uget (buf : @& FloatBuffer) (i : @& Nat) : Float

/-- Compute RMS (root mean square) of the buffer contents. -/
@[extern "lean_float_buffer_rms"]
opaque rms (buf : @& FloatBuffer) : IO Float

/-- Append a single element. Returns the (mutated) buffer. -/
@[extern "lean_float_buffer_push"]
opaque push (buf : FloatBuffer) (x : Float) : IO FloatBuffer

/-- Append all elements from `src`. Returns the (mutated) `buf`. -/
@[extern "lean_float_buffer_append"]
opaque append (buf : FloatBuffer) (src : @& FloatBuffer) : IO FloatBuffer

/-- Append all elements from an `Array Float` (unboxes each element). -/
@[extern "lean_float_buffer_append_array"]
opaque appendArray (buf : FloatBuffer) (arr : @& Array Float) : IO FloatBuffer

/-- Reset length to 0, keeping allocated capacity. -/
@[extern "lean_float_buffer_clear"]
opaque clear (buf : FloatBuffer) : IO FloatBuffer

/-- Keep only the last `n` samples, dropping earlier ones (memmove).
    If `n ≥ size`, including arbitrarily large `Nat` values, keep all samples. -/
@[extern "lean_float_buffer_keep_last"]
opaque keepLast (buf : FloatBuffer) (n : @& Nat) : IO FloatBuffer

/-- Convert to `Array Float` (boxes each element). Use at pipeline boundaries. -/
@[extern "lean_float_buffer_to_array"]
opaque toArray (buf : @& FloatBuffer) : IO (Array Float)

-- Inhabited is derived automatically from the `opaque ... := Unit` declaration.

end FloatBuffer
