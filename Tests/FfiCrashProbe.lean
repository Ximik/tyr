/-
  Tests/FfiCrashProbe.lean

  Manual probe for the FFI failure mode. Intentionally triggers a libtorch
  device error (transfer to CUDA on a build/machine without CUDA), so it is
  NOT part of the test suite. Build and run the `ffi_crash_probe` executable
  to check that an uncaught C++ exception terminates the process with an
  intelligible message (the `c10::Error` text) instead of a bare
  SIGABRT/SIGSEGV. On a machine where CUDA is available the transfer
  succeeds and the probe reports that nothing crashed.
-/
import Tyr.Typed

open torch

def main : IO Unit := do
  let m : DTensor #[2, 2] .Float32 := Tensor.ones #[2, 2]
  IO.println "triggering libtorch device error via .to (CUDA 0)..."
  (← IO.getStdout).flush
  let c := m.to (.CUDA 0)
  let arr ← c.toFloatArray
  IO.println s!"no crash (CUDA available?): {arr}"
