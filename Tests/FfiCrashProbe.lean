/-
Raise a libtorch error with no FFI guard to exercise Tyr's fatal exception
report. This probe must abort on CPU and does not need an accelerator.
-/
import Tyr.RawTensor.Runtime

open torch

def main : IO Unit := do
  IO.println "deliberately raising an uncaught libtorch error..."
  (← IO.getStdout).flush
  RawTensor.throwUncaughtForTest
  IO.println "unexpected return from the native error hook"
