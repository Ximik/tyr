import Examples.IOLift.Demo

open torch Examples.IOLift

/-- Runs `act`, reporting how many native matmuls it issued and the first few values. -/
def report (name : String) (act : IO (T s)) : IO Unit := do
  mmCalls.set 0
  let out ← act
  let vals ← out.toFloatArray
  IO.println s!"{name}: {← mmCalls.get} matmul(s), first values {vals.toList.take 3}"

def main : IO Unit := do
  let x : T #[2, 4] := Tensor.full #[2, 4] 0.5
  let w : T #[4, 4] := Tensor.full #[4, 4] 0.25
  let w₁ : T #[4, 8] := Tensor.full #[4, 8] 0.1
  let w₂ : T #[8, 4] := Tensor.full #[8, 4] 0.2
  let kt : T #[4, 2] := Tensor.full #[4, 2] 1.0
  report "mlp      " (mlp x w₁ w₂)
  report "attention" (attention x kt x)
  report "gated    " (gated x w)
  report "gatedBad " (gatedBad x w)
  report "squaredBad" (squaredBad x w)
