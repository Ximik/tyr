import Tests.TestDistributedValues

/-- Run two processes with the same port and ranks 0 and 1, using a Gloo-enabled
    native build. Each process checks preservation of its own input values. -/
def main (args : List String) : IO Unit := do
  let [rankText, portText] := args
    | throw (IO.userError "Usage: RunDistributedValues <rank: 0|1> <port>")
  let some rank := rankText.toNat?
    | throw (IO.userError "Invalid rank")
  let some port := portText.toNat?
    | throw (IO.userError "Invalid port")
  unless rank < 2 do throw (IO.userError "Expected rank 0 or 1")
  Tests.DistributedValues.checkTwoRankSession rank.toUInt64 port.toUInt64
  IO.println s!"Distributed value and lifecycle regressions passed on rank {rank}"
