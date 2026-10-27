import Tests.TestDistributedValues

/-- Run two processes with the same pair of session ports and ranks 0 and 1, using a Gloo-enabled
    native build. Each process checks preservation of its own input values. -/
def main (args : List String) : IO Unit := do
  let [rankText, portText, restartPortText] := args
    | throw (IO.userError "Usage: RunDistributedValues <rank: 0|1> <port> <restart-port>")
  let some rank := rankText.toNat?
    | throw (IO.userError "Invalid rank")
  let some port := portText.toNat?
    | throw (IO.userError "Invalid port")
  let some restartPort := restartPortText.toNat?
    | throw (IO.userError "Invalid restart port")
  unless 0 < port && port < 65536 && 0 < restartPort && restartPort < 65536 do
    throw (IO.userError "Ports must be in [1, 65535]")
  unless port != restartPort do
    throw (IO.userError "Sessions require distinct rendezvous ports")
  unless rank < 2 do throw (IO.userError "Expected rank 0 or 1")
  Tests.DistributedValues.checkTwoRankSession rank.toUInt64 port.toUInt64 restartPort.toUInt64
  IO.println s!"Distributed value and lifecycle regressions passed on rank {rank}"
