import Examples.AlphaGradPort.A0Train

/-!
  Examples/AlphaGradPort/TaskSweep.lean

  Sequential runner that targets Graphax/AlphaGrad benchmark names one-by-one.
  Default order starts from RoeFlux_1d and advances through the remaining tasks.
-/

namespace Examples.AlphaGradPort

private def parseNatArg? (s : String) : Option Nat :=
  s.toNat?

private def usage : String :=
  String.intercalate "\n" <| ([
    "Usage:",
    "  lake exe AlphaGradPortSweep",
    "  lake exe AlphaGradPortSweep <episodes>",
    "  lake exe AlphaGradPortSweep <task-name> [episodes]",
    "  lake exe AlphaGradPortSweep all [episodes]",
    s!"Tasks: {taskNamesCsv}"
  ] : List String)

private def runOne (taskName : TaskName) (episodes : Nat) : IO Bool := do
  match (← materializeTask taskName) with
  | .error msg =>
    IO.eprintln s!"[AlphaGradPort] task={taskName} materialization failed: {msg}"
    pure false
  | .ok task =>
    let cfg : RunConfig := {
      episodes := episodes
      backend := .dagGumbel
      logEvery := some ⟨max 1 (episodes / 4), by omega⟩
    }
    match (← runTask task cfg) with
    | .error msg =>
      IO.eprintln s!"[AlphaGradPort] task={task.name} failed: {msg}"
      pure false
    | .ok _ =>
      pure true

private def runAll (episodes : Nat) : IO Bool := do
  let mut ok := true
  for h : i in [:taskSequence.size] do
    let taskName := taskSequence[i]
    IO.println s!"[AlphaGradPort] ({i + 1}/{taskSequence.size}) targeting task {taskName}"
    let thisOk ← runOne taskName episodes
    ok := ok && thisOk
  pure ok

def main (args : List String) : IO UInt32 := do
  match args with
  | [] =>
    let ok ← runAll 8
    pure (if ok then 0 else 1)
  | [a] =>
    match parseNatArg? a with
    | some episodes =>
      let ok ← runAll episodes
      pure (if ok then 0 else 1)
    | none =>
      if a.trimAscii.toString = "all" then
        let ok ← runAll 8
        pure (if ok then 0 else 1)
      else
        match parseTaskName? a with
        | some taskName =>
          let ok ← runOne taskName 8
          pure (if ok then 0 else 1)
        | none =>
          IO.eprintln s!"Unknown task/episodes argument: {a}"
          IO.eprintln usage
          pure 1
  | a :: b :: _ =>
    if a.trimAscii.toString = "all" then
      match parseNatArg? b with
      | some episodes =>
        let ok ← runAll episodes
        pure (if ok then 0 else 1)
      | none =>
        IO.eprintln s!"Invalid episodes argument: {b}"
        IO.eprintln usage
        pure 1
    else
      match parseTaskName? a with
      | none =>
        IO.eprintln s!"Unknown task: {a}"
        IO.eprintln usage
        pure 1
      | some taskName =>
        match parseNatArg? b with
        | some episodes =>
          let ok ← runOne taskName episodes
          pure (if ok then 0 else 1)
        | none =>
          IO.eprintln s!"Invalid episodes argument: {b}"
          IO.eprintln usage
          pure 1

end Examples.AlphaGradPort

def main : List String → IO UInt32 := Examples.AlphaGradPort.main
