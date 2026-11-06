import Examples.AlphaGradPort.A0Train

/-!
  Examples/AlphaGradPort/RoeFlux1dA0.lean

  End-to-end AlphaGrad-style planning demo for the first concrete port target:
  Graphax `RoeFlux_1d`.
-/

namespace Examples.AlphaGradPort

private def parseNatArg? (s : String) : Option Nat :=
  s.toNat?

private def usage : String :=
  String.intercalate "\n" <| ([
    "Usage: lake exe AlphaGradRoeFlux1dA0 [episodes]",
    "  episodes: optional natural number (default: 16)"
  ] : List String)

def main (args : List String) : IO UInt32 := do
  let episodes? :=
    match args with
    | [] => some 16
    | a :: _ => parseNatArg? a
  match episodes? with
  | none =>
    let badArg := args.headD ""
    IO.eprintln s!"Invalid episodes argument: {badArg}"
    IO.eprintln usage
    return 1
  | some episodes =>
    let task ←
      match (← materializeTask .roeFlux1d) with
      | .error msg =>
        IO.eprintln s!"Failed to materialize RoeFlux_1d task: {msg}"
        return 1
      | .ok task =>
        pure task
    let cfg : RunConfig := {
      episodes := episodes
      backend := .dagGumbel
      logEvery := some ⟨max 1 (episodes / 4), by omega⟩
    }

    match (← runTask task cfg) with
    | .error msg =>
      IO.eprintln s!"AlphaGrad RoeFlux_1d port failed: {msg}"
      return 1
    | .ok summary =>
      IO.println s!"AlphaGrad RoeFlux_1d summary: {reprStr summary}"
      return 0

end Examples.AlphaGradPort

def main : List String → IO UInt32 := Examples.AlphaGradPort.main
