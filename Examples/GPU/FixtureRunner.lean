/- Shared fixture/runner helpers for GPU kernel end-to-end examples. -/
import Examples.GPU.Parity
import Tyr.Torch

namespace Examples.GPU

open torch

structure FixtureSpec where
  dir : System.FilePath
  names : Array String
  deriving Inhabited

structure CommonArgs where
  regen : Bool := false
  genOnly : Bool := false
  trials : Nat := 1
  seed : UInt64 := 0
  regenEveryTrial : Bool := false
  deriving Inhabited

def fixturePath (spec : FixtureSpec) (name : String) : System.FilePath :=
  spec.dir / s!"{name}.pt"

def fixturesPresent (spec : FixtureSpec) : IO Bool := do
  let mut ok := true
  for name in spec.names do
    ok := ok && (← (fixturePath spec name).pathExists)
  pure ok

private def parseNatArg (args : List String) (flag : String) (default : Nat) : Nat := Id.run do
  let rec loop (acc : Nat) (xs : List String) : Nat :=
    match xs with
    | key :: value :: rest =>
        if key == flag then
          match value.toNat? with
          | some n => loop n rest
          | none => loop acc rest
        else
          loop acc (value :: rest)
    | _ => acc
  loop default args

private def parseUInt64Arg (args : List String) (flag : String) (default : UInt64) : UInt64 := Id.run do
  let rec loop (acc : UInt64) (xs : List String) : UInt64 :=
    match xs with
    | key :: value :: rest =>
        if key == flag then
          match value.toNat? with
          | some n => loop n.toUInt64 rest
          | none => loop acc rest
        else
          loop acc (value :: rest)
    | _ => acc
  loop default args

def parseCommonArgs (args : List String) : CommonArgs := {
  regen := args.contains "--regen"
  genOnly := args.contains "--gen-only"
  trials := max 1 (parseNatArg args "--trials" 1)
  seed := parseUInt64Arg args "--seed" 0
  regenEveryTrial := args.contains "--regen-every-trial"
}

def vendoredThunderKittensAvailable : IO Bool := do
  (⟨"external/git/ThunderKittens/kernels"⟩ : System.FilePath).pathExists

def runWithFixtures
    (args : List String)
    (suite : String)
    (spec : FixtureSpec)
    (generateFixtures : IO Unit)
    (runOnce : IO Bool)
    : IO UInt32 := do
  let cfg := parseCommonArgs args
  let vendoredAvailable ← vendoredThunderKittensAvailable
  IO.println s!"reference_availability pytorch=true vendored_thunderkittens={vendoredAvailable}"

  let mut allOk := true
  let initialFixturesPresent ← fixturesPresent spec
  for trial in [:cfg.trials] do
    let trialSeed := cfg.seed + trial.toUInt64
    IO.println s!"fixture_trial index={trial + 1}/{cfg.trials}"
    seedFixtures suite trialSeed

    let shouldGenerate :=
      cfg.regen ||
      (!initialFixturesPresent && trial == 0) ||
      (cfg.regenEveryTrial && trial > 0)

    if shouldGenerate then
      generateFixtures

    if !cfg.genOnly then
      seedFixtures suite trialSeed
      let ok ← runOnce
      allOk := allOk && ok

  let vendoredOk ← runVendoredReferenceIfConfigured suite spec.dir
  allOk := allOk && vendoredOk

  pure (if allOk then 0 else 1)

end Examples.GPU
