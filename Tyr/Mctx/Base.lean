import Std

/-!
# Tyr.Mctx.Base

Core data model for Tyr's Monte Carlo tree search (Lean port of mctx):
action/node-index aliases, root/recurrent function I/O, policy outputs, and
search summaries shared by all `Tyr.Mctx` submodules.
-/

namespace torch.mctx

abbrev Action := Nat
abbrev NodeIndex := Nat
abbrev Depth := Nat

/-- Model step output used by recurrent dynamics. -/
structure RecurrentFnOutput where
  reward : Float
  discount : Float
  priorLogits : Array Float
  value : Float
  deriving Repr, Inhabited

/-- Batched model step output used by recurrent dynamics. -/
structure BatchedRecurrentFnOutput where
  reward : Array Float
  discount : Array Float
  priorLogits : Array (Array Float)
  value : Array Float
  deriving Repr, Inhabited

/-- Root representation output. -/
structure RootFnOutput (S : Type) where
  priorLogits : Array Float
  value : Float
  embedding : S
  deriving Repr

/-- Batched root representation output. -/
structure BatchedRootFnOutput (S : Type) where
  priorLogits : Array (Array Float)
  value : Array Float
  embedding : Array S
  deriving Repr

/-- Generic policy output carrying selected action and search tree. -/
structure PolicyOutput (TreeType : Type) where
  action : Action
  actionWeights : Array Float
  searchTree : TreeType
  deriving Repr

/-- Batched policy output carrying selected actions and search tree. -/
structure BatchedPolicyOutput (TreeType : Type) where
  action : Array Action
  actionWeights : Array (Array Float)
  searchTree : TreeType
  deriving Repr

/-- Root summary statistics after search. -/
structure SearchSummary where
  visitCounts : Array Nat
  visitProbs : Array Float
  value : Float
  qvalues : Array Float
  deriving Repr

/-- Batched root summary statistics after search. -/
structure BatchedSearchSummary where
  visitCounts : Array (Array Nat)
  visitProbs : Array (Array Float)
  value : Array Float
  qvalues : Array (Array Float)
  deriving Repr

/-- Recurrent dynamics signature. -/
abbrev RecurrentFn (P S : Type) :=
  P → UInt64 → Action → S → RecurrentFnOutput × S

/-- Batched recurrent dynamics signature. -/
abbrev BatchedRecurrentFn (P S : Type) :=
  P → UInt64 → Array Action → Array S → BatchedRecurrentFnOutput × Array S

end torch.mctx
