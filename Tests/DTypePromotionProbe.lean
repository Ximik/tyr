import Tyr.Typed.DType
import Std.Data.HashSet

open torch

/-! Differential validation against the independently enumerated native dtype
inventory. Run through `scripts/test_dtype_promotion.sh` after building the dtype
module. Native errors and Undefined both mean no supported promotion result. -/

private def nativeResult (status name : String) : Option DType :=
  if status == "ok" then some (DType.parse name) else none

private def rankedPromotion (ranks : String) (left right : DType) : Option DType :=
  if ranks == "vs" then DType.promoteScalar? left right
  else if ranks == "sv" then DType.promoteScalar? right left else DType.promote? left right

def main (args : List String) : IO UInt32 := do
  let [path] := args | throw (IO.userError "Usage: DTypePromotionProbe NATIVE_REPORT.tsv")
  let contents ← IO.FS.readFile path
  let mut inventory : Array DType := #[]
  let mut ordinals : Array Nat := #[]
  let mut promotionCases := 0
  let mut resultCases := 0
  let mut failures := 0
  let mut skips := 0
  let mut constructed := 0
  let mut operationCases := 0
  let mut operationSuccesses := 0
  let mut operationRejections := 0
  let mut unavailableKernels := 0
  let mut operationSummary : Option Nat := none
  let mut summary : Option (Nat × Nat × Nat × Nat) := none
  let mut caseKeys : Std.HashSet String := {}
  for line in contents.splitOn "\n" do
    let mut failure : Option String := none
    match line.splitOn "\t" with
    | ["dtype", ordinal, name] =>
      let some index := ordinal.toNat? | throw (IO.userError s!"Invalid native dtype ordinal: {ordinal}")
      let dtype := DType.parse name
      if ordinals.contains index || inventory.contains dtype then
        throw (IO.userError s!"Duplicate native dtype or parser collision: {ordinal}/{name}")
      ordinals := ordinals.push index
      inventory := inventory.push dtype
      if (DType.ofString? name).isNone then
        failure := some s!"native dtype {ordinal}/{name} is missing from Lean's parser"
      else if !(DType.all.contains dtype) then
        failure := some s!"native dtype {ordinal}/{name} is missing from DType.all"
      else if DType.ofString? dtype.canonicalName != some dtype then
        failure := some s!"canonical dtype spelling does not roundtrip: {name}/{dtype.canonicalName}"
    | ["promote", left, right, status, result] =>
      let key := s!"promote/{left}/{right}"
      if caseKeys.contains key then throw (IO.userError s!"Duplicate case: {key}")
      caseKeys := caseKeys.insert key
      if !(#["ok", "error", "undefined"].contains status) then
        throw (IO.userError s!"Invalid native status: {line}")
      promotionCases := promotionCases + 1
      let actual := DType.promote? (DType.parse left) (DType.parse right)
      let expected := nativeResult status result
      if actual != expected then
        failure := some s!"promote {left}/{right}: Lean={repr actual}, native={status}/{result}"
    | ["result", ranks, left, right, status, result] =>
      let key := s!"result/{ranks}/{left}/{right}"
      if caseKeys.contains key then throw (IO.userError s!"Duplicate case: {key}")
      caseKeys := caseKeys.insert key
      if !(#["vv", "vs", "sv", "ss"].contains ranks) ||
          !(#["ok", "error", "undefined"].contains status) then
        throw (IO.userError s!"Invalid native result_type case: {line}")
      resultCases := resultCases + 1
      let lhs := DType.parse left
      let rhs := DType.parse right
      let actual := rankedPromotion ranks lhs rhs
      let expected := nativeResult status result
      if actual != expected then
        failure := some s!"result_type {ranks} {left}/{right}: Lean={repr actual}, native={status}/{result}"
    | ["operation", operation, ranks, left, right, status, result] =>
      let key := s!"operation/{operation}/{ranks}/{left}/{right}"
      if caseKeys.contains key then throw (IO.userError s!"Duplicate case: {key}")
      caseKeys := caseKeys.insert key
      if !(#["add", "sub", "mul", "div"].contains operation) ||
          !(#["vv", "vs", "sv", "ss"].contains ranks) then
        throw (IO.userError s!"Invalid native arithmetic case: {line}")
      operationCases := operationCases + 1
      let lhs := DType.parse left
      let rhs := DType.parse right
      let boolSubtraction := operation == "sub" && (lhs == .Bool || rhs == .Bool)
      let expectedBoolError :=
        if lhs == .Bool && rhs == .Bool then
          "Subtraction, the `-` operator, with two bool tensors is not supported. Use the `^` or `logical_xor()` operator instead."
        else
          "Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead."
      let predicted := (rankedPromotion ranks lhs rhs).map fun dtype =>
        if operation == "div" then dtype.atLeastFloat else dtype
      if status == "ok" then
        operationSuccesses := operationSuccesses + 1
        if boolSubtraction || predicted != some (DType.parse result) then
          failure := some s!"{key}: Lean={repr predicted}, native={result}"
      else if status == "rejected" && boolSubtraction && result == expectedBoolError then
        operationRejections := operationRejections + 1
      else if status == "unavailable" && operation == "div" &&
          predicted == some .Complex32 &&
          result == "\"div_cpu\" not implemented for 'ComplexHalf'" then
        unavailableKernels := unavailableKernels + 1
        if unavailableKernels ≤ 5 then IO.println s!"UNAVAILABLE {key}: {result}"
      else
        failure := some s!"Unexpected arithmetic failure {key}: {status}/{result}"
    | ["operations", count] =>
      if operationSummary.isSome then throw (IO.userError "Duplicate arithmetic summary")
      operationSummary := count.toNat?
    | ["tensor", name, "skip", reason] =>
      skips := skips + 1
      failure := some s!"cannot construct result_type operands {name}: {reason}"
    | ["tensor", _, "ok", _] => constructed := constructed + 1
    | ["summary", count, pairs, ranks, unavailable] =>
      if summary.isSome then throw (IO.userError "Duplicate native summary")
      match count.toNat?, pairs.toNat?, ranks.toNat?, unavailable.toNat? with
      | some n, some p, some r, some u => summary := some (n, p, r, u)
      | _, _, _, _ => throw (IO.userError s!"Invalid native summary: {line}")
    | ["version", version] => IO.println s!"Pinned libtorch {version}"
    | [""] => pure ()
    | _ => throw (IO.userError s!"Malformed native promotion record: {line}")
    if let some message := failure then
      if failures < 20 then IO.eprintln s!"FAIL {message}"
      failures := failures + 1
  if inventory.isEmpty then throw (IO.userError "Empty native promotion inventory")
  for dtype in DType.all do
    if !(inventory.contains dtype) then
      IO.eprintln s!"FAIL DType.all entry absent from the native enum: {dtype}"
      failures := failures + 1
  if inventory.size != DType.all.size then
    IO.eprintln s!"FAIL native inventory has {inventory.size} dtypes; DType.all has {DType.all.size}"
    failures := failures + 1
  if promotionCases != inventory.size * inventory.size ||
      resultCases != 4 * constructed * constructed || constructed + skips != inventory.size ||
      summary != some (inventory.size, promotionCases, resultCases, skips) then
    throw (IO.userError "Incomplete native differential matrix or inconsistent summary")
  if operationCases != 13 * 13 * 4 * 4 || operationSummary != some operationCases then
    throw (IO.userError "Incomplete native arithmetic matrix")
  IO.println s!"Dtype promotion: {inventory.size} native dtypes, {promotionCases} promoteTypes and {resultCases} result_type cases, {skips} construction skips, {failures} failures"
  IO.println s!"CPU arithmetic: {operationCases} cases, {operationSuccesses} successful dtype checks, {operationRejections} Bool subtraction rejections, {unavailableKernels} unavailable kernels"
  pure (if failures == 0 then 0 else 1)
