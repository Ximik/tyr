import LeanTest
import Tyr.EventSkeleton

namespace Tests.EventSkeletonSaltation

open LeanTest
open Tyr.EventSkeleton

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertArrayApprox
    (actual expected : Array Float)
    (tol : Float)
    (label : String) : IO Unit := do
  let diff := FloatArray.maxAbsDiff actual expected
  LeanTest.assertTrue (diff < tol)
    s!"{label}: max abs diff {diff}, actual={actual}, expected={expected}"

@[test]
def testSaltationReverseMatchesExplicitMatrix : IO Unit := do
  let data : SaltationData := {
    resetJac := #[#[2.0, 0.0], #[0.0, 3.0]]
    guardGrad := #[1.0, -1.0]
    a := #[4.0, -2.0]
    gamma := 2.0
  }
  let pPlus := #[0.5, -1.5]
  match data.reverseState? pPlus, data.saltationTransposeApply? pPlus with
  | .ok rankOne, .ok dense =>
      assertArrayApprox rankOne dense 1.0e-12
        "rank-one reverse update should equal explicit S^T p"
      assertArrayApprox rankOne #[3.5, -7.0] 1.0e-12
        "rank-one reverse update should match the hand-computed value"
  | .error msg, _ => LeanTest.fail s!"reverseState? failed: {msg}"
  | _, .error msg => LeanTest.fail s!"saltationTransposeApply? failed: {msg}"

@[test]
def testSaltationEventCostAndThetaUpdate : IO Unit := do
  let data : SaltationData := {
    resetJac := #[#[2.0, 0.0], #[0.0, 3.0]]
    guardGrad := #[1.0, -1.0]
    a := #[4.0, -2.0]
    gamma := 2.0
    beta := 1.0
    costStateGrad := #[0.25, 0.5]
    resetTheta := #[#[1.0, 0.0], #[0.0, 2.0]]
    guardTheta := #[2.0, -1.0]
    costThetaGrad := #[0.1, 0.2]
  }
  let pPlus := #[0.5, -1.5]
  match data.timingAdjoint? pPlus with
  | .error msg => LeanTest.fail s!"timingAdjoint? failed: {msg}"
  | .ok alpha =>
      LeanTest.assertTrue (approx alpha 2.0 1.0e-12)
        s!"Expected alpha=2.0, got {alpha}"
  match data.reverseState? pPlus with
  | .error msg => LeanTest.fail s!"reverseState? failed: {msg}"
  | .ok pMinus =>
      assertArrayApprox pMinus #[3.25, -6.0] 1.0e-12
        "event cost should shift the reverse state update"
  match data.reverseTheta? pPlus with
  | .error msg => LeanTest.fail s!"reverseTheta? failed: {msg}"
  | .ok theta =>
      assertArrayApprox theta #[4.6, -4.8] 1.0e-12
        "theta update should include reset transpose, guard timing, and event cost"

@[test]
def testSaltationDataFromFieldsComputesGammaAndA : IO Unit := do
  let data :=
    SaltationData.mkFromFields
      #[#[2.0, 0.0], #[0.0, 3.0]]
      #[1.0, -1.0]
      #[1.0, 2.0]
      #[5.0, 7.0]
      (resetTime := #[0.5, 1.0])
      (guardTime := 3.0)
  LeanTest.assertTrue (approx data.gamma 2.0 1.0e-12)
    s!"Expected gamma=2.0, got {data.gamma}"
  assertArrayApprox data.a #[2.5, 0.0] 1.0e-12
    "a should equal fPlus - R_x fMinus - R_t"

@[test]
def testSaltationRejectsZeroGamma : IO Unit := do
  let data : SaltationData := {
    resetJac := #[#[1.0]]
    guardGrad := #[1.0]
    a := #[1.0]
    gamma := 0.0
  }
  match data.reverseState? #[1.0] with
  | .ok _ => LeanTest.fail "Zero gamma should reject non-transverse event elimination"
  | .error msg =>
      LeanTest.assertTrue (msg.contains "not transverse")
        s!"Expected transversality diagnostic, got: {msg}"

private def assertRejected (result : Except String α) (label : String) : IO Unit := do
  match result with
  | .ok _ => LeanTest.fail s!"{label}: expected an input diagnostic"
  | .error _ => pure ()

@[test] def testMaxAbsDiffRejectsInvalidNumericalComparisons : IO Unit := do
  let nan := (0.0 / 0.0 : Float)
  let infinity := (1.0 / 0.0 : Float)
  for invalid in #[nan, infinity, -infinity] do
    LeanTest.assertTrue (!(FloatArray.maxAbsDiff #[invalid] #[0.0] < 1.0e-12))
      "Nonfinite actual values must fail numerical comparisons"
    LeanTest.assertTrue (!(FloatArray.maxAbsDiff #[0.0] #[invalid] < 1.0e-12))
      "Nonfinite expected values must fail numerical comparisons"
    LeanTest.assertTrue (!(FloatArray.maxAbsDiff #[invalid] #[invalid] < 1.0e-12))
      "Matching infinities or NaNs must not pass a finite numerical comparison"
  LeanTest.assertTrue (!(FloatArray.maxAbsDiff #[1.0] #[1.0, 0.0] < 1.0e-12))
    "Missing trailing zeros must not conceal a dimension mismatch"
  LeanTest.assertTrue (!(FloatArray.maxAbsDiff #[1.0, 0.0] #[1.0] < 1.0e-12))
    "Excess trailing zeros must not conceal a dimension mismatch"
  assertArrayApprox #[] #[] 1.0e-12 "Empty finite vectors agree"
  LeanTest.assertTrue (approx (FloatArray.maxAbsDiff #[1.0, -3.0] #[1.25, -2.0]) 1.0 1e-12)
    "Finite equal-sized vectors retain the maximum absolute difference"

private def scalarData : SaltationData := {
  resetJac := #[#[1.0]], guardGrad := #[1.0], a := #[1.0], gamma := 1.0
}

@[test] def testSaltationRejectsDimensionMismatches : IO Unit := do
  let malformed := #[
    { scalarData with resetJac := #[#[1.0, 0.0]] },
    { scalarData with resetJac := #[#[1.0], #[]], a := #[1.0, 0.0] },
    { scalarData with guardGrad := #[1.0, 0.0] },
    { scalarData with a := #[1.0, 0.0] },
    { scalarData with costStateGrad := #[1.0, 0.0] },
    { scalarData with resetTheta := #[#[1.0], #[2.0]] },
    { scalarData with resetTheta := #[#[1.0]], guardTheta := #[1.0, 0.0] },
    { scalarData with guardTheta := #[1.0], costThetaGrad := #[1.0, 0.0] }
  ]
  for data in malformed do
    assertRejected (data.reverseState? #[1.0]) "Malformed saltation reverse state"
    assertRejected data.saltationMatrix? "Malformed saltation matrix"
  for pPlus in #[#[], #[1.0, 0.0]] do
    assertRejected (scalarData.timingAdjoint? pPlus) "Wrong timing cotangent dimension"
    assertRejected (scalarData.reverseTheta? pPlus) "Wrong theta cotangent dimension"
    assertRejected (scalarData.saltationTransposeApply? pPlus) "Wrong matrix cotangent dimension"

@[test] def testSaltationRejectsNonfiniteInputsAndResults : IO Unit := do
  for invalid in #[(0.0 / 0.0 : Float), 1.0 / 0.0, -1.0 / 0.0] do
    let malformed := #[
      { scalarData with gamma := invalid },
      { scalarData with beta := invalid },
      { scalarData with resetJac := #[#[invalid]] },
      { scalarData with guardGrad := #[invalid] },
      { scalarData with a := #[invalid] },
      { scalarData with costStateGrad := #[invalid] },
      { scalarData with resetTheta := #[#[invalid]] },
      { scalarData with guardTheta := #[invalid] },
      { scalarData with costThetaGrad := #[invalid] }
    ]
    for data in malformed do
      assertRejected (data.reverseState? #[1.0]) "Nonfinite saltation field"
      assertRejected data.saltationMatrix? "Nonfinite saltation matrix input"
    assertRejected (scalarData.reverseState? #[invalid]) "Nonfinite cotangent"
  let large := { scalarData with a := #[1.0e308] }
  assertRejected (large.timingAdjoint? #[1.0e308]) "Overflowing timing contraction"
  let largeReset := { scalarData with resetJac := #[#[1.0e308]], a := #[0.0] }
  assertRejected (largeReset.reverseState? #[2.0]) "Overflowing reset contraction"

@[test] def testSaltationValidatedFieldsAndOptionalZeros : IO Unit := do
  assertRejected
    (SaltationData.mkFromFields? #[#[1.0]] #[1.0] #[1.0, 0.0] #[1.0])
    "Validated fields reject truncated pre-event vector field"
  assertRejected
    (SaltationData.mkFromFields? #[#[1.0]] #[1.0] #[1.0] #[1.0, 0.0])
    "Validated fields reject extra post-event vector field entries"
  assertRejected
    (SaltationData.mkFromFields? #[#[1.0]] #[1.0] #[1.0] #[1.0]
      (resetTime := #[0.0, 0.0]))
    "Validated fields reject mismatched reset time derivative"
  let data : SaltationData := {
    resetJac := #[#[2.0], #[3.0]], guardGrad := #[1.0], a := #[1.0, 0.0], gamma := 1.0,
    costThetaGrad := #[2.0, 3.0]
  }
  match data.reverseState? #[4.0, 5.0], data.reverseTheta? #[4.0, 5.0] with
  | .ok state, .ok theta =>
      assertArrayApprox state #[27.0] 1e-12 "Rectangular state reset"
      assertArrayApprox theta #[2.0, 3.0] 1e-12 "Omitted parameter terms are zeros"
  | _, _ => LeanTest.fail "Valid rectangular resets and optional zeros should be accepted"
  match SaltationData.mkFromFields? #[#[2.0]] #[1.0] #[1.0] #[3.0] with
  | .error msg => LeanTest.fail s!"Valid fields rejected: {msg}"
  | .ok validated =>
      assertArrayApprox validated.a #[1.0] 1e-12 "Validated field flow correction"
      LeanTest.assertTrue (approx validated.gamma 1.0 1e-12) "Validated field gamma"

end Tests.EventSkeletonSaltation
