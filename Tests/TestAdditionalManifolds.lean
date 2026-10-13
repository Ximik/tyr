import LeanTest
import Tyr.Manifolds.Sphere
import Tyr.Manifolds.Oblique
import Tyr.Manifolds.Positive
import Tyr.Manifolds.PoincareBall
import Tyr.Manifolds.PositiveDefinite
import Tyr.Manifolds.PSD
import Tyr.Manifolds.FixedRank

namespace Tests.AdditionalManifolds

open LeanTest
open torch
open Tyr.AD

@[test]
def testSphereProjectionKeepsUnitNorm : IO Unit := do
  let x0 ← Tensor.randn #[16] false
  let x := Sphere.project 16 x0
  let n := Tensor.l2Norm x.coords
  LeanTest.assertTrue (Float.abs (n - 1.0) < 1e-5)
    s!"Expected sphere norm 1, got {n}"

@[test]
def testObliqueColumnsUnit : IO Unit := do
  let w0 ← Tensor.randn #[12, 5] false
  let w := Oblique.project 12 5 w0
  let wtW := Tensor.mm (Tensor.transpose2d w.matrix) w.matrix
  let d := Tensor.diag wtW
  let onesV := Tensor.ones #[5]
  LeanTest.assertTrue (Tensor.l2Norm (d - onesV) < 1e-4)
    "Expected oblique columns to have unit norm"

@[test]
def testPositiveProjectionPositiveEntries : IO Unit := do
  let a0 ← Tensor.randn #[8, 8] false
  let p := Positive.project 8 8 a0
  let values := p.matrix.raw.getValues
  for i in [:values.size] do
    LeanTest.assertTrue (values[i]! > 0.0)
      s!"Expected all entries > 0, found {values[i]!}"

@[test]
def testPoincareProjectionInsideBall : IO Unit := do
  let x0 ← Tensor.randn #[10] false
  let x := PoincareBall.project 10 (Tensor.mulScalar x0 10.0)
  let n := Tensor.l2Norm x.coords
  LeanTest.assertTrue (n < 1.0)
    s!"Expected poincare point norm < 1, got {n}"

@[test]
def testSPDProjectionSymmetric : IO Unit := do
  let a0 ← Tensor.randn #[6, 6] false
  let x := SymmetricPositiveDefinite.project 6 a0
  let symErr := x.matrix - Tensor.transpose2d x.matrix
  let e := Tensor.frobeniusNorm symErr
  LeanTest.assertTrue (e < 1e-5)
    s!"Expected SPD projection symmetric, error {e}"

@[test]
def testElliptopeRowsUnit : IO Unit := do
  let y0 ← Tensor.randn #[9, 4] false
  let y := Elliptope.project 9 4 y0
  let rn := Tensor.rowNorms y.factor
  let onesV := Tensor.ones #[9]
  LeanTest.assertTrue (Tensor.l2Norm (rn - onesV) < 1e-4)
    "Expected elliptope rows to have unit norm"

@[test]
def testFixedRankProjectionFinite : IO Unit := do
  let a0 ← Tensor.randn #[8, 6] false
  let x := FixedRankEmbedded.project 8 6 3 a0
  let fn := Tensor.frobeniusNorm x.matrix
  LeanTest.assertTrue (Float.isFinite fn && fn > 0.0)
    s!"Expected finite non-zero fixed-rank projection norm, got {fn}"

example (_x : Sphere 3) (_v : DTensor #[4] .Float32) : True := by
  fail_if_success
    have _ := SphereTangent.project _x _v
  trivial

example (_x : Sphere 3) (_v : DTensor #[3] .Float64) : True := by
  fail_if_success
    have _ := SphereTangent.project _x _v
  trivial

example (_x : PoincareBall 3 .Float64 (.exact .CPU)) : True := by
  fail_if_success
    have _wrongDevice : PoincareBall 3 .Float64 (.exact (.CUDA 0)) := _x
  fail_if_success
    have _wrongShape : PoincareBall 4 .Float64 (.exact .CPU) := _x
  trivial

@[test]
def testIndexedSphereTangentAndRetraction : IO Unit := do
  -- The zero-input fallback must retain both dtype and placement.
  let zero : Tensor ⟨#[4], .Float64, .exact .CPU⟩ := Tensor.zerosFor _
  let x := Sphere.project 4 zero
  match x.coords.validate with
  | .error message => LeanTest.fail message
  | .ok () => pure ()
  LeanTest.assertTrue (Float.abs (Tensor.l2Norm x.coords - 1.0) < 1e-12)
    "Zero-vector projection should produce a unit Float64 sphere point"
  let half : Tensor ⟨#[2], .Float64, .exact .CPU⟩ := Tensor.zerosFor _
  let direction := Tensor.cat (Tensor.onesLike half) half 0
  let v := SphereTangent.project x direction
  LeanTest.assertTrue (Tensor.l2Norm v.vec > 0.0)
    "Fixture should exercise a nonzero tangent"
  let dot := Tensor.item (Tensor.sumAll (Mul.mul x.coords v.vec))
  LeanTest.assertTrue (Float.abs dot < 1e-12)
    "Projected sphere tangent must be orthogonal to its point"
  let y : Sphere 4 .Float64 (.exact .CPU) := DifferentiableManifold.retract x v
  LeanTest.assertTrue (Float.abs (Tensor.l2Norm y.coords - 1.0) < 1e-12)
    "Retraction must retain the unit sphere constraint"
  let geodesic : Sphere 4 .Float64 (.exact .CPU) := DifferentiableManifold.exp x v
  LeanTest.assertTrue (Float.abs (Tensor.l2Norm geodesic.coords - 1.0) < 1e-12)
    "Exponential map must retain the unit sphere constraint"
  let tangent : SphereTangent 4 .Float64 (.exact .CPU) :=
    DifferentiableManifold.zeroTangent x
  LeanTest.assertEqual (Tensor.l2Norm tangent.vec) 0.0
    "Zero tangent should retain the point's full specification"

@[test]
def testIndexedPoincareRetraction : IO Unit := do
  let zero : Tensor ⟨#[3], .Float64, .exact .CPU⟩ := Tensor.zerosFor _
  let x := PoincareBall.project 3 (Tensor.onesLike zero)
  let tangent : PoincareBallTangent 3 .Float64 (.exact .CPU) :=
    ⟨Tensor.Preserving.mulScalar x.coords 10.0⟩
  let y : PoincareBall 3 .Float64 (.exact .CPU) :=
    DifferentiableManifold.retract x tangent
  match y.coords.validate with
  | .error message => LeanTest.fail message
  | .ok () => pure ()
  LeanTest.assertTrue (Tensor.l2Norm y.coords < 1.0)
    "Poincare retraction must stay inside the ball"
  LeanTest.assertTrue (PoincareBall.conformalFactor y > 0.0)
    "Conformal factor must be positive inside the ball"

private def assertValid {σ : StaticSpec} (t : Tensor σ) : IO Unit := do
  match t.validate with
  | .ok () => pure ()
  | .error message => LeanTest.fail message

example (_x : Oblique 2 3) (_v : DTensor #[3, 2] .Float32) : True := by
  fail_if_success
    have _ := ObliqueTangent.project _x _v
  trivial

example (_x : SymmetricPositiveDefinite 2 .Float64 (.exact .CPU)) : True := by
  fail_if_success
    have _wrongDtype : SymmetricPositiveDefinite 2 := _x
  fail_if_success
    have _wrongShape : SymmetricPositiveDefinite 3 .Float64 (.exact .CPU) := _x
  trivial

example (_x : PSDFixedRank 2 3) : True := by
  fail_if_success
    have _wrongFactorShape : PSDFixedRank 3 2 := _x
  trivial

@[test]
def testIndexedColumnAndRowTangents : IO Unit := do
  let row : Tensor ⟨#[1, 3], .Float64, .exact .CPU⟩ := Tensor.onesLike (Tensor.zerosFor _)
  let input := Tensor.cat row (Tensor.Preserving.mulScalar row 2.0) 0
  let first : Tensor ⟨#[2, 1], .Float64, .exact .CPU⟩ := Tensor.onesLike (Tensor.zerosFor _)
  let rest : Tensor ⟨#[2, 2], .Float64, .exact .CPU⟩ := Tensor.zerosFor _
  let direction := Tensor.cat first rest 1
  let columns := Oblique.project 2 3 input
  let columnTangent := ObliqueTangent.project columns direction
  assertValid columnTangent.vec
  let columnDots := Tensor.Preserving.sumDim (Mul.mul columns.matrix columnTangent.vec) 0 true
  LeanTest.assertTrue (Tensor.frobeniusNorm columnDots < 1e-10)
    "Oblique tangent must be orthogonal to each column"
  let moved : Oblique 2 3 .Float64 (.exact .CPU) :=
    DifferentiableManifold.retract columns columnTangent
  let norms := Tensor.diag (Tensor.mm (Tensor.transpose2d moved.matrix) moved.matrix)
  LeanTest.assertTrue (Tensor.l2Norm (Sub.sub norms (Tensor.onesLike norms)) < 1e-10)
    "Oblique retraction must restore column unit norms"
  let rows := Elliptope.project 2 3 input
  let rowTangent := ElliptopeTangent.project rows direction
  assertValid rowTangent.vec
  let rowDots := Tensor.Preserving.sumDim (Mul.mul rows.factor rowTangent.vec) 1 true
  LeanTest.assertTrue (Tensor.frobeniusNorm rowDots < 1e-10)
    "Elliptope tangent must be orthogonal to each row"
  let movedRows : Elliptope 2 3 .Float64 (.exact .CPU) :=
    DifferentiableManifold.retract rows rowTangent
  let rowNorms := Tensor.rowNorms movedRows.factor
  LeanTest.assertTrue (Tensor.l2Norm (Sub.sub rowNorms (Tensor.onesLike rowNorms)) < 1e-10)
    "Elliptope retraction must restore row unit norms"

@[test]
def testIndexedPositiveMetricAndRetraction : IO Unit := do
  let ones : Tensor ⟨#[2, 3], .Float64, .exact .CPU⟩ := Tensor.onesLike (Tensor.zerosFor _)
  let x := Positive.project 2 3 (Tensor.Preserving.mulScalar ones 2.0)
  let g : PositiveTangent 2 3 .Float64 (.exact .CPU) := ⟨ones⟩
  let sharp : PositiveTangent 2 3 .Float64 (.exact .CPU) :=
    DifferentiableManifold.sharp (x := x) g
  let flat : PositiveTangent 2 3 .Float64 (.exact .CPU) :=
    DifferentiableManifold.flat (x := x) sharp
  LeanTest.assertTrue (Tensor.frobeniusNorm (Sub.sub flat.vec g.vec) < 1e-10)
    "Positive metric maps should invert each other within regularization error"
  let step : PositiveTangent 2 3 .Float64 (.exact .CPU) :=
    ⟨Tensor.Preserving.mulScalar ones (-1.0)⟩
  let y : Positive 2 3 .Float64 (.exact .CPU) := DifferentiableManifold.retract x step
  assertValid y.matrix
  LeanTest.assertTrue
    (Tensor.frobeniusNorm (Sub.sub y.matrix (Tensor.Preserving.mulScalar ones 1.25)) < 1e-10)
    "Positive retraction should include the quadratic correction"
  let clamped := Positive.project 2 3 step.vec
  assertValid clamped.matrix
  LeanTest.assertTrue
    (Tensor.frobeniusNorm (Sub.sub clamped.matrix (Tensor.Preserving.mulScalar ones 1e-6)) < 1e-10)
    "Negative entries must clamp to the positive floor without changing dtype"

@[test]
def testIndexedSPDMatrixFunctions : IO Unit := do
  let zero : Tensor ⟨#[2, 2], .Float64, .exact .CPU⟩ := Tensor.zerosFor _
  let identity := Tensor.matrixExp zero
  let x := SymmetricPositiveDefinite.project 2
    (Tensor.Preserving.mulScalar (Tensor.onesLike zero) 0.1)
  assertValid x.matrix
  let inverseProduct := Tensor.mm (Tensor.inv x.matrix) x.matrix
  LeanTest.assertTrue (Tensor.frobeniusNorm (Sub.sub inverseProduct identity) < 1e-10)
    "Typed inverse should recover the identity"
  let roundtrip := Tensor.matrixExp (Tensor.matrixLog x.matrix)
  LeanTest.assertTrue (Tensor.frobeniusNorm (Sub.sub roundtrip x.matrix) < 1e-10)
    "Real SPD logarithm/exponential should roundtrip"
  LeanTest.assertTrue (SymmetricPositiveDefinite.distance x x < 1e-10)
    "SPD self-distance must be approximately zero"
  let tangent := SPDTangent.project x (Tensor.Preserving.mulScalar identity 0.1)
  let y : SymmetricPositiveDefinite 2 .Float64 (.exact .CPU) :=
    DifferentiableManifold.retract x tangent
  assertValid y.matrix
  LeanTest.assertTrue
    (Tensor.frobeniusNorm (Sub.sub y.matrix (Tensor.transpose2d y.matrix)) < 1e-10)
    "SPD retraction must stay symmetric"

@[test]
def testIndexedPSDFactorShapeAndUpdate : IO Unit := do
  let ones : Tensor ⟨#[2, 3], .Float64, .exact .CPU⟩ := Tensor.onesLike (Tensor.zerosFor _)
  let x : PSDFixedRank 2 3 .Float64 (.exact .CPU) := ⟨ones⟩
  let ambient : Tensor ⟨#[2, 2], .Float64, .exact .CPU⟩ := x.toPSD
  assertValid ambient
  LeanTest.assertTrue
    (Tensor.frobeniusNorm (Sub.sub ambient (Tensor.Preserving.mulScalar (Tensor.onesLike ambient) 3.0)) < 1e-10)
    "PSD factor must produce the correctly shaped Gram matrix"
  let tangent : PSDFixedRankTangent 2 3 .Float64 (.exact .CPU) :=
    ⟨Tensor.Preserving.mulScalar ones 0.5⟩
  let y : PSDFixedRank 2 3 .Float64 (.exact .CPU) := DifferentiableManifold.retract x tangent
  assertValid y.factor
  LeanTest.assertTrue
    (Tensor.frobeniusNorm (Sub.sub y.toPSD (Tensor.Preserving.mulScalar (Tensor.onesLike ambient) 6.75)) < 1e-10)
    "PSD update must retain its factor dimensions and produce the expected Gram matrix"

example (_a : DTensor #[2, 3] .Float32) : True := by
  fail_if_success
    have _ := FixedRankEmbedded.project 2 3 4 _a
  fail_if_success
    have _ := Tensor.slice _a 2 0 1
  fail_if_success
    have _ := Tensor.slice _a 1 2 2
  fail_if_success
    have _ := Tensor.slice _a 1 18446744073709551615 1
  trivial

@[test]
def testIndexedFixedRankSlicingAndUpdate : IO Unit := do
  let ones : Tensor ⟨#[2, 3], .Float64, .exact .CPU⟩ := Tensor.onesLike (Tensor.zerosFor _)
  let input := Tensor.Preserving.mulScalar ones 2.0
  let x := FixedRankEmbedded.project 2 3 1 input
  assertValid x.matrix
  LeanTest.assertTrue
    (Tensor.frobeniusNorm (Sub.sub x.matrix (Tensor.Preserving.mulScalar ones 4.0)) < 1e-10)
    "Rank-one approximation should use the requested column and row slices"
  let tangent : FixedRankTangent 2 3 1 .Float64 (.exact .CPU) := ⟨ones⟩
  let y : FixedRankEmbedded 2 3 1 .Float64 (.exact .CPU) :=
    DifferentiableManifold.retract x tangent
  assertValid y.matrix
  LeanTest.assertTrue
    (Tensor.frobeniusNorm (Sub.sub y.matrix (Tensor.Preserving.mulScalar ones 25.0)) < 1e-10)
    "Retraction should preserve valid rank bounds and apply the factor-product approximation"
  let empty := FixedRankEmbedded.project 2 3 0 input
  assertValid empty.matrix
  LeanTest.assertEqual (Tensor.frobeniusNorm empty.matrix) 0.0
    "Zero-rank approximation must produce the full-shaped zero matrix"
  let endpoint : Tensor ⟨#[2, 0], .Float64, .exact .CPU⟩ := Tensor.slice input 1 3 0
  assertValid endpoint
  let interior : Tensor ⟨#[2, 2], .Float64, .exact .CPU⟩ := Tensor.slice input 1 1 2
  assertValid interior
  LeanTest.assertTrue
    (Tensor.frobeniusNorm (Sub.sub interior (Tensor.Preserving.mulScalar (Tensor.onesLike interior) 2.0)) < 1e-10)
    "Nonzero-start exact slice should preserve values and metadata"

end Tests.AdditionalManifolds
