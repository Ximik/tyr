import LeanTest
import Tyr.Typed.Neural

namespace Tests.ComplexResults

open torch

private def diagonal (dtype : DType) : Tensor ⟨#[2, 2], dtype, .any⟩ :=
  ((Tensor.fromFloatArray #[3, 0, 0, 4]).reshapeChecked #[2, 2]).castDType dtype

private def expectError {α : Type} (result : Except String α) : IO Unit :=
  match result with
  | .ok _ => LeanTest.fail "Expected a recoverable domain or native error"
  | .error _ => pure ()

example (x : Tensor ⟨#[2, 2], .Complex64, .any⟩) :
    Tensor ⟨#[2], .Float32, .any⟩ := Tensor.rowNorms x

example (x : Tensor ⟨#[2, 2], .Complex32, .any⟩) :
    Tensor ⟨#[2], .Float16, .any⟩ := Tensor.rowNorms x

example (x : Tensor ⟨#[2, 2], .Complex128, .any⟩) :
    Tensor ⟨#[2], .Float64, .any⟩ := Tensor.svdvals x

example (x : Tensor ⟨#[2, 2], .Complex64, .any⟩) :
    Tensor ⟨#[2, 2], .Complex64, .any⟩ × Tensor ⟨#[2], .Float32, .any⟩ ×
      Tensor ⟨#[2, 2], .Complex64, .any⟩ := Tensor.svdReduced x

example {d : DType} [DType.RealFloating d] (x : Tensor ⟨#[2, 2], d, .any⟩) :
    Tensor ⟨#[2, 2], d, .any⟩ := Tensor.matrixLog x

example (_x : Tensor ⟨#[2, 2], .Complex64, .any⟩) : True := by
  fail_if_success have _ := Tensor.matrixLog _x
  trivial

example (_x : Tensor ⟨#[2, 2], .Float8E4M3FN, .any⟩) : True := by
  fail_if_success have _ := Tensor.matrixLog _x
  trivial

example (_x : Tensor ⟨#[2, 2], .Unknown "dynamic", .any⟩) : True := by
  fail_if_success have _ := Tensor.matrixLog _x
  trivial

@[test] def testComplexRowNormsAreReal : IO Unit := do
  for (complex, real) in #[(DType.Complex64, DType.Float32), (.Complex128, .Float64)] do
    let input := diagonal complex
    let norms := Tensor.rowNorms input
    LeanTest.assertEqual norms.raw.dtype real
    LeanTest.assertEqual (← norms.toFloatArray) #[3, 4]
    LeanTest.assertTrue norms.validate.toOption.isSome
    LeanTest.assertTrue (norms.raw.device == input.raw.device)
    LeanTest.assertEqual (Tensor.frobeniusNorm input) 5
    LeanTest.assertEqual (Tensor.maxRowNorm input) 4

@[test] def testComplexSingularValuesAreReal : IO Unit := do
  for (complex, real) in #[(DType.Complex64, DType.Float32), (.Complex128, .Float64)] do
    let input := diagonal complex
    let values := Tensor.svdvals input
    LeanTest.assertEqual values.raw.dtype real
    LeanTest.assertEqual (← values.toFloatArray) #[4, 3]
    LeanTest.assertTrue values.validate.toOption.isSome
    LeanTest.assertTrue (values.raw.device == input.raw.device)

@[test] def testComplexSVDKeepsComplexVectorsAndRealValues : IO Unit := do
  for (complex, real) in #[(DType.Complex64, DType.Float32), (.Complex128, .Float64)] do
    let input := diagonal complex
    let (u, values, vh) := Tensor.svdReduced input
    LeanTest.assertEqual values.raw.dtype real
    LeanTest.assertEqual u.raw.dtype complex
    LeanTest.assertEqual vh.raw.dtype complex
    LeanTest.assertEqual (← values.toFloatArray) #[4, 3]
    -- Singular values are converted explicitly only when reconstructing a
    -- complex matrix with the same-dtype matrix-multiplication primitive.
    let diagonalValues := (Tensor.diagflat values).castDType complex
    let reconstructed := Tensor.mm (Tensor.mm u diagonalValues) vh
    LeanTest.assertEqual (← reconstructed.toFloatArray) #[3, 0, 0, 4]

@[test] def testRealMatrixLogRoundtrip : IO Unit := do
  let input := diagonal .Float64
  let logarithm := Tensor.matrixLog input
  let restored := Tensor.matrixExp logarithm
  LeanTest.assertEqual logarithm.raw.dtype .Float64
  LeanTest.assertTrue ((Tensor.frobeniusNorm (restored - input)) < 1e-10)

@[test] def testDynamicComplexResultsRetainActualRealDtype : IO Unit := do
  let input := (diagonal .Complex128).forgetDType
  match Tensor.rowNorms? input, Tensor.svdvals? input, Tensor.svdReduced? input with
  | .ok norms, .ok values, .ok (u, singular, vh) =>
    for tensor in #[norms, values, singular] do
      LeanTest.assertEqual tensor.raw.dtype .Float64
      LeanTest.assertTrue (tensor.raw.device == input.raw.device)
    LeanTest.assertEqual (← norms.toFloatArray) #[3, 4]
    LeanTest.assertEqual (← values.toFloatArray) #[4, 3]
    LeanTest.assertEqual (← singular.toFloatArray) #[4, 3]
    LeanTest.assertEqual u.raw.dtype .Complex128
    LeanTest.assertEqual vh.raw.dtype .Complex128
  | _, _, _ => LeanTest.fail "Valid dynamic complex factorization was rejected"

@[test] def testMatrixLogRejectsInvalidRuntimeDomain : IO Unit := do
  expectError (Tensor.matrixLog? (diagonal .Complex64))
  expectError (Tensor.matrixLog? (diagonal .Complex64).forgetDType)
  expectError (Tensor.matrixLog? (diagonal .Int64).forgetDType)
  let input := (diagonal .Float32).forgetDType
  match Tensor.matrixLog? input with
  | .error message => LeanTest.fail message
  | .ok result =>
    LeanTest.assertEqual result.raw.dtype .Float32
    LeanTest.assertTrue (result.raw.device == input.raw.device)
    LeanTest.assertTrue result.validate.toOption.isSome

@[test] def testFactorizationFailuresAreRecoverable : IO Unit := do
  -- The real logarithm is undefined for these matrices; neither should
  -- return the input as though a logarithm had been computed.
  expectError (Tensor.matrixLog? (Tensor.zeros #[2, 2]))
  expectError (Tensor.matrixLog? (Tensor.full #[1, 1] (-1)))
  -- CPU SVD does not implement ComplexHalf, even though its metadata and
  -- real singular-value dtype are represented by the type system.
  let halfComplex := diagonal .Complex32
  expectError (Tensor.svdvals? halfComplex)
  expectError (Tensor.svdReduced? halfComplex)
  let integers := diagonal .Int64
  expectError (Tensor.rowNorms? integers)
  expectError (Tensor.frobeniusNorm? integers)
  expectError (Tensor.maxRowNorm? integers)
  expectError (Tensor.l2Norm? (integers.reshapeChecked #[4]))

end Tests.ComplexResults
