import LeanTest
import Tyr.Typed.Neural

namespace Tests.ComplexResults

open torch

private def diagonal (dtype : DType) : Tensor ⟨#[2, 2], dtype, .CPU⟩ :=
  ((Tensor.fromFloatArray #[3, 0, 0, 4]).reshape #[2, 2]).castDType dtype

private def expectError {α : Type} (result : Except String α) : IO Unit :=
  match result with
  | .ok _ => LeanTest.fail "Expected a recoverable domain or native error"
  | .error _ => pure ()

example (x : Tensor ⟨#[2, 2], .Complex64, .CPU⟩) :
    Tensor ⟨#[2], .Float32, .CPU⟩ := Tensor.rowNorms x

example (x : Tensor ⟨#[2, 2], .Complex32, .CPU⟩) :
    Tensor ⟨#[2], .Float16, .CPU⟩ := Tensor.rowNorms x

example (x : Tensor ⟨#[2, 2], .Complex128, .CPU⟩) :
    Tensor ⟨#[2], .Float64, .CPU⟩ := Tensor.svdvals x

example (x : Tensor ⟨#[2, 2], .Complex64, .CPU⟩) :
    Tensor ⟨#[2, 2], .Complex64, .CPU⟩ × Tensor ⟨#[2], .Float32, .CPU⟩ ×
      Tensor ⟨#[2, 2], .Complex64, .CPU⟩ := Tensor.svdReduced x

example {d : DType} [DType.RealFloating d] (x : Tensor ⟨#[2, 2], d, .CPU⟩) :
    Tensor ⟨#[2, 2], d, .CPU⟩ := Tensor.matrixLog x

example (_x : Tensor ⟨#[2, 2], .Complex64, .CPU⟩) : True := by
  fail_if_success have _ := Tensor.matrixLog _x
  trivial

example (_x : Tensor ⟨#[2, 2], .Float8E4M3FN, .CPU⟩) : True := by
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

@[test] def testMatrixLogDefectiveInputCannotReturnIncorrectResult : IO Unit := do
  -- I + N has the exact real principal log N because N² = 0. Eigenvector
  -- reconstruction is unreliable for this defective (nondiagonalizable) input.
  -- A checked solver may reject it, but must never report an inaccurate success.
  let jordan := ((Tensor.fromFloatArray #[1, 1, 0, 1]).reshape #[2, 2]).castDType .Float64
  let identity := ((Tensor.fromFloatArray #[1, 0, 0, 1]).reshape #[2, 2]).castDType .Float64
  let nilpotent := ((Tensor.fromFloatArray #[0, 1, 0, 0]).reshape #[2, 2]).castDType .Float64
  for scale in (#[1.0, 1e-20, 1e20] : Array Float) do
    let input := jordan * scale
    match Tensor.matrixLog? input with
    | .error _ => pure ()
    | .ok logarithm =>
      let logScale := Float.log scale
      let expected := identity * logScale + nilpotent
      LeanTest.assertTrue (Tensor.frobeniusNorm (logarithm - expected) < 1e-10)
        "matrixLog? returned an inaccurate logarithm for a real Jordan block"
      LeanTest.assertTrue (Tensor.frobeniusNorm ((Tensor.matrixExp logarithm - input) / scale) < 1e-10)

@[test] def testMatrixLogGeneralRealRoundtrips : IO Unit := do
  -- Preserve general real inputs: nonsymmetric diagonalizable matrices, real
  -- rotations, and the nonsymmetric inv(A) B intermediate used by SPD.distance.
  for selected in #[RealFloatingDType.ofDType .Float32, RealFloatingDType.ofDType .Float64] do
    let dtype := selected.val
    for data in #[#[2.0, 1.0, 0.0, 3.0], #[0.0, -1.0, 1.0, 0.0],
        #[1.0, 0.5, 1.0, 2.0]] do
      let input := (← IO.ofExcept (Tensor.fromFloatArrayAs? data #[2, 2])).castDType dtype
      match Tensor.matrixLog? input with
      | .error message => LeanTest.fail s!"Supported real matrix logarithm failed: {message}"
      | .ok logarithm =>
        LeanTest.assertEqual logarithm.raw.dtype dtype
        LeanTest.assertTrue (logarithm.raw.device == input.raw.device)
        let error := Tensor.frobeniusNorm (Tensor.sub (Tensor.matrixExp logarithm) input)
        LeanTest.assertTrue (error < if dtype == .Float64 then 1e-10 else 1e-5)
          "General real matrix logarithm did not reconstruct its input"

@[test] def testDynamicComplexResultsRetainActualRealDtype : IO Unit := do
  let input := (diagonal .Complex128)
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
  | _, _, _ => LeanTest.fail "Valid exact complex factorization was rejected"

@[test] def testMatrixLogRejectsInvalidRuntimeDomain : IO Unit := do
  expectError (Tensor.matrixLog? (diagonal .Complex64))
  expectError (Tensor.matrixLog? (diagonal .Int64))
  let input := (diagonal .Float32)
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
  expectError (Tensor.l2Norm? (integers.reshape #[4]))

end Tests.ComplexResults
