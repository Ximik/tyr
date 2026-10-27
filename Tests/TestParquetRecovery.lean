import LeanTest
import Tyr.Data.Pretraining

open torch.Data.Pretraining

namespace Tests.TestParquetRecovery

private def expectFailure {α : Type} (action : IO α) : IO Unit := do
  let error ← try
    let _ ← action
    pure none
  catch error => pure (some error.toString)
  match error with
  | some message => LeanTest.assertTrue (!message.isEmpty)
  | none => LeanTest.fail "Expected a recoverable Parquet error"

/-- A malformed existing file reaches the native Parquet parser, rather than
only testing the ordinary missing-file status path. -/
@[test]
def testMalformedParquetIsRecoverable : IO Unit := IO.FS.withTempDir fun dir => do
  let path := dir / "malformed.parquet"
  IO.FS.writeFile path "not a parquet file"
  expectFailure (getParquetMetadata path.toString)
  expectFailure (readRowGroup path.toString 0)
  expectFailure (readParquetAsJson path.toString)
  expectFailure (readRowGroupAsJson path.toString 0)

/-- A single oversized filename component makes filesystem queries throw
ENAMETOOLONG without allocating a large file or requiring special permissions. -/
@[test]
def testParquetFilesystemFailuresAreRecoverable : IO Unit := IO.FS.withTempDir fun dir => do
  let path := dir / String.ofList (List.replicate 300 'x')
  expectFailure (listParquetFiles path.toString)
  expectFailure (parquetFileExists path.toString)

/-- Uncompressed Parquet 1.0: one UTF-8 column `text`, one row group, two rows.
Generated with Arrow 25, without its optional schema metadata or statistics.
Keeping the fixture here makes the test independent of a Python installation. -/
private def validParquet : ByteArray := ByteArray.mk #[
  80, 65, 82, 49, 21, 0, 21, 48, 21, 48, 44, 21, 4, 21, 0, 21, 6, 21, 6, 28,
  0, 0, 0, 2, 0, 0, 0, 4, 1, 5, 0, 0, 0, 104, 101, 108, 108, 111, 5, 0,
  0, 0, 119, 111, 114, 108, 100, 21, 2, 25, 44, 53, 0, 24, 6, 115, 99, 104,
  101, 109, 97, 21, 2, 0, 21, 12, 37, 2, 24, 4, 116, 101, 120, 116, 37, 0,
  76, 28, 0, 0, 0, 22, 4, 25, 28, 25, 28, 38, 0, 28, 21, 12, 25, 37, 6, 0,
  25, 24, 4, 116, 101, 120, 116, 21, 0, 22, 4, 22, 86, 22, 86, 38, 8, 73, 28,
  21, 0, 21, 0, 21, 2, 0, 60, 22, 20, 25, 6, 25, 38, 0, 4, 0, 0, 0, 22, 86,
  22, 4, 38, 8, 22, 86, 0, 40, 32, 112, 97, 114, 113, 117, 101, 116, 45, 99,
  112, 112, 45, 97, 114, 114, 111, 119, 32, 118, 101, 114, 115, 105, 111, 110,
  32, 50, 53, 46, 48, 46, 49, 25, 28, 28, 0, 0, 0, 136, 0, 0, 0, 80, 65, 82, 49]

@[test]
def testParquetReadersRemainUsableAfterErrors : IO Unit := IO.FS.withTempDir fun dir => do
  let path := dir / "valid.parquet"
  IO.FS.writeBinFile path validParquet
  LeanTest.assertTrue (← parquetFileExists path.toString)
  let malformed := dir / "malformed.bin"
  IO.FS.writeFile malformed "not a parquet file"
  expectFailure (getParquetMetadata malformed.toString)
  expectFailure (readRowGroup malformed.toString 0)
  expectFailure (readParquetAsJson malformed.toString)
  expectFailure (readRowGroupAsJson malformed.toString 0)
  expectFailure (getParquetMetadata (dir / "missing.parquet").toString)
  expectFailure (readRowGroup path.toString 1)
  expectFailure (readRowGroup path.toString 0 "missing")
  expectFailure (readRowGroupAsJson path.toString 1)
  LeanTest.assertEqual (← listParquetFiles dir.toString) #[path.toString]
  LeanTest.assertTrue (!(← parquetFileExists (dir / "missing.parquet").toString))
  let metadata ← getParquetMetadata path.toString
  LeanTest.assertEqual metadata.numRows 2
  LeanTest.assertEqual metadata.numRowGroups 1
  LeanTest.assertEqual metadata.columns #["text"]
  let rows ← readRowGroup path.toString 0
  LeanTest.assertEqual rows.numDocs 2
  LeanTest.assertEqual rows.documents #["hello", "world"]
  let json := #["{\"text\":\"hello\"}", "{\"text\":\"world\"}"]
  LeanTest.assertEqual (← readParquetAsJson path.toString) json
  LeanTest.assertEqual (← readRowGroupAsJson path.toString 0) json

end Tests.TestParquetRecovery
