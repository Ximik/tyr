namespace torch.Data.Pretraining

/-! ## Parquet FFI Declarations

FFI declarations for parquet file operations. These are implemented in
`cc/src/tyr_parquet.cpp` using Apache Arrow.
-/

/-- Parquet file metadata -/
structure ParquetMetadata where
  /-- Number of row groups in the file -/
  numRowGroups : Nat
  /-- Total number of rows -/
  numRows : Nat
  /-- Column names -/
  columns : Array String
  deriving Repr, Inhabited

/-- Row group data -/
structure RowGroupData where
  /-- Documents in this row group (text column) -/
  documents : Array String
  /-- Number of documents -/
  numDocs : Nat
  deriving Repr

/-- List all parquet files in a directory.
    Returns sorted list of file paths. -/
@[extern "lean_parquet_list_files"]
opaque listParquetFiles (dirPath : @& String) : IO (Array String)

/-- Get metadata for a parquet file -/
@[extern "lean_parquet_get_metadata"]
opaque getParquetMetadata (filePath : @& String) : IO ParquetMetadata

/-- Read a specific row group from a parquet file.
    Returns the text column as an array of strings. -/
@[extern "lean_parquet_read_row_group"]
opaque readRowGroup (filePath : @& String) (rowGroupIdx : @& Nat) (textColumn : @& String := "text")
    : IO RowGroupData

/-- Check if a file exists -/
@[extern "lean_parquet_file_exists"]
opaque parquetFileExists (filePath : @& String) : IO Bool

/-- Read all rows from a parquet file as JSON strings.
    Returns an array where each element is a JSON object string for one row.
    This reads all columns and handles various Arrow types (string, int, float, bool, list, struct). -/
@[extern "lean_parquet_read_as_json"]
opaque readParquetAsJson (filePath : @& String) : IO (Array String)

/-- Read a specific row group from a parquet file as JSON strings.
    More memory-efficient for large files. -/
@[extern "lean_parquet_read_row_group_as_json"]
opaque readRowGroupAsJson (filePath : @& String) (rowGroupIdx : @& Nat) : IO (Array String)

end torch.Data.Pretraining
