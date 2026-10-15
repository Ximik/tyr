#pragma once
#include <lean/lean.h>

// Match the generated Lean declarations, including IO world-argument erasure.
// File paths and column names are borrowed by the native implementations.
#ifdef __cplusplus
extern "C" {
#endif
lean_object* lean_parquet_list_files(lean_object*);
lean_object* lean_parquet_get_metadata(lean_object*);
lean_object* lean_parquet_read_row_group(lean_object*, uint64_t, lean_object*);
lean_object* lean_parquet_file_exists(lean_object*);
lean_object* lean_parquet_read_as_json(lean_object*);
lean_object* lean_parquet_read_row_group_as_json(lean_object*, uint64_t);
#ifdef __cplusplus
}
#endif
