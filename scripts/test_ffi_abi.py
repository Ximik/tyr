#!/usr/bin/env python3
"""CLI regressions for complete and explicitly scoped native ABI audits."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


CHECKER = Path(__file__).with_name("check_ffi_abi.py")


class AbiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tyr-ffi-abi-")
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.header = root / "audited.h"
        self.generated = root / "generated.c"
        self.declaration = "lean_object* lean_parquet_get_metadata(lean_object*);\n"
        self.header.write_text(self.declaration)
        self.generated.write_text(self.declaration)

    def run_check(self, *args):
        return subprocess.run(
            [sys.executable, str(CHECKER), "--header", str(self.header),
             "--generated", str(self.generated), *args],
            text=True, capture_output=True, check=False)

    def test_unrelated_import_ignored_only_by_explicit_scope(self):
        self.generated.write_text(self.declaration +
                                  "lean_object* lean_torch_dist_is_initialized(void);\n")
        result = self.run_check("--require-complete")
        self.assertEqual(result.returncode, 1)
        self.assertIn("lean_torch_dist_is_initialized: generated declaration is not covered", result.stderr)
        result = self.run_check("--require-complete", "--symbol-prefix", "lean_parquet_")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1 audited declarations", result.stdout)

    def test_scoped_completeness_rejects_unlisted_matching_export(self):
        self.generated.write_text(self.declaration +
                                  "lean_object* lean_parquet_new_operation(lean_object*);\n")
        result = self.run_check("--require-complete", "--symbol-prefix", "lean_parquet_")
        self.assertEqual(result.returncode, 1)
        self.assertIn("lean_parquet_new_operation: generated declaration is not covered", result.stderr)

    def test_scoping_does_not_hide_abi_mismatch(self):
        self.generated.write_text("lean_object* lean_parquet_get_metadata(lean_object*, uint64_t);\n")
        for options in [(), ("--symbol-prefix", "lean_parquet_")]:
            with self.subTest(options=options):
                result = self.run_check(*options)
                self.assertEqual(result.returncode, 1)
                self.assertIn("lean_parquet_get_metadata: header", result.stderr)

    def test_header_must_be_nonempty_and_entirely_within_scope(self):
        unrelated = "lean_object* lean_torch_dist_is_initialized(void);\n"
        for header in ["", unrelated, self.declaration + unrelated]:
            with self.subTest(header=header):
                self.header.write_text(header)
                self.generated.write_text(self.declaration + unrelated)
                result = self.run_check("--symbol-prefix", "lean_parquet_")
                self.assertEqual(result.returncode, 1)
                self.assertIn("audited header", result.stderr)

    def test_scoping_does_not_hide_missing_generated_declaration(self):
        self.generated.write_text("lean_object* lean_torch_dist_is_initialized(void);\n")
        result = self.run_check("--symbol-prefix", "lean_parquet_")
        self.assertEqual(result.returncode, 1)
        self.assertIn("lean_parquet_get_metadata: missing generated declaration", result.stderr)

    def test_default_partial_audit_retains_existing_behavior(self):
        self.generated.write_text(self.declaration +
                                  "lean_object* lean_torch_dist_is_initialized(void);\n")
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_symbol_prefix_cannot_be_empty_or_invalid(self):
        for prefix in ["", "lean-parquet", "*"]:
            with self.subTest(prefix=prefix):
                result = self.run_check("--symbol-prefix", prefix)
                self.assertEqual(result.returncode, 2)
                self.assertIn("nonempty C identifier prefix", result.stderr)


if __name__ == "__main__":
    unittest.main()
