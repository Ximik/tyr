#!/usr/bin/env python3
"""Regressions for the cache's binary compatibility boundary."""
import copy
import unittest

from cache_key import keys


class CacheKeyTests(unittest.TestCase):
    def setUp(self):
        self.identity = {"os": "Linux", "arch": "aarch64", "workspace": "/work/tyr",
                         "compiler": "gcc 12", "lean": "4.29.0",
                         "files": {"lake-manifest.json": "a", "torch/version.h": "2.10"},
                         "environment": {"GPU_CODE": "sm_121", "CXXFLAGS": "-O2"}}

    def test_source_change_reuses_only_compatible_incremental_base(self):
        old = keys(self.identity, "source-a")
        new = keys(self.identity, "source-b")
        self.assertNotEqual(old["build"], new["build"])
        self.assertEqual(old["build_prefix"], new["build_prefix"])
        self.assertEqual(old["dependencies"], new["dependencies"])

    def test_abi_toolchain_dependency_and_location_changes_cannot_restore_old_build(self):
        old = keys(self.identity, "same-source")
        for field in self.identity:
            changed = copy.deepcopy(self.identity)
            changed[field] = "changed"
            with self.subTest(field=field):
                new = keys(changed, "same-source")
                self.assertNotEqual(old["build_prefix"], new["build_prefix"])
                self.assertNotEqual(old["dependencies"], new["dependencies"])

    def test_gpu_architecture_and_compiler_flags_invalidate(self):
        for key in ("GPU_CODE", "CXXFLAGS"):
            changed = copy.deepcopy(self.identity)
            changed["environment"][key] = "different"
            self.assertNotEqual(keys(self.identity, "x")["build_prefix"],
                                keys(changed, "x")["build_prefix"])


if __name__ == "__main__":
    unittest.main()
