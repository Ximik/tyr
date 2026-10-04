#!/usr/bin/env python3
"""Exercise the real native Make rules with tiny sources in a temporary checkout.

No LibTorch, Lean, CUDA compilation, or workspace build artifacts are needed.
"""

from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


REPO = Path(__file__).resolve().parents[1]


class NativeBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tyr-native-build-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cc = self.root / "cc"
        for directory in ("cc/src", "cc/include/nested", "cc/tools",
                          "external/git/soxr", "external/wheels/torch/lib",
                          "external/wheels/torch/include/torch/csrc/api/include/torch",
                          "Tyr/GPU/Kernels", ".lake/build/ir/Tyr/GPU/Kernels", "lean/include"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / "cc/Makefile", self.cc / "Makefile")
        for script in ("write_build_config.py", "generate_gpu_kernel_stubs.py"):
            shutil.copy(REPO / "cc/tools" / script, self.cc / "tools" / script)
        (self.root / "external/git/soxr/CMakeLists.txt").touch()
        (self.root / "external/wheels/torch/include/torch/csrc/api/include/torch/torch.h").touch()
        self.header = self.cc / "include/nested/config.h"
        self.header.write_text("#define VALUE 3\n")
        (self.cc / "src/probe.cpp").write_text(
            '#include "nested/config.h"\nextern "C" int probe() { return VALUE; }\n')
        self.obj = self.cc / "build/probe.o"
        self.archive = self.cc / "build/libTyrC.a"

    def make(self, target="build/libTyrC.a", jobs=1, **variables):
        defaults = {
            "LEAN_HOME": str(self.root / "lean"), "NVCC": "/missing/tyr-test-nvcc",
            "OBJ_FILES": "build/probe.o", "SRCS": "probe.cpp", "CU_SRCS": "",
            "MM_SRCS": "", "SOXR_SRCS": "", "GPU": "H100",
            "DEP_FILES": "build/probe.d", "PYTHON": "/missing/tyr-test-python",
        }
        defaults.update(variables)
        result = subprocess.run(["make", "--no-print-directory", f"-j{jobs}", target] +
                                [f"{key}={value}" for key, value in defaults.items()],
                                cwd=self.cc, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def probe(self, library=None):
        (self.root / "main.cpp").write_text(
            '#include <cstdio>\nextern "C" int probe();\nint main() { std::printf("%d", probe()); }\n')
        subprocess.run(["c++", str(self.root / "main.cpp"), str(library or self.archive),
                        "-o", str(self.root / "probe")], check=True, capture_output=True)
        return subprocess.check_output([str(self.root / "probe")], cwd=self.cc, text=True)

    def test_shared_library_tracks_its_dependency_archive(self):
        (self.cc / "src/probe.cpp").write_text(
            'extern "C" int fixture_dependency();\n'
            'extern "C" int probe() { return fixture_dependency(); }\n')
        dependency_source = self.cc / "src/dependency.cpp"
        dependency_source.write_text('extern "C" int fixture_dependency() { return 3; }\n')
        dependency_archive = self.cc / "build/dependency/libfixture.a"
        extension = "dylib" if sys.platform == "darwin" else "so"
        shared = self.cc / f"build/libTyrC.{extension}"
        settings = {
            "SOXR_OBJS": "build/dependency.o",
            "SOXR_LIB": "build/dependency/libfixture.a",
            "SOXR_LIBPATH": "build/dependency",
            "DYLIB_LINK_FLAGS": str(dependency_archive),
            "DEP_FILES": "build/probe.d build/dependency.d",
        }

        # Request the shared file directly: its own prerequisites must build
        # the archive before linking, even under parallel Make.
        self.make(str(shared.relative_to(self.cc)), jobs=4, **settings)
        self.assertEqual(self.probe(shared), "3")
        self.make("all", jobs=4, **settings)
        artifacts = [self.obj, self.archive, dependency_archive, shared]
        original = [path.stat().st_mtime_ns for path in artifacts]
        self.make("all", jobs=4, **settings)
        self.assertEqual([path.stat().st_mtime_ns for path in artifacts], original,
                         "unchanged objects/archives/shared library must not rebuild")

        time.sleep(1.05)  # Whole-second mtime precision in Apple's Make 3.81.
        dependency_source.write_text('extern "C" int fixture_dependency() { return 17; }\n')
        self.make("all", jobs=4, **settings)
        changed = [path.stat().st_mtime_ns for path in artifacts]
        self.assertEqual(changed[:2], original[:2], "unrelated primary objects/archive rebuilt")
        self.assertNotEqual(changed[2], original[2], "dependency archive was not rebuilt")
        self.assertNotEqual(changed[3], original[3], "shared library did not relink")
        self.assertEqual(self.probe(shared), "17", "shared library retained old dependency code")
        self.make("all", jobs=4, **settings)
        self.assertEqual([path.stat().st_mtime_ns for path in artifacts], changed)

    def test_header_and_effective_configuration_invalidation(self):
        self.make()
        self.assertEqual(self.probe(), "3")
        first = self.obj.stat().st_mtime_ns
        self.make()
        self.assertEqual(self.obj.stat().st_mtime_ns, first, "unchanged build recompiled")
        self.assertIn(str(self.header), (self.cc / "build/native-dependencies.txt").read_text())

        # Apple's bundled GNU Make 3.81 compares mtimes at whole-second
        # precision. Separate edits from the preceding successful compile.
        time.sleep(1.05)
        self.header.write_text("#define VALUE 11\n")
        self.make()
        self.assertNotEqual(self.obj.stat().st_mtime_ns, first)
        self.assertEqual(self.probe(), "11")
        second = self.obj.stat().st_mtime_ns
        time.sleep(1.05)
        self.make(GPU="GB10")
        self.assertNotEqual(self.obj.stat().st_mtime_ns, second)
        config = json.loads((self.cc / "build/native-build.json").read_text())
        self.assertEqual(config["GPU_CODE"], "sm_121")
        third = self.obj.stat().st_mtime_ns
        self.make(GPU="GB10")
        self.assertEqual(self.obj.stat().st_mtime_ns, third)

        # Dependency stubs allow a removed/renamed header when its consumer is
        # updated too; an old .d file must not break the incremental build.
        time.sleep(1.05)
        self.header.rename(self.header.with_name("renamed.h"))
        (self.cc / "src/probe.cpp").write_text(
            '#include "nested/renamed.h"\nextern "C" int probe() { return VALUE; }\n')
        self.make(GPU="GB10")
        incremental = self.probe()
        shutil.rmtree(self.cc / "build")
        self.make(GPU="GB10")
        self.assertEqual(self.probe(), incremental)

    def test_stub_add_remove_and_noop_with_stale_ir(self):
        source = self.root / "Tyr/GPU/Kernels/Fixture.lean"
        source.write_text("namespace Tyr.GPU.Kernels\nnamespace Nested\nend Nested\n"
                          "@[gpu_kernel .SM90]\ndef first := 0\nend Tyr.GPU.Kernels\n")
        self.make("gpu-stubs")
        output = self.cc / "src/generated/tyr_gpu_kernel_stubs.cpp"
        self.assertIn("lean_launch_Tyr_GPU_Kernels_first", output.read_text())
        self.assertNotIn("Nested_first", output.read_text())
        first = output.stat().st_mtime_ns
        self.make("gpu-stubs")
        self.assertEqual(output.stat().st_mtime_ns, first)
        ir = self.root / ".lake/build/ir/Tyr/GPU/Kernels/Fixture.c.o.export"
        ir.write_bytes(b"lean_launch_Tyr_GPU_Kernels_first\x00")
        source.write_text("/- @[gpu_kernel .SM90]\ndef fake := 0 -/\n"
                          "namespace Tyr.GPU.Kernels\n@[gpu_kernel .SM90] def second := 0\n"
                          "end Tyr.GPU.Kernels\n")
        self.make("gpu-stubs")
        text = output.read_text()
        self.assertIn("lean_launch_Tyr_GPU_Kernels_second", text)
        self.assertNotIn("lean_launch_Tyr_GPU_Kernels_first", text)
        self.assertNotIn("lean_launch_fake", text)
        source.unlink()
        self.make("gpu-stubs")
        self.assertNotIn("lean_launch_Tyr_GPU_Kernels_", output.read_text())

    def test_configuration_values_are_shell_quoted(self):
        value = "-DNAME='quoted value' -DOTHER=literal"
        self.make("native-config", EXTRA_CXX_FLAGS=value)
        config = json.loads((self.cc / "build/native-build.json").read_text())
        self.assertEqual(config["EXTRA_CXX_FLAGS"], value)


if __name__ == "__main__":
    unittest.main()
