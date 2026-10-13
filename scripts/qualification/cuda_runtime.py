#!/usr/bin/env python3
"""Keep Torch's CUDA libraries together and record the libraries actually loaded."""
import argparse
import ctypes
from pathlib import Path
import re

LIBRARIES = ("libcublas.so", "libcublasLt.so", "libcudart.so")


def library_path(libtorch, cuda_home, previous=""):
    torch = Path(libtorch).resolve()
    paths = [torch / "lib"]
    # Python wheels put CUDA dependencies beside torch, while standalone
    # LibTorch archives normally bundle them in lib/ itself.
    if (torch / "__init__.py").is_file():
        paths.extend(sorted((torch.parent / "nvidia").glob("*/lib")))
    paths.append(Path(cuda_home) / "lib64")
    result = []
    candidates = [str(p.resolve()) for p in paths if p.is_dir()]
    candidates.extend(str(Path(p).resolve()) for p in previous.split(":") if p)
    for path in candidates:
        if path and path not in result:
            result.append(path)
    return ":".join(result)


def is_cuda_library(path):
    return any(Path(path).name.startswith(name) for name in LIBRARIES)


def process_libraries():
    maps = Path("/proc/self/maps")
    if not maps.exists():
        return []
    paths = {line.split(maxsplit=5)[5] for line in maps.read_text().splitlines()
             if len(line.split(maxsplit=5)) == 6}
    return sorted(str(Path(path).resolve()) for path in paths if is_cuda_library(path))


def loader_libraries(logs):
    # glibc writes one LD_DEBUG_OUTPUT.<pid> file for each executed process,
    # including the actual Lean/Python children of Lake and shell wrappers.
    paths = set()
    for log in logs:
        for match in re.finditer(r"calling init:\s*(.+)", Path(log).read_text(errors="replace")):
            path = match.group(1).strip()
            if is_cuda_library(path):
                paths.add(str(Path(path).resolve()))
    return sorted(paths)


def library_versions(paths):
    result = []
    for path in paths:
        lib = ctypes.CDLL(path)
        name = Path(path).name
        if name.startswith("libcudart.so"):
            query = lib.cudaRuntimeGetVersion
            query.argtypes = [ctypes.POINTER(ctypes.c_int)]
            query.restype = ctypes.c_int
            version = ctypes.c_int()
            if query(ctypes.byref(version)) != 0:
                raise ValueError(f"Cannot query CUDA runtime version: {path}")
            value = {"cuda_runtime_api": version.value}
        else:
            query = lib.cublasLtGetProperty if name.startswith("libcublasLt.so") else lib.cublasGetProperty
            query.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
            query.restype = ctypes.c_int
            parts = []
            for property_type in range(3):  # MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL
                part = ctypes.c_int()
                if query(property_type, ctypes.byref(part)) != 0:
                    raise ValueError(f"Cannot query cuBLAS version: {path}")
                parts.append(part.value)
            value = {"version": ".".join(map(str, parts))}
        result.append({"path": str(Path(path).resolve()), **value})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libtorch", type=Path, required=True)
    parser.add_argument("--cuda-home", type=Path, required=True)
    parser.add_argument("--previous", default="")
    args = parser.parse_args()
    print(library_path(args.libtorch, args.cuda_home, args.previous))


if __name__ == "__main__":
    main()
