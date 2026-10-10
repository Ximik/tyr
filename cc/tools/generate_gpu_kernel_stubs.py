#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import re



def write_if_changed(path: pathlib.Path, text: str) -> None:
    """Keep the timestamp when nothing changed, so Make does not recompile."""
    if path.exists() and path.read_text() == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


PATTERN = re.compile(rb"lean_launch_[A-Za-z0-9_]+")
GPU_KERNEL_ATTR_RE = re.compile(r"^\s*@\[gpu_kernel(?:\s+[^\]]+)?\]")
NAMESPACE_RE = re.compile(r"^\s*namespace\s+([A-Za-z0-9_.]+)")
DEF_RE = re.compile(
    r"^\s*(?:private\s+|unsafe\s+|partial\s+|protected\s+|noncomputable\s+)*"
    r"(?:def|abbrev|opaque)\s+([A-Za-z0-9_.']+)"
)


def without_comments(source: str) -> str:
    """Remove nested Lean comments while preserving line breaks and strings."""
    result: list[str] = []
    depth = 0
    quoted = False
    i = 0
    while i < len(source):
        pair = source[i:i + 2]
        char = source[i]
        if depth:
            if pair == "/-":
                depth += 1
                i += 2
            elif pair == "-/":
                depth -= 1
                i += 2
            else:
                result.append("\n" if char == "\n" else " ")
                i += 1
        elif quoted:
            result.append(char)
            i += 1
            if char == "\\" and i < len(source):
                result.append(source[i])
                i += 1
            elif char == '"':
                quoted = False
        elif pair == "/-":
            depth = 1
            result.append(" ")
            i += 2
        elif pair == "--":
            while i < len(source) and source[i] != "\n":
                i += 1
        else:
            result.append(char)
            quoted = char == '"'
            i += 1
    return "".join(result)


def source_symbols(kernel_src_root: pathlib.Path) -> set[str]:
    symbols: set[str] = set()
    for path in sorted(kernel_src_root.rglob("*.lean")):
        # Sections also consume `end`, but do not extend declaration names.
        scopes: list[str] = []
        pending_gpu_kernel = False
        for line in without_comments(path.read_text(encoding="utf-8")).splitlines():
            namespace_match = NAMESPACE_RE.match(line)
            if namespace_match:
                scopes.append(namespace_match.group(1))
            elif re.match(r"^\s*section(?:\s|$)", line):
                scopes.append("")
            elif re.match(r"^\s*end(?:\s|$)", line) and scopes:
                scopes.pop()
            attr_match = GPU_KERNEL_ATTR_RE.match(line)
            if attr_match:
                pending_gpu_kernel = True
                line = line[attr_match.end():]
            if not pending_gpu_kernel:
                continue
            def_match = DEF_RE.match(line)
            if def_match:
                name = def_match.group(1)
                namespace = ".".join(scope for scope in scopes if scope)
                full_name = f"{namespace}.{name}" if namespace else name
                if full_name.startswith("_root_."):
                    full_name = full_name[len("_root_."):]
                symbols.add("lean_launch_" + full_name.replace(".", "_"))
                pending_gpu_kernel = False
            elif line.strip():
                pending_gpu_kernel = False
    return symbols


def collect_symbols(ir_root: pathlib.Path, kernel_src_root: pathlib.Path) -> list[str]:
    symbols = source_symbols(kernel_src_root)
    if ir_root.exists():
        for path in ir_root.rglob("*.c.o.export"):
            data = path.read_bytes()
            for match in PATTERN.finditer(data):
                symbol = match.group().decode("utf-8")
                # Kernel sources are authoritative: old IR can retain launchers
                # after a declaration/file is removed. Keep IR-only launchers
                # for codegen tests outside this source tree, and architecture
                # specializations of declarations which still exist.
                base = re.sub(r"_SM(?:80|90|100)$", "", symbol)
                if not symbol.startswith("lean_launch_Tyr_GPU_Kernels_") or base in symbols:
                    symbols.add(symbol)
    return sorted(symbols)


def render(symbols: list[str]) -> str:
    lines = [
        "#include <lean/lean.h>",
        "#include <string>",
        "",
        "static lean_object* gpuKernelUnavailable(const char* launcher) {",
        '  std::string msg = "GPU kernel launcher unavailable in this build (missing NVCC/CUDA): ";',
        "  msg += launcher;",
        "  return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(msg.c_str())));",
        "}",
        "",
        'extern "C" {',
        "",
    ]
    for symbol in symbols:
        lines.extend([
            f"__attribute__((weak)) lean_object* {symbol}(...) {{",
            f'  return gpuKernelUnavailable("{symbol}");',
            "}",
            "",
        ])
    lines.append("} // extern \"C\"")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-root", required=True)
    parser.add_argument("--kernel-src-root", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parents[2] / "Tyr/GPU/Kernels")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ir_root = pathlib.Path(args.ir_root)
    output = pathlib.Path(args.output)
    write_if_changed(output, render(collect_symbols(ir_root, args.kernel_src_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
