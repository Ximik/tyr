#!/usr/bin/env python3
"""Compare the audited C++ ABI header with Lean's generated C declarations.

Run after building the relevant native adapters. The native compiler independently checks
the other side of this contract because tyr.cpp includes tyr_ffi_abi.h.
This intentionally checks the named, audited boundaries in that header; it
does not claim that the remaining legacy FFI surface has been audited.
"""

import argparse
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parent.parent
DECLARATION = re.compile(
    r"\b(lean_object\s*\*|uint(?:8|16|32|64)_t|size_t|double|float|void)"
    r"\s+(lean_torch_\w+)\s*\(([^()]*)\)\s*;"
)


def declarations(path: Path) -> dict[str, tuple[str, tuple[str, ...]]]:
    source = re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.S)
    result = {}
    for match in DECLARATION.finditer(source):
        ret, name, args = match.groups()
        signature = (
            re.sub(r"\s+", "", ret),
            tuple(re.sub(r"\s+", "", arg) for arg in args.split(",") if arg.strip()),
        )
        if name in result and result[name] != signature:
            raise ValueError(f"{path}: conflicting declarations for {name}")
        result[name] = signature
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", type=Path, default=ROOT / "cc/src/tyr_ffi_abi.h")
    parser.add_argument("--generated", type=Path, nargs="+", default=[
        ROOT / ".lake/build/ir/Tyr/Torch.c",
        ROOT / ".lake/build/ir/Tyr/RawTensor.c",
        ROOT / ".lake/build/ir/Tyr/RawTensor/Neural.c",
    ])
    args = parser.parse_args()
    try:
        expected = declarations(args.header)
        generated = {}
        for source in args.generated:
            for name, signature in declarations(source).items():
                if name in generated and generated[name] != signature:
                    raise ValueError(f"{source}: conflicting declarations for {name}")
                generated[name] = signature
    except (OSError, ValueError) as error:
        print(f"FFI ABI check: {error}", file=sys.stderr)
        return 1
    if not expected:
        print("FFI ABI check: audited header contains no declarations", file=sys.stderr)
        return 1
    errors = []
    for name, signature in expected.items():
        actual = generated.get(name)
        if actual is None:
            errors.append(f"{name}: missing generated declaration; rebuild the native adapter")
        elif actual != signature:
            errors.append(f"{name}: header {signature}, Lean {actual}")
    if errors:
        print("FFI ABI check failed:\n" + "\n".join(errors), file=sys.stderr)
        return 1
    print(f"FFI ABI check passed: {len(expected)} audited declarations match Lean C output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
