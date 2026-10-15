#!/usr/bin/env python3
"""Compare the audited C++ ABI header with Lean's generated C declarations.

Run after building the relevant native adapters. The native compiler independently checks
the other side of this contract because tyr.cpp includes tyr_ffi_abi.h.
This intentionally checks the named, audited boundaries in that header; it
does not claim that the remaining legacy FFI surface has been audited. Use
--require-complete for an adapter whose recognized declarations must all appear
in the audited header. Add --symbol-prefix to scope generated declarations when
an adapter also references imports from other native subsystems; every audited
header declaration must belong to that scope.
"""

import argparse
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parent.parent
DECLARATION = re.compile(
    r"\b(lean_object\s*\*|uint(?:8|16|32|64)_t|size_t|double|float|void)"
    r"\s+(lean_(?:torch|float_buffer|parquet)_\w+)\s*\(([^()]*)\)\s*;"
)


def declarations(path: Path, symbol_prefix: str | None = None) -> dict[str, tuple[str, tuple[str, ...]]]:
    source = re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.S)
    result = {}
    for match in DECLARATION.finditer(source):
        ret, name, args = match.groups()
        if symbol_prefix is not None and not name.startswith(symbol_prefix):
            continue
        signature = (
            re.sub(r"\s+", "", ret),
            tuple(re.sub(r"\s+", "", arg) for arg in args.split(",") if arg.strip()),
        )
        if name in result and result[name] != signature:
            raise ValueError(f"{path}: conflicting declarations for {name}")
        result[name] = signature
    return result


def symbol_prefix(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", value):
        raise argparse.ArgumentTypeError("expected a nonempty C identifier prefix")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", type=Path, default=ROOT / "cc/src/tyr_ffi_abi.h")
    parser.add_argument("--generated", type=Path, nargs="+", default=[
        ROOT / ".lake/build/ir/Tyr/Basic.c",
        ROOT / ".lake/build/ir/Tyr/RawTensor.c",
        ROOT / ".lake/build/ir/Tyr/RawTensor/Neural.c",
    ])
    parser.add_argument("--require-complete", action="store_true",
                        help="reject generated declarations missing from the audited header")
    parser.add_argument("--symbol-prefix", type=symbol_prefix,
                        help="audit only generated symbols with this prefix; the entire header must match")
    args = parser.parse_args()
    try:
        expected = declarations(args.header)
        if not expected:
            raise ValueError("audited header contains no declarations")
        if args.symbol_prefix is not None:
            outside = sorted(name for name in expected if not name.startswith(args.symbol_prefix))
            if outside:
                raise ValueError("audited header declarations outside --symbol-prefix "
                                 f"{args.symbol_prefix}: {', '.join(outside)}")
        generated = {}
        for source in args.generated:
            for name, signature in declarations(source, args.symbol_prefix).items():
                if name in generated and generated[name] != signature:
                    raise ValueError(f"{source}: conflicting declarations for {name}")
                generated[name] = signature
    except (OSError, ValueError) as error:
        print(f"FFI ABI check: {error}", file=sys.stderr)
        return 1
    errors = []
    for name, signature in expected.items():
        actual = generated.get(name)
        if actual is None:
            errors.append(f"{name}: missing generated declaration; rebuild the native adapter")
        elif actual != signature:
            errors.append(f"{name}: header {signature}, Lean {actual}")
    if args.require_complete:
        for name in sorted(generated.keys() - expected.keys()):
            errors.append(f"{name}: generated declaration is not covered by the audited header")
    if errors:
        print("FFI ABI check failed:\n" + "\n".join(errors), file=sys.stderr)
        return 1
    print(f"FFI ABI check passed: {len(expected)} audited declarations match Lean C output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
