#!/usr/bin/env python3
"""Require complete exception guards on audited recoverable native exports.

Inventory is derived from IO/Except result types in Tyr/RawTensor.lean, its
submodules except OwnedKV, Tyr/Basic.lean, and Tyr/ConstantSupport.lean.
Other native subsystems are outside this explicitly reviewed scope.
Every source definition, including inactive
platform branches, must contain only a checkedCall/checkedIoCall return with a
literal context and reference-capture lambda. This forbids throwable work in
argument evaluation before the guard as well as statements before/after it.

This is a conservative source convention check, complementary to the generated FFI header
and native runtime tests. It does not prove C++ exception interoperability or
the implementation of tyr_checked_result.h. Macro-generated exports and result
type aliases require explicit scanner support rather than silently passing.
No compiler, LibTorch installation, or generated build output is required.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parent.parent
NATIVE_SUFFIXES = {".cpp", ".cc", ".cxx", ".cu", ".mm", ".c", ".h", ".hpp"}
# The existing pure result constructors are deliberately explicit. A new result
# alias must not silently hide an IO/Except export from the inventory.
PURE_RESULT_HEADS = {"RawTensor", "DType", "Bool", "Float", "String", "Shape", "Device",
                     "FloatArray", "Int64", "Array"}

# This one polymorphic extern never returns a value. Audit its complete native
# body instead of permitting arbitrary type variables to hide IO/Except aliases.
POLYMORPHIC_ABORT = ("Tyr/Basic.lean", "lean_torch_native_invariant_failure", "α")
FIXED_ABORT_BODY = [
    "std", "::", "fprintf", "(", "stderr", ",", "<string>", ",",
    "lean_string_cstr", "(", "message", ")", ")", ";",
    "std", "::", "fflush", "(", "stderr", ")", ";",
    "std", "::", "abort", "(", ")", ";",
]

# These build configurations only construct a fixed Lean IO.error; they run no
# C++/LibTorch operation. The body still has to match FIXED_IO_ERROR exactly.
FIXED_ERROR_STUBS = {
    ("cc/src/tyr_media_stub.cpp", "lean_torch_media_load_image_patchified"),
    ("cc/src/tyr_media_stub.cpp", "lean_torch_media_load_video_patchified"),
    ("cc/src/tyr_media_stub.cpp", "lean_torch_media_load_image_patch_grid_gemma4"),
    ("cc/src/apple_vl_processor.mm", "lean_torch_media_load_image_patchified"),
    ("cc/src/apple_vl_processor.mm", "lean_torch_media_load_video_patchified"),
    ("cc/src/tyr_laguna_moe.cu", "lean_torch_laguna_moe_fp4_forward"),
}
FIXED_IO_ERROR = [
    "return", "lean_io_result_mk_error", "(", "lean_mk_io_user_error", "(",
    "lean_mk_string", "(", "<string>", ")", ")", ")", ";",
]


@dataclass(frozen=True)
class Token:
    text: str
    line: int


@dataclass(frozen=True)
class Export:
    symbol: str
    carrier: str
    path: str
    line: int


@dataclass(frozen=True)
class Definition:
    symbol: str
    path: str
    line: int
    body: tuple[Token, ...]


def tokens(source: str, language: str) -> list[Token]:
    """Lex delimiters without interpreting comments or quoted text as code."""
    result = []
    i = 0
    line = 1
    while i < len(source):
        start, start_line = i, line
        if source[i].isspace():
            i += 1
        elif source.startswith("--" if language == "lean" else "//", i):
            end = source.find("\n", i)
            i = len(source) if end < 0 else end
        elif source.startswith("/-" if language == "lean" else "/*", i):
            opening, closing = ("/-", "-/") if language == "lean" else ("/*", "*/")
            i += 2
            depth = 1
            while depth:
                if i >= len(source):
                    raise ValueError(f"line {start_line}: unterminated comment")
                if language == "lean" and source.startswith(opening, i):
                    depth += 1
                    i += 2
                elif source.startswith(closing, i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
        elif language == "cpp" and source.startswith('R"', i):
            match = re.match(r'R"([^\s()\\]{0,16})\(', source[i:])
            if not match:
                raise ValueError(f"line {line}: unsupported C++ raw string")
            closing = ")" + match[1] + '"'
            end = source.find(closing, i + match.end())
            if end < 0:
                raise ValueError(f"line {line}: unterminated C++ raw string")
            i = end + len(closing)
            result.append(Token("<string>", line))
        elif source[i] == '"' or (language == "cpp" and source[i] == "'"):
            quote = source[i]
            i += 1
            while i < len(source) and source[i] != quote:
                i += 2 if source[i] == "\\" else 1
            if i >= len(source):
                raise ValueError(f"line {line}: unterminated quoted literal")
            i += 1
            # Keep Lean strings for the extern name; C++ strings are never code.
            result.append(Token(source[start:i] if language == "lean" else
                                "<string>" if quote == '"' else "<char>", line))
        else:
            match = re.match(r"[\w']+|::|:=|=>|->|[^\s]", source[i:])
            if not match:
                raise ValueError(f"line {line}: unsupported token")
            result.append(Token(match[0], line))
            i += len(match[0])
        line += source[start:i].count("\n")
    return result


def group_end(ts: list[Token] | tuple[Token, ...], start: int) -> int:
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack = [pairs[ts[start].text]]
    for i in range(start + 1, len(ts)):
        value = ts[i].text
        if value in pairs:
            stack.append(pairs[value])
        elif value in pairs.values():
            if value != stack.pop():
                raise ValueError(f"line {ts[i].line}: mismatched delimiter {value}")
            if not stack:
                return i
    raise ValueError(f"line {ts[start].line}: unclosed delimiter {ts[start].text}")


def lean_exports(source: str, path: str) -> list[Export]:
    ts = tokens(source, "lean")
    exports = []
    for i in range(len(ts) - 1):
        if (ts[i].text, ts[i + 1].text) != ("@", "["):
            continue
        end = group_end(ts, i + 1)
        attribute = ts[i + 2:end]
        external = [j for j, token in enumerate(attribute) if token.text == "extern"]
        if not external:
            continue
        if len(external) != 1 or external[0] + 1 >= len(attribute):
            raise ValueError(f"{path}:{ts[i].line}: unsupported extern attribute")
        symbol = attribute[external[0] + 1].text
        if not re.fullmatch(r'"[A-Za-z_][A-Za-z_0-9]*"', symbol):
            raise ValueError(f"{path}:{ts[i].line}: extern needs a literal C symbol")
        j = end + 1
        while j < len(ts) and ts[j].text in {"private", "protected", "noncomputable", "unsafe"}:
            j += 1
        if j >= len(ts) or ts[j].text not in {"opaque", "def"}:
            raise ValueError(f"{path}:{ts[i].line}: unsupported extern declaration syntax")
        j += 2  # declaration keyword and name
        while j < len(ts) and ts[j].text != ":":
            if ts[j].text in {"(", "[", "{"}:
                j = group_end(ts, j) + 1
            elif ts[j].text in {"@", ":=", "end", "namespace", "opaque", "def"}:
                break
            else:
                j += 1  # qualified name or universe annotation
        if j >= len(ts) or ts[j].text != ":":
            raise ValueError(f"{path}:{ts[i].line}: extern needs an explicit result type")
        j += 1
        while j < len(ts) and ts[j].text == "(":
            j += 1
        if j < len(ts) and ts[j].text == "_root_":
            j += 2
        if j >= len(ts):
            raise ValueError(f"{path}:{ts[i].line}: missing extern result type")
        carrier = ts[j].text
        if j + 1 < len(ts) and ts[j + 1].text == ".":
            raise ValueError(f"{path}:{ts[i].line}: qualified extern result constructor; "
                             "resolve its carrier explicitly for the recovery audit")
        if (path, symbol[1:-1], carrier) == POLYMORPHIC_ABORT:
            exports.append(Export(symbol[1:-1], "NoReturn", path, ts[i].line))
        elif carrier in {"IO", "Except"}:
            if carrier == "Except":
                error_type = j + 1
                if error_type < len(ts) and ts[error_type].text == "_root_":
                    error_type += 2
                if (error_type >= len(ts) or ts[error_type].text != "String"
                        or error_type + 1 < len(ts) and ts[error_type + 1].text == "."):
                    raise ValueError(f"{path}:{ts[i].line}: checkedCall constructs Except String; "
                                     "other error types require a separate recovery adapter")
            exports.append(Export(symbol[1:-1], carrier, path, ts[i].line))
        elif carrier not in PURE_RESULT_HEADS:
            raise ValueError(f"{path}:{ts[i].line}: unrecognized extern result constructor {carrier}; "
                             "resolve any type alias and extend the recovery audit")
    return exports


def native_definitions(source: str, path: str, symbols: set[str]) -> list[Definition]:
    ts = tokens(source, "cpp")
    definitions = []
    for i, token in enumerate(ts):
        if token.text not in symbols:
            continue
        if i + 1 >= len(ts) or ts[i + 1].text != "(":
            raise ValueError(f"line {token.line}: unsupported use of recoverable export {token.text}; "
                             "macro-generated definitions require explicit audit support")
        end = group_end(ts, i + 1)
        body = end + 1
        # Find definitions independently of their return spelling. Requiring a
        # lean_object* prefix would silently miss an unguarded platform variant
        # using a typedef or `auto f(...) -> lean_object*` whenever another
        # platform already supplied one recognized definition.
        while body < len(ts):
            if ts[body].text in {"noexcept", "throw", "__attribute__", "__declspec"}:
                body += 1
                if body < len(ts) and ts[body].text == "(":
                    body = group_end(ts, body) + 1
            elif ts[body].text == "[":  # standard [[attributes]]
                body = group_end(ts, body) + 1
            else:
                break
        if body < len(ts) and ts[body].text == "->":
            body += 1
            while body < len(ts) and ts[body].text not in {"{", ";"}:
                if ts[body].text in {"(", "["}:
                    body = group_end(ts, body) + 1
                else:
                    body += 1
        if body < len(ts) and ts[body].text == "{":
            close = group_end(ts, body)
            definitions.append(Definition(token.text, path, token.line, tuple(ts[body + 1:close])))
        elif body >= len(ts) or re.fullmatch(r"[\w']+", ts[body].text):
            raise ValueError(f"line {token.line}: unsupported declaration/use of {token.text}; "
                             "extend native definition discovery instead of skipping this variant")
        # A declaration ending in ';' is not an implementation. Other expression
        # punctuation can follow calls, but never an unadorned function header.
    return definitions


def guarded_body(body: tuple[Token, ...], carrier: str) -> bool:
    wrapper = "checkedIoCall" if carrier == "IO" else "checkedCall"
    prefix = ["return", "tyr", "::", wrapper, "(", "<string>", ",", "[", "&", "]", "{"]
    if [t.text for t in body[:len(prefix)]] != prefix:
        return False
    end = group_end(body, len(prefix) - 1)
    return [t.text for t in body[end + 1:]] == [")", ";"]


def audit(root: Path) -> tuple[list[str], Counter]:
    errors = []
    exports = {}
    lean_paths = [root / "Tyr/Basic.lean", root / "Tyr/RawTensor.lean",
                  root / "Tyr/ConstantSupport.lean",
                  *sorted(path for path in (root / "Tyr/RawTensor").rglob("*.lean")
                          if path.name != "OwnedKV.lean")]
    for path in lean_paths:
        label = path.relative_to(root).as_posix()
        try:
            for export in lean_exports(path.read_text(), label):
                previous = exports.get(export.symbol)
                if previous and previous.carrier != export.carrier:
                    errors.append(f"{label}:{export.line}: conflicting carriers for {export.symbol}")
                exports[export.symbol] = export
        except (OSError, ValueError) as error:
            errors.append(f"{label}: {error}")
    if not exports:
        errors.append("No recoverable extern declarations found")
    implementations = {symbol: [] for symbol in exports}
    for path in sorted((root / "cc/src").rglob("*")):
        if path.suffix not in NATIVE_SUFFIXES:
            continue
        label = path.relative_to(root).as_posix()
        try:
            for definition in native_definitions(path.read_text(), label, set(exports)):
                implementations[definition.symbol].append(definition)
        except (OSError, ValueError) as error:
            errors.append(f"{label}: {error}")
    counts = Counter(export.carrier for export in exports.values())
    for symbol, export in sorted(exports.items()):
        source = f"{export.path}:{export.line} ({export.carrier})"
        if not implementations[symbol]:
            errors.append(f"{source}: no recognized native definition for {symbol}")
        for definition in implementations[symbol]:
            if export.carrier == "NoReturn":
                if [t.text for t in definition.body] == FIXED_ABORT_BODY:
                    counts["noreturn definitions"] += 1
                else:
                    errors.append(
                        f"{definition.path}:{definition.line}: {symbol} from {source} must contain only "
                        "fprintf of the borrowed message, fflush(stderr), and std::abort()")
                continue
            if guarded_body(definition.body, export.carrier):
                counts["guarded definitions"] += 1
                continue
            if (export.carrier == "IO" and (definition.path, symbol) in FIXED_ERROR_STUBS
                    and [t.text for t in definition.body] == FIXED_IO_ERROR):
                counts["fixed-error stubs"] += 1
                continue
            wrapper = "checkedIoCall" if export.carrier == "IO" else "checkedCall"
            errors.append(
                f"{definition.path}:{definition.line}: {symbol} from {source} must contain only "
                f'return tyr::{wrapper}("literal context", [&] {{ ... }}); '
                "(no work before the guard, computed context, or initializing captures)")
    return errors, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root")
    args = parser.parse_args()
    errors, counts = audit(args.root.resolve())
    if errors:
        print("FFI recovery check failed:\n" + "\n".join(errors), file=sys.stderr)
        return 1
    print("FFI recovery check passed: " + ", ".join(f"{count} {kind}" for kind, count in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
