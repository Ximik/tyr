import Tyr.Lib.Tokenizer.Types
import Tyr.Lib.Tokenizer.ByteLevel
import Tyr.Lib.Tokenizer.Pretokenize
import Tyr.Lib.Tokenizer.Encode
import Tyr.Lib.Tokenizer.Decode
import Tyr.Lib.Tokenizer.SpecialTokens
import Tyr.Lib.Tokenizer.TokenBytes
import Tyr.Lib.Tokenizer.Training
import Tyr.Lib.Tokenizer.IO

/-!
# Tyr.Lib.Tokenizer

`Tyr.Lib.Tokenizer` is the umbrella import for tokenization utilities in Tyr.
It re-exports core token types, byte-level processing, encode/decode pipelines,
training helpers, and model-specific tokenizer support.

## Major Components

- Core/token metadata: `Types`, `SpecialTokens`, `TokenBytes`.
- Text processing: `ByteLevel` and `Pretokenize`.
- Encoding/decoding paths: `Encode`, `Decode`, and `IO` helpers.
- Tokenizer training routines plus Qwen3/Qwen3.5/Qwen3.6 and Gemma4 support.

## Scope

Use this module when you need the full tokenizer stack through one import.
Specialized workflows can import individual tokenizer submodules directly.
-/
