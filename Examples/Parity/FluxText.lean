/-
  Examples/Parity/FluxText.lean

  Parity probe for `parity/tests/flux_text.py`. `--weights` is a FLUX.2-klein
  directory with `tokenizer/` and `text_encoder/`. Request:
  `{"prompt", "input_ids", "attention_mask"}`, the ids and mask being the
  reference's (512 each). Writes tyr's own `input_ids` and `attention_mask` for
  the prompt and the Float32 prompt embedding of the request's ids
  (`embeddings`, `[1, 512, 7680]`).
-/
import Examples.Parity.Common
import Tyr.Model.Qwen
import Tyr.Model.Qwen.Weights
import Tyr.Model.Qwen3.ConfigIO
import Tyr.Model.Qwen.Tokenizer

namespace Examples.Parity.FluxText

open Lean torch torch.qwen

structure Request where
  prompt : String
  input_ids : Array Nat
  attention_mask : Array Nat
  deriving FromJson

def maxSeqLen : Nat := 512

def probe (args : Args) (device : Device) (json : Json) : IO Output := do
  let request : Request ← IO.ofExcept (fromJson? json)
  let dir := args.weights.toString
  let tok ← tokenizer.qwen3.loadTokenizer s!"{dir}/tokenizer"
  let (ownIds, ownMask) := tokenizer.qwen3.encodePrompt tok request.prompt maxSeqLen
  let ids ← IO.ofExcept (Tensor.fromInt64ArrayAs? (request.input_ids.map (·.toInt64)) #[1, maxSeqLen])
  let mask ← IO.ofExcept (Tensor.fromInt64ArrayAs? (request.attention_mask.map (·.toInt64)) #[1, maxSeqLen])
  let cfg ← qwen3.Config.loadFromPretrainedDir s!"{dir}/text_encoder"
  let embedder ← loadQwenFluxEmbedder s!"{dir}/text_encoder" cfg maxSeqLen (device := device)
  let embeddings := embedder.encodeMasked (batch := 1) (seq := maxSeqLen) cfg maxSeqLen
    (ids.to device) (mask.to device)
  pure {
    json := Json.mkObj [
      ("input_ids", toJson ((tokenizer.qwen3.toInt64Array ownIds).map (·.toInt))),
      ("attention_mask", toJson ((tokenizer.qwen3.toInt64Array ownMask).map (·.toInt)))]
    tensors := #[("embeddings", Tensor.toTensor embeddings)]
  }

end Examples.Parity.FluxText

def main (argv : List String) : IO UInt32 :=
  Examples.Parity.run argv Examples.Parity.FluxText.probe
