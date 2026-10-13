import Tyr.Model.Qwen3
import Tyr.Model.Qwen25Omni
import LeanTest

namespace Tests.QwenTypedLoading
open torch Lean

private def cfg : qwen3.Config := {
  vocab_size := 8, hidden_size := 8, intermediate_size := 16,
  num_hidden_layers := 1, num_attention_heads := 2, num_key_value_heads := 1,
  head_dim := 4, max_position_embeddings := 16
}

/-- Generate actual SafeTensors bytes, without Python or downloaded weights. -/
private def writeCheckpoint (dir : System.FilePath) (bf16 omni tied : Bool) : IO Unit := do
  let namePrefix := if omni then "thinker." else ""
  let entries : Array (String × Array Nat) := #[
    ("model.embed_tokens.weight", #[8, 8]),
    ("model.layers.0.input_layernorm.weight", #[8]),
    ("model.layers.0.post_attention_layernorm.weight", #[8]),
    ("model.layers.0.self_attn.q_proj.weight", #[8, 8]),
    ("model.layers.0.self_attn.k_proj.weight", #[4, 8]),
    ("model.layers.0.self_attn.v_proj.weight", #[4, 8]),
    ("model.layers.0.self_attn.o_proj.weight", #[8, 8]),
    ("model.layers.0.mlp.gate_proj.weight", #[16, 8]),
    ("model.layers.0.mlp.up_proj.weight", #[16, 8]),
    ("model.layers.0.mlp.down_proj.weight", #[8, 16]),
    ("model.norm.weight", #[8])
  ] ++ (if tied then #[] else #[("lm_head.weight", #[8, 8])])
  let mut fields : List (String × Json) := []
  let mut index : List (String × Json) := []
  let mut data := ByteArray.empty
  for (name, shape) in entries do
    let start := data.size
    for _ in [:shape.foldl (· * ·) 1] do
      -- Exactly 1.0 in either storage dtype.
      data := data ++ (if bf16 then ByteArray.mk #[128, 63] else ByteArray.mk #[0, 0, 128, 63])
    let name := namePrefix ++ name
    fields := fields ++ [(name, Json.mkObj [
      ("dtype", toJson (if bf16 then "BF16" else "F32")),
      ("shape", toJson shape), ("data_offsets", toJson #[start, data.size])])]
    index := index ++ [(name, toJson "model.safetensors")]
  let header := (Json.mkObj fields).compress.toUTF8
  let mut size := ByteArray.empty
  for i in [:8] do
    size := size.push ((header.size.toUInt64 >>> (i * 8).toUInt64).toUInt8)
  IO.FS.writeBinFile (dir / "model.safetensors") (size ++ header ++ data)
  IO.FS.writeFile (dir / "model.safetensors.index.json")
    (Json.mkObj [("weight_map", Json.mkObj index)]).compress

@[test]
def testQwenTypedSingleAndShardedLoading : IO Unit := do
  for bf16 in [false, true] do
    for omni in [false, true] do
      for tied in [false, true] do
        IO.FS.withTempDir fun dir => do
          writeCheckpoint dir bf16 omni tied
          for sharded in [false, true] do
            let model ← if omni then
                if sharded then qwen25omni.Qwen25OmniForCausalLM.loadSharded dir.toString cfg
                else qwen25omni.Qwen25OmniForCausalLM.load (dir / "model.safetensors").toString cfg
              else
                if sharded then qwen3.Qwen3ForCausalLM.loadSharded dir.toString cfg
                else qwen3.Qwen3ForCausalLM.load (dir / "model.safetensors").toString cfg
            LeanTest.assertEqual model.tieWordEmbeddings tied
            LeanTest.assertEqual model.lmHead.actualSpec.dtype DType.Float32
            LeanTest.assertEqual model.model.embed_tokens.actualSpec.dtype DType.Float32
            LeanTest.assertEqual (← model.lmHead.toFloatArray) (Array.replicate 64 1.0)
            let ids := (Tensor.fromInt64Array #[1, 2]).reshapeChecked #[1, 2]
            let logits := model.forward ids
            match logits.validate with
            | .error message => LeanTest.fail message
            | .ok () => pure ()
            LeanTest.assertTrue ((← logits.toFloatArray).all Float.isFinite)
              "Loaded checkpoint must execute with finite logits"

end Tests.QwenTypedLoading
