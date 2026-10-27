import Tyr.Model.Qwen36
import LeanTest

open torch
open torch.qwen36

private def tinyCfg : Config := Config.normalize {
  Config.qwen36_35B_A3B with
  vocab_size := 64
  hidden_size := 32
  intermediate_size := 64
  num_hidden_layers := 2
  num_attention_heads := 4
  num_key_value_heads := 2
  head_dim := 8
  max_position_embeddings := 128
  linear_key_head_dim := 4
  linear_value_head_dim := 4
  linear_num_key_heads := 2
  linear_num_value_heads := 4
  layer_types := #[.linearAttention, .fullAttention]
  moe_intermediate_size := 8
  shared_expert_intermediate_size := 8
  num_experts_per_tok := 2
  num_experts := 4
}

@[test]
def testQwen36InitAndForward : IO Unit := do
  let model ← Qwen36ForCausalLM.init tinyCfg false
  let ids : Tensor { shape := #[1, 4], dtype := .Int64, device := .CPU } := (Tensor.fromInt64Array #[1, 2, 3, 4]).reshapeChecked #[1, 4]
  let logits := model.forward tinyCfg ids
  LeanTest.assertEqual logits.runtimeShape #[1, 4, tinyCfg.vocab_size]
    "Qwen3.6 forward should return [batch, seq, vocab]"

@[test]
def testQwen36ConfigDefaultsMatchPublishedCheckpoint : IO Unit := do
  let cfg := Config.qwen36_35B_A3B
  LeanTest.assertEqual cfg.hidden_size 2048
    "Qwen3.6-35B-A3B should expose the published hidden size"
  LeanTest.assertEqual cfg.max_position_embeddings 262144
    "Qwen3.6-35B-A3B should expose the published native context length"
  LeanTest.assertEqual cfg.mrope_interleaved true
    "Qwen3.6-35B-A3B should default to mRoPE interleaving"
  LeanTest.assertEqual cfg.mtp_num_hidden_layers 1
    "Qwen3.6-35B-A3B should preserve the published MTP depth"
  LeanTest.assertEqual cfg.layer_types.size 40
    "Qwen3.6-35B-A3B should synthesize the published 40-layer hybrid schedule"
