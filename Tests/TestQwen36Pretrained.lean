import Tyr.Model.Qwen36
import Tyr.Model.Qwen35
import LeanTest

open torch
open torch.qwen36

@[test]
def testQwen36CollectionListCoverage : IO Unit := do
  LeanTest.assertTrue (hub.isQwen36CollectionRepoId "Qwen/Qwen3.6-35B-A3B")
    "Qwen3.6 collection helper should include the published 35B-A3B checkpoint"
  LeanTest.assertTrue (torch.qwen35.hub.isQwen35CollectionRepoId "Qwen/Qwen3.6-35B-A3B")
    "shared Qwen3.5 resolver should recognize the Qwen3.6 checkpoint"

@[test]
def testQwen36MultimodalConfigLoad : IO Unit := do
  let path := "/tmp/qwen36_vl_config_test.json"
  let json :=
    "{\"architectures\":[\"Qwen3_5MoeForConditionalGeneration\"]," ++
    "\"image_token_id\":248056,\"video_token_id\":248057," ++
    "\"vision_start_token_id\":248053,\"vision_end_token_id\":248054," ++
    "\"text_config\":{" ++
    "\"attention_bias\":false,\"attention_dropout\":0.0,\"attn_output_gate\":true," ++
    "\"bos_token_id\":248044,\"eos_token_id\":248044,\"full_attention_interval\":4," ++
    "\"head_dim\":256,\"hidden_act\":\"silu\",\"hidden_size\":2048," ++
    "\"linear_conv_kernel_dim\":4,\"linear_key_head_dim\":128," ++
    "\"linear_num_key_heads\":16,\"linear_num_value_heads\":32,\"linear_value_head_dim\":128," ++
    "\"mamba_ssm_dtype\":\"float32\",\"max_position_embeddings\":262144," ++
    "\"moe_intermediate_size\":512,\"mtp_num_hidden_layers\":1," ++
    "\"mtp_use_dedicated_embeddings\":false,\"num_attention_heads\":16," ++
    "\"num_experts\":256,\"num_experts_per_tok\":8,\"num_hidden_layers\":40," ++
    "\"num_key_value_heads\":2,\"partial_rotary_factor\":0.25,\"rms_norm_eps\":1e-6," ++
    "\"rope_parameters\":{\"mrope_interleaved\":true,\"mrope_section\":[11,11,10]," ++
    "\"partial_rotary_factor\":0.25,\"rope_theta\":10000000},\"shared_expert_intermediate_size\":512," ++
    "\"tie_word_embeddings\":false,\"use_cache\":true,\"vocab_size\":248320}," ++
    "\"vision_config\":{" ++
    "\"depth\":27,\"hidden_act\":\"gelu_pytorch_tanh\",\"hidden_size\":1152," ++
    "\"in_channels\":3,\"initializer_range\":0.02,\"intermediate_size\":4304," ++
    "\"num_heads\":16,\"num_position_embeddings\":2304,\"out_hidden_size\":2048," ++
    "\"patch_size\":16,\"spatial_merge_size\":2,\"temporal_patch_size\":2}}"
  IO.FS.writeFile path json

  let cfg ← VLConfig.loadFromFile path
  LeanTest.assertEqual cfg.text_config.hidden_size 2048
    "Qwen3.6 text hidden size should load from nested text_config"
  LeanTest.assertEqual cfg.text_config.num_experts 256
    "Qwen3.6 MoE expert count should load from nested text_config"
  LeanTest.assertEqual cfg.vision_config.out_hidden_size 2048
    "Qwen3.6 vision projection size should load from nested vision_config"
  LeanTest.assertEqual cfg.image_token_id 248056
    "Qwen3.6 multimodal image token id should load from root config"

@[test]
def testQwen36TextConfigLoadFromMultimodalRoot : IO Unit := do
  let path := "/tmp/qwen36_text_config_from_vl_root.json"
  let json :=
    "{\"architectures\":[\"Qwen3_5MoeForConditionalGeneration\"]," ++
    "\"text_config\":{" ++
    "\"attention_bias\":false,\"attention_dropout\":0.0,\"attn_output_gate\":true," ++
    "\"bos_token_id\":248044,\"eos_token_id\":248044,\"full_attention_interval\":4," ++
    "\"head_dim\":256,\"hidden_act\":\"silu\",\"hidden_size\":2048," ++
    "\"linear_conv_kernel_dim\":4,\"linear_key_head_dim\":128," ++
    "\"linear_num_key_heads\":16,\"linear_num_value_heads\":32,\"linear_value_head_dim\":128," ++
    "\"mamba_ssm_dtype\":\"float32\",\"max_position_embeddings\":262144," ++
    "\"moe_intermediate_size\":512,\"mtp_num_hidden_layers\":1," ++
    "\"mtp_use_dedicated_embeddings\":false,\"num_attention_heads\":16," ++
    "\"num_experts\":256,\"num_experts_per_tok\":8,\"num_hidden_layers\":40," ++
    "\"num_key_value_heads\":2,\"partial_rotary_factor\":0.25,\"rms_norm_eps\":1e-6," ++
    "\"rope_parameters\":{\"mrope_interleaved\":true,\"mrope_section\":[11,11,10]," ++
    "\"partial_rotary_factor\":0.25,\"rope_theta\":10000000},\"shared_expert_intermediate_size\":512," ++
    "\"tie_word_embeddings\":false,\"use_cache\":true,\"vocab_size\":248320}}"
  IO.FS.writeFile path json

  let cfg ← Config.loadFromFile path
  LeanTest.assertEqual cfg.hidden_size 2048
    "Qwen3.6 text loader should read nested text_config from multimodal root config"
