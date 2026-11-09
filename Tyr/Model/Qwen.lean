/-
  Tyr/Model/Qwen.lean

  Qwen3 transformer building blocks (config, RoPE, attention, MLP, layers,
  full model, text embedder, weight loading).

  This is the shared substrate for the Qwen3 and Qwen2.5-Omni causal LMs
  (and is also reused by Qwen3-ASR, Qwen3.5, Qwen3-TTS, and Gemma4
  components). The Flux examples use only the embedder for text encoding.
  Re-exports all Qwen components.
-/

import Tyr.Model.Qwen.Config
import Tyr.Model.Qwen.ConfigIO
import Tyr.Model.Qwen.RoPE
import Tyr.Model.Qwen.Attention
import Tyr.Model.Qwen.MLP
import Tyr.Model.Qwen.Layer
import Tyr.Model.Qwen.Model
import Tyr.Model.Qwen.Embedder
import Tyr.Model.Qwen.Weights
