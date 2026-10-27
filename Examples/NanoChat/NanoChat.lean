/-
  Examples/NanoChat/NanoChat.lean

  Root module for NanoChat - a port of nanochat to Tyr.

  NanoChat is a chat-oriented LLM with:
  - ModdedGPT architecture (YaRN RoPE, sliding window attention)
  - Supervised fine-tuning (ChatSFT)
  - RL fine-tuning (GRPO)
  - Tool-augmented generation
  - LLM evaluation tasks (ARC, MMLU, GSM8K, etc.)
-/

-- Model
import Examples.NanoChat.ModdedGPT

-- Training
import Examples.NanoChat.ModdedTrain
import Examples.NanoChat.ChatSFT
import Examples.NanoChat.GRPO
import Examples.NanoChat.Pipeline

-- Generation
import Examples.NanoChat.Generator.State
import Examples.NanoChat.Generator.TypedState
import Examples.NanoChat.Generator.KVCache
import Examples.NanoChat.Generator.Tools
import Examples.NanoChat.Generator.Engine

-- Tasks & Evaluation
import Examples.NanoChat.Tasks.LLM
import Examples.NanoChat.Eval.CORE
