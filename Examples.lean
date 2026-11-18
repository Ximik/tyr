/-
  Examples.lean

  Root module for Tyr examples.

  This module imports all example implementations:
  - GPT: Simple GPT-2 style training (Shakespeare, etc.)
  - NanoChat: Advanced chat model with tools (modded-nanogpt port)
-/

-- Simple GPT
import Examples.GPT.GPT
import Examples.GPT.Train
import Examples.GPT.GPTDataLoader
import Examples.GPT.Pretraining
import Examples.GPT.NanoGPTCopy
import Examples.GPT.RiemannianNanoGPT

-- NanoChat (modded GPT + tools + tasks)
import Examples.NanoChat.NanoChat



-- Executable entry points excluded (they define global `main`):
-- import Examples.TrainGPT
-- import Examples.TrainDiffusion
-- import Examples.NanoChat.TrainNanoChat
