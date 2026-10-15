import Tyr.Model.Gemma4.Model

open torch torch.gemma4

/-- The activation and cache index is selected explicitly, even with independent
checkpoint dtypes for normalization and routing parameters. -/
example {device : Device} {cfg : Config} {batch seq : UInt64}
    [DType.ConstantSupported .BFloat16 device]
    (model : Gemma4ForCausalLM (device := device) cfg .BFloat16)
    (ids : Tensor ⟨#[batch, seq], .Int64, device⟩) :
    Tensor ⟨#[batch, seq, cfg.vocab_size], .BFloat16, device⟩ :=
  model.forward cfg ids
