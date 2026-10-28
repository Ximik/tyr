import Tyr.Model.Qwen35.Model
import LeanTest

open torch torch.qwen35

namespace Tests.Qwen35DTypes

/-- Model outputs and caches keep the selected activation dtype. -/
example {device : Device} {cfg : Config} {batch seq : Nat}
    [DType.ConstantSupported .BFloat16 device]
    (model : Qwen35ForCausalLM (device := device) cfg .BFloat16)
    (ids : Tensor ⟨#[batch, seq], .Int64, device⟩) :
    Tensor ⟨#[batch, seq, cfg.vocab_size], .BFloat16, device⟩ :=
  model.forward cfg ids

/-- Norm parameters may retain a separate checkpoint dtype. -/
example {device : Device} {n dim : Nat}
    (norm : Qwen35RMSNorm (device := device) dim)
    (input : Tensor ⟨#[n, dim], .BFloat16, device⟩) :
    Tensor ⟨#[n, dim], .BFloat16, device⟩ := norm.forward2d input

end Tests.Qwen35DTypes
