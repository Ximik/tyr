import Tyr.Core.Typed.Neural
import Tyr.Core.TensorStruct

namespace torch.gpt

variable {device : Device}

abbrev Activation (s : Shape) := Tensor { shape := s, dtype := .Float32, device := device }
abbrev Tokens (s : Shape) := Tensor { shape := s, dtype := .Int64, device := device }

end torch.gpt
