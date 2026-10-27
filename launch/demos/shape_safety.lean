import Tyr.Typed

open torch
open torch.Tensor

-- The batch dimension and both feature dimensions are visible in the type.
def project {device : Device}
    (x : Tensor { shape := #[32, 768], dtype := .Float32, device := device })
    (weight : Tensor { shape := #[512, 768], dtype := .Float32, device := device }) :
    Tensor { shape := #[32, 512], dtype := .Float32, device := device } :=
  linear x weight

#check project
