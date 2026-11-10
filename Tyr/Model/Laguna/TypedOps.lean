import Tyr.Model.Qwen35.TypedOps
import Tyr.RawTensor.Laguna

namespace torch.laguna

variable {device : Device} {dtype : DType}
variable [DType.RealFloating dtype]
def scatterAdd {s t : Shape} (input : Tensor { dtype := dtype, device := device, shape := s }) (dim : Nat)
    (indices : Tensor { device := device, shape := t, dtype := .Int64 }) (src : Tensor { dtype := dtype, device := device, shape := t }) : Tensor { dtype := dtype, device := device, shape := s } :=
  Tensor.ofRaw! (RawTensor.lagunaScatterAdd input.raw dim indices.raw src.raw)

def transposeMatrixBatch {b m n : Nat} {d : DType}
    (x : Tensor { device := device, shape := #[b, m, n], dtype := d }) : Tensor { device := device, shape := #[b, n, m], dtype := d } := x.transpose 1 2

def mulSame {s : Shape} (x y : Tensor { dtype := dtype, device := device, shape := s }) : Tensor { dtype := dtype, device := device, shape := s } := Tensor.mul x y
def addSame {s : Shape} (x y : Tensor { dtype := dtype, device := device, shape := s }) : Tensor { dtype := dtype, device := device, shape := s } := Tensor.add x y

def transposeAttention {b s h d : Nat} {dt : DType}
    (x : Tensor { device := device, shape := #[b, s, h, d], dtype := dt }) : Tensor { device := device, shape := #[b, h, s, d], dtype := dt } := x.transpose 1 2

def randnDynamic (shape : Shape) (device : Device := .CPU) (dtype : DType := .Float32) : IO (Tensor { dtype := dtype, device := device, shape := shape }) := do
  pure ((← Tensor.randn (device := device) shape).castDType dtype)

def hasGradFn {σ : StaticSpec} (x : Tensor σ) : Bool :=
  RawTensor.lagunaHasGradFn x.raw

end torch.laguna
