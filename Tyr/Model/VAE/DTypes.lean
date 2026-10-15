import Tyr.Typed.Promotion
import Tyr.Typed.Tensor

namespace torch.vae

/-- Stored normalization statistics may have independent precision. Their
mathematical promotion must still produce the decoder's activation dtype. -/
structure AutoEncoderDTypes where
  compute : RealFloatingDType
  mean : ExactDType
  variance : RealFloatingDType
  scaled : DType
  scalePromotion : DType.Promotes compute.val variance.val scaled
  offsetPromotion : DType.Promotes scaled mean.val compute.val

def float32AutoEncoderDTypes : AutoEncoderDTypes := {
  compute := RealFloatingDType.ofDType .Float32, mean := ⟨.Float32, by decide⟩,
  variance := RealFloatingDType.ofDType .Float32, scaled := .Float32,
  scalePromotion := ⟨rfl⟩, offsetPromotion := ⟨rfl⟩ }

def AutoEncoderDTypes.ofDTypes? (compute mean variance : ExactDType) :
    Except String AutoEncoderDTypes := do
  let computeFloating ← DType.realFloating? compute.val
  let _ := computeFloating.down
  let varianceFloating ← DType.realFloating? variance.val
  let _ := varianceFloating.down
  let compute := RealFloatingDType.ofDType compute.val
  let variance := RealFloatingDType.ofDType variance.val
  match h : DType.promote? compute.val variance.val with
  | none => throw s!"Unsupported VAE variance promotion: {compute.val} * {variance.val}"
  | some scaled =>
    if h' : DType.promote? scaled mean.val = some compute.val then
      pure { compute, mean, variance, scaled, scalePromotion := ⟨h⟩, offsetPromotion := ⟨h'⟩ }
    else
      throw s!"VAE statistics {variance.val}/{mean.val} do not preserve decoder precision {compute.val}"

end torch.vae
