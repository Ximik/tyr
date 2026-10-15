import Tyr.Typed.Promotion
import Tyr.Typed.Tensor

namespace torch.vae

/-- Stored normalization statistics may have independent precision. Their
mathematical promotion must still produce the decoder's activation dtype. -/
structure AutoEncoderDTypes where
  compute : ExactDType
  mean : ExactDType
  variance : ExactDType
  scaled : DType
  scalePromotion : DType.Promotes compute.val variance.val scaled
  offsetPromotion : DType.Promotes scaled mean.val compute.val

def float32AutoEncoderDTypes : AutoEncoderDTypes := {
  compute := ⟨.Float32, by decide⟩, mean := ⟨.Float32, by decide⟩,
  variance := ⟨.Float32, by decide⟩, scaled := .Float32,
  scalePromotion := ⟨rfl⟩, offsetPromotion := ⟨rfl⟩ }

def AutoEncoderDTypes.ofDTypes? (compute mean variance : ExactDType) :
    Except String AutoEncoderDTypes := do
  match h : DType.promote? compute.val variance.val with
  | none => throw s!"Unsupported VAE variance promotion: {compute.val} * {variance.val}"
  | some scaled =>
    if h' : DType.promote? scaled mean.val = some compute.val then
      pure { compute, mean, variance, scaled, scalePromotion := ⟨h⟩, offsetPromotion := ⟨h'⟩ }
    else
      throw s!"VAE statistics {variance.val}/{mean.val} do not preserve decoder precision {compute.val}"

end torch.vae
