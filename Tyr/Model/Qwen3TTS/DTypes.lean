import Tyr.Typed.Tensor

namespace torch.qwen3tts

/-- Default precision for newly initialized Qwen3-TTS components. -/
def float32DType : RealFloatingDType := RealFloatingDType.ofDType .Float32

/-- Dtypes shared by connected model components. The speaker encoder may use
its own checkpoint dtype; embeddings, both decoders, heads and caches agree. -/
structure DTypeLayout where
  talker : RealFloatingDType := float32DType
  speaker : RealFloatingDType := float32DType

end torch.qwen3tts
