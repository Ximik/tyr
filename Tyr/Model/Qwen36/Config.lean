/-
  Tyr/Model/Qwen36/Config.lean

  Qwen3.6 aliases/helpers layered on top of the shared Qwen3.5-MoE
  implementation used by Tyr.
-/
import Tyr.Model.Qwen35.VLConfig

namespace torch.qwen36

abbrev LayerType := qwen35.LayerType
abbrev Config := qwen35.Config
abbrev VisionConfig := qwen35.VisionConfig
abbrev VLConfig := qwen35.VLConfig

namespace Config

/-- Default Qwen3.6-35B-A3B text config from the public HF checkpoint. -/
def qwen36_35B_A3B : Config := qwen35.Config.qwen36_35B_A3B

def normalize (cfg : Config) : Config := qwen35.Config.normalize cfg

def normalizedLayerTypes (cfg : Config) : Array LayerType :=
  qwen35.Config.normalizedLayerTypes cfg

def numHeadsPerKVGroup (cfg : Config) : Nat :=
  qwen35.Config.numHeadsPerKVGroup cfg

def linearKeyDim (cfg : Config) : Nat :=
  qwen35.Config.linearKeyDim cfg

def linearValueDim (cfg : Config) : Nat :=
  qwen35.Config.linearValueDim cfg

def linearConvDim (cfg : Config) : Nat :=
  qwen35.Config.linearConvDim cfg

def linearKVRepeat (cfg : Config) : Nat :=
  qwen35.Config.linearKVRepeat cfg

def isMoE (cfg : Config) : Bool :=
  qwen35.Config.isMoE cfg

def rotaryDim (cfg : Config) : Nat :=
  qwen35.Config.rotaryDim cfg

def rotaryHalfDim (cfg : Config) : Nat :=
  qwen35.Config.rotaryHalfDim cfg

end Config

namespace VisionConfig

/-- Default Qwen3.6-35B-A3B vision config from the public HF checkpoint. -/
def qwen36_35B_A3B : VisionConfig := qwen35.VisionConfig.qwen36_35B_A3B

def patchDim (cfg : VisionConfig) : Nat := qwen35.VisionConfig.patchDim cfg

def mergeUnit (cfg : VisionConfig) : Nat := qwen35.VisionConfig.mergeUnit cfg

def headDim (cfg : VisionConfig) : Nat := qwen35.VisionConfig.headDim cfg

def mergedTokenCount (cfg : VisionConfig) (nPatches : Nat) : Nat :=
  qwen35.VisionConfig.mergedTokenCount cfg nPatches

end VisionConfig

namespace VLConfig

/-- Default Qwen3.6-35B-A3B multimodal config from the public HF checkpoint. -/
def qwen36_35B_A3B : VLConfig := qwen35.VLConfig.qwen36_35B_A3B

def normalize (cfg : VLConfig) : VLConfig := qwen35.VLConfig.normalize cfg

end VLConfig

end torch.qwen36
