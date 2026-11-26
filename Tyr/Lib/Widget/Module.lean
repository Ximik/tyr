/-
  Tyr/Lib/Widget/Module.lean

  `ToModuleDisplay` instances for the layers in `Tyr.Lib.Module`.
-/
import Tyr.Lib.Widget
import Tyr.Lib.Module

namespace torch

deriving instance ToModuleDisplay for Linear, LayerNorm, RMSNorm

end torch
