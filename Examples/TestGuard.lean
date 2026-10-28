import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Widget  -- This enables deriving ToModuleDisplay
import Tyr.Module.Linear


open torch

-- Example: Visualize a Linear layer's parameters
def exampleModule : IO ModuleDisplayProps := do
  let weight ← Tensor.randn #[64, 128]
  let bias ← Tensor.randn #[64]
  let weightNode := ModuleNode.tensor (← tensorToPropsNamed weight "weight" #["out", "in"])
  let biasNode := ModuleNode.tensor (← tensorToPropsNamed bias "bias" #["out"])

  return {
    root := ModuleNode.group "Linear" #[
      ("weight", weightNode),
      ("bias", biasNode)
    ]
  }

-- Nested example: A small MLP
def mlpModule : IO ModuleDisplayProps := do
  let w1 ← Tensor.randn #[128, 64]
  let b1 ← Tensor.randn #[128]
  let w2 ← Tensor.randn #[10, 128]
  let b2 ← Tensor.randn #[10]
  let layer1 := ModuleNode.group "layer1" #[
    ("weight", ModuleNode.tensor (← tensorToPropsNamed w1 "weight" #["out", "in"])),
    ("bias", ModuleNode.tensor (← tensorToPropsNamed b1 "bias" #["out"]))
  ]
  let layer2 := ModuleNode.group "layer2" #[
    ("weight", ModuleNode.tensor (← tensorToPropsNamed w2 "weight" #["out", "in"])),
    ("bias", ModuleNode.tensor (← tensorToPropsNamed b2 "bias" #["out"]))
    ]
  return {
    root := ModuleNode.group "MLP" #[
      ("fc1", layer1),
      ("fc2", layer2)
    ]
  }

#module mlpModule



#eval Tensor.randn #[3, 4]
#tensor Tensor.randn #[10,49]
#tensor Tensor.randn #[20,10]

#tensor Tensor.randn #[20,100,4]


 /-- Example: Custom structure with deriving -/
structure MLP (in_dim hidden out_dim : Nat) where
  fc1 : Linear in_dim hidden
  fc2 : Linear hidden out_dim
  deriving ToModuleDisplay

def exampleMLP : IO ModuleDisplayProps := do
  let fc1 ← Linear.init 64 256
  let fc2 ← Linear.init 256 10
  let mlp : MLP 64 256 10 := { fc1, fc2 }
  toModuleDisplayProps mlp "MLP(64 → 256 → 10)"

#module exampleMLP
