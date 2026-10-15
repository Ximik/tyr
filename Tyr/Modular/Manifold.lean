import Tyr.Modular.Norm
import Tyr.Modular.Budget
import Tyr.Modular.Compose
import Tyr.Manifolds.Optimizer
import Tyr.Optim.DualOptimizer

namespace Tyr.Modular

open torch

variable {device : Device}
open Tyr.AD

/-- Ambient matrix type for manifold carriers: Float32 weights with
    explicit placement, matching the carrier instances' manifold types. -/
abbrev AmbientMatrix (m n : UInt64) (device : Device := .CPU) : Type :=
  Tensor { shape := #[m, n], dtype := .Float32, device }

/-- Ambient vector type for manifold carriers. -/
abbrev AmbientVector (n : UInt64) (device : Device := .CPU) : Type :=
  Tensor { shape := #[n], dtype := .Float32, device }

/-- Shape-legality constraint for manifold-carrier matrices: at least as many
    rows as columns (the square case included). A reducible abbrev so `decide`
    discharges it at concrete call sites. -/
abbrev MatrixShapeOK (m n : UInt64) : Prop := n.toNat ≤ m.toNat

/-- Matrix-manifold carrier interface for manifold-native linear layers.

    Shape-legality witnesses travel as plain proof arguments on `project`
    (discharged by `decide` at concrete call sites) or are reused from an
    existing manifold value (`reAmbient`/`projectLike`), so no typeclass
    synthesis of propositions is needed. -/
class MatrixManifoldCarrier (M : UInt64 → UInt64 → Type)
    (device : outParam Device) where
  /-- Project an ambient matrix to the manifold, given a legality witness. -/
  project : {m n : UInt64} → MatrixShapeOK m n → AmbientMatrix (device := device) m n → M m n
  /-- Convert manifold value to its ambient matrix representation. -/
  toMatrix : {m n : UInt64} → M m n → AmbientMatrix (device := device) m n
  /-- Replace the ambient contents of an existing manifold value, keeping its
      shape witnesses. Does not re-project; the caller asserts validity. -/
  reAmbient : {m n : UInt64} → M m n → AmbientMatrix (device := device) m n → M m n
  /-- Re-project onto the manifold, reusing an existing value's witnesses. -/
  projectLike : {m n : UInt64} → M m n → AmbientMatrix (device := device) m n → M m n
  /-- Take one geometry-aware step and retract on-manifold. -/
  dualMapStep : {m n : UInt64} → M m n → AmbientMatrix (device := device) m n → Float → M m n

instance : MatrixManifoldCarrier (fun m n => Stiefel m n .Float32 device) device where
  project := fun h x => Stiefel.project _ _ x h
  toMatrix := fun x => x.matrix
  reAmbient := fun x mat => ⟨mat, x.tall⟩
  projectLike := fun x mat => Stiefel.project _ _ mat x.tall
  dualMapStep := fun x g lr =>
    DualMapGeometry.dualMapStep x (StiefelTangent.project x g) lr

instance : MatrixManifoldCarrier (fun m n => Grassmann m n .Float32 device) device where
  project := fun h x => Grassmann.project _ _ x h
  toMatrix := fun x => x.matrix
  reAmbient := fun x mat => ⟨mat, x.tall⟩
  projectLike := fun x mat => Grassmann.project _ _ mat x.tall
  dualMapStep := fun x g lr =>
    DualMapGeometry.dualMapStep x (GrassmannTangent.project x g) lr

/-- Orthogonal manifold viewed in matrix-family form with square-shape legality. -/
abbrev OrthogonalMatrix (_m n : UInt64) (device : Device := .CPU) := Orthogonal n .Float32 device

/-- View an ambient matrix as square, validating the runtime shape. Only
    reachable through `MatrixShapeOK`-witnessed paths; the square requirement
    of the orthogonal family itself is checked here at runtime. -/
private def asSquare {m n : UInt64} (mat : AmbientMatrix (device := device) m n) : AmbientMatrix (device := device) n n :=
  Tensor.reshapeChecked mat #[n, n]

instance : MatrixManifoldCarrier (fun m n => OrthogonalMatrix m n device) device where
  project := fun {m n} _h mat =>
    if h : m = n then
      Orthogonal.project n (h ▸ mat)
    else
      Orthogonal.project n (asSquare mat)  -- runtime-validated square conversion
  toMatrix := fun x => Tensor.reshapeChecked x.matrix #[_, _]
  reAmbient := fun _ mat => ⟨asSquare mat⟩
  projectLike := fun _ mat => Orthogonal.project _ (asSquare mat)
  dualMapStep := fun x g lr =>
    DualMapGeometry.dualMapStep x (OrthogonalTangent.fromAmbient x (asSquare g)) lr

/-- Vector-manifold carrier interface where ambient dimension may differ from intrinsic one. -/
class VectorManifoldCarrier (V : UInt64 → Type)
    (device : outParam Device) where
  /-- Ambient coordinate dimension for intrinsic manifold dimension `n`. -/
  ambientDim : UInt64 → UInt64
  /-- Project ambient vector coordinates to the manifold. -/
  project : (n : UInt64) → AmbientVector (device := device) (ambientDim n) → V n
  /-- Convert manifold value to ambient coordinates. -/
  toVector : {n : UInt64} → V n → AmbientVector (device := device) (ambientDim n)
  /-- Wrap already-valid ambient coordinates without re-projecting. -/
  fromAmbientUnchecked : {n : UInt64} → AmbientVector (device := device) (ambientDim n) → V n
  /-- Take one geometry-aware step on the manifold. -/
  dualMapStep : {n : UInt64} → V n → AmbientVector (device := device) (ambientDim n) → Float → V n

instance : VectorManifoldCarrier (fun n => Hyperbolic n .Float32 device) device where
  ambientDim n := n + 1
  project := Hyperbolic.project
  toVector := fun x => x.coords
  fromAmbientUnchecked := fun x => ⟨x⟩
  dualMapStep := fun x g lr =>
    DualMapGeometry.dualMapStep x (HyperbolicTangent.project x g) lr

/--
Linear layer whose weight is stored on a matrix manifold.

The weight is natively represented by `M out_dim in_dim`, so retractions are
part of the layer-level update path rather than an external afterthought.
-/
structure ManifoldLinear (M : UInt64 → UInt64 → Type) [MatrixManifoldCarrier M device]
    (in_dim out_dim : UInt64) where
  weight : M out_dim in_dim
  bias : Option (AmbientVector (device := device) out_dim) := none

namespace ManifoldLinear

private def makeLeafWeight [MatrixManifoldCarrier M device] {in_dim out_dim : UInt64}
    (w : M out_dim in_dim) : IO (M out_dim in_dim) := do
  let leaf ← Tensor.parameter (MatrixManifoldCarrier.toMatrix (device := device) w)
  pure (MatrixManifoldCarrier.reAmbient (device := device) w leaf)

/-- Initialize manifold-linear weights by projecting a random matrix. -/
def init [MatrixManifoldCarrier M device] (in_dim out_dim : UInt64)
    (withBias : Bool := true)
    (h : MatrixShapeOK out_dim in_dim := by decide)
    : IO (ManifoldLinear (device := device) M in_dim out_dim) := do
  let w0 ← Tensor.randn #[out_dim, in_dim] false device
  let weight ← makeLeafWeight (MatrixManifoldCarrier.project (device := device) h w0)
  let bias ← if withBias then do
      pure (some (← Tensor.parameter (Tensor.zeros #[out_dim] (device := device))))
    else
      pure none
  pure { weight, bias }

/-- Forward pass for `[batch, in_dim] -> [batch, out_dim]`. -/
def forward2d [MatrixManifoldCarrier M device] {in_dim out_dim batch : UInt64}
    (lin : ManifoldLinear (device := device) M in_dim out_dim)
    (x : AmbientMatrix (device := device) batch in_dim) : AmbientMatrix (device := device) batch out_dim :=
  let y := Tensor.linear x (MatrixManifoldCarrier.toMatrix (device := device) lin.weight)
  match lin.bias with
  | some b => Add.add y (Tensor.expand b #[batch, out_dim])
  | none => y

/-- Apply a manifold-native step to weights and an Euclidean step to bias. -/
def applyDualMapUpdate [MatrixManifoldCarrier M device] {in_dim out_dim : UInt64}
    (lin : ManifoldLinear (device := device) M in_dim out_dim)
    (weightGrad : AmbientMatrix (device := device) out_dim in_dim)
    (biasGrad? : Option (AmbientVector (device := device) out_dim))
    (lr : Float) : IO (ManifoldLinear (device := device) M in_dim out_dim) := do
  let weight' ← makeLeafWeight (MatrixManifoldCarrier.dualMapStep (device := device) lin.weight weightGrad lr)
  let bias' ←
    match lin.bias, biasGrad? with
    | some b, some g => do
      pure (some (← Tensor.parameter (Sub.sub b (Tensor.Preserving.mulScalar g lr))))
    | some b, none => pure (some b)
    | none, _ => pure none
  pure { lin with weight := weight', bias := bias' }

end ManifoldLinear

/-- Single vector parameter living natively on a vector manifold. -/
structure ManifoldVectorParam (V : UInt64 → Type) [VectorManifoldCarrier V device] (n : UInt64) where
  value : V n

namespace ManifoldVectorParam

/-- Ensure manifold-vector value is represented by a trainable leaf tensor. -/
private def makeLeafValue [VectorManifoldCarrier V device] {n : UInt64}
    (v : V n) : IO (V n) := do
  let leaf ← Tensor.parameter (VectorManifoldCarrier.toVector (device := device) v)
  pure (VectorManifoldCarrier.fromAmbientUnchecked (device := device) leaf)

/-- Initialize from random ambient coordinates projected to manifold. -/
def init [VectorManifoldCarrier V device] (n : UInt64) : IO (ManifoldVectorParam (device := device) V n) := do
  let d := (VectorManifoldCarrier.ambientDim (device := device) (V := V) n)
  let vRaw ← Tensor.randn #[d] false device
  let v ← makeLeafValue (VectorManifoldCarrier.project (device := device) n vRaw)
  pure { value := v }

/-- Apply one dual-map manifold update for this vector parameter. -/
def applyDualMapUpdate [VectorManifoldCarrier V device] {n : UInt64}
    (p : ManifoldVectorParam (device := device) V n)
    (grad : AmbientVector (device := device) (VectorManifoldCarrier.ambientDim (device := device) (V := V) n))
    (lr : Float) : IO (ManifoldVectorParam (device := device) V n) := do
  let v' := VectorManifoldCarrier.dualMapStep (device := device) p.value grad lr
  pure { value := ← makeLeafValue v' }

end ManifoldVectorParam

instance [MatrixManifoldCarrier M device] (in_dim out_dim : UInt64) :
    TensorStruct (ManifoldLinear (device := device) M in_dim out_dim) where
  map f lin := {
    weight := MatrixManifoldCarrier.reAmbient (device := device) lin.weight
      (TensorStruct.map f (MatrixManifoldCarrier.toMatrix (device := device) lin.weight))
    bias := TensorStruct.map f lin.bias
  }
  mapM f lin := do
    let w ← TensorStruct.mapM f (MatrixManifoldCarrier.toMatrix (device := device) lin.weight)
    let bias ← TensorStruct.mapM f lin.bias
    pure { weight := MatrixManifoldCarrier.reAmbient (device := device) lin.weight w, bias }
  zipWith f a b := {
    weight := MatrixManifoldCarrier.reAmbient (device := device) a.weight
      (TensorStruct.zipWith f (MatrixManifoldCarrier.toMatrix (device := device) a.weight)
        (MatrixManifoldCarrier.toMatrix (device := device) b.weight))
    bias := TensorStruct.zipWith f a.bias b.bias
  }
  fold f init lin :=
    TensorStruct.fold f
      (TensorStruct.fold f init (MatrixManifoldCarrier.toMatrix (device := device) lin.weight)) lin.bias

instance [VectorManifoldCarrier V device] (n : UInt64) : TensorStruct (ManifoldVectorParam (device := device) V n) where
  map f p := {
    value := VectorManifoldCarrier.fromAmbientUnchecked (device := device)
      (TensorStruct.map f (VectorManifoldCarrier.toVector (device := device) p.value))
  }
  mapM f p := do
    let v ← TensorStruct.mapM f (VectorManifoldCarrier.toVector (device := device) p.value)
    pure { value := VectorManifoldCarrier.fromAmbientUnchecked (device := device) v }
  zipWith f a b := {
    value := VectorManifoldCarrier.fromAmbientUnchecked (device := device)
      (TensorStruct.zipWith f (VectorManifoldCarrier.toVector (device := device) a.value)
        (VectorManifoldCarrier.toVector (device := device) b.value))
  }
  fold f init p := TensorStruct.fold f init (VectorManifoldCarrier.toVector (device := device) p.value)

instance [MatrixManifoldCarrier M device] (in_dim out_dim : UInt64) :
    NormedModule (ManifoldLinear (device := device) M in_dim out_dim) where
  norm lin :=
    let weightNorm := Tensor.spectralNorm (MatrixManifoldCarrier.toMatrix (device := device) lin.weight)
    match lin.bias with
    | some b => floatMax weightNorm (Tensor.l2Norm b)
    | none => weightNorm
  dualNorm lin :=
    let weightDual := Tensor.nuclearNorm (MatrixManifoldCarrier.toMatrix (device := device) lin.weight)
    match lin.bias with
    | some b => weightDual + Tensor.l2Norm b
    | none => weightDual
  nu lin := Tensor.spectralNorm (MatrixManifoldCarrier.toMatrix (device := device) lin.weight)
  mu _ := 1.0
  normalize lin :=
    let w := MatrixManifoldCarrier.toMatrix (device := device) lin.weight
    let sNorm := Tensor.spectralNorm w
    let w' := if sNorm == 0.0 then w else Tensor.Preserving.mulScalar w (1.0 / sNorm)
    let weight := MatrixManifoldCarrier.projectLike (device := device) lin.weight w'
    let bias :=
      match lin.bias with
      | some b =>
        let bNorm := Tensor.l2Norm b
        if bNorm == 0.0 then some b else some (Tensor.Preserving.mulScalar b (1.0 / bNorm))
      | none => none
    { lin with weight, bias }
  normalizeDual lin :=
    let w := MatrixManifoldCarrier.toMatrix (device := device) lin.weight
    let fnorm := Tensor.frobeniusNorm w
    let w' := if fnorm == 0.0 then w else Tensor.Preserving.mulScalar w (1.0 / fnorm)
    let weight := MatrixManifoldCarrier.projectLike (device := device) lin.weight w'
    let bias :=
      match lin.bias with
      | some b =>
        let bNorm := Tensor.l2Norm b
        if bNorm == 0.0 then some b else some (Tensor.Preserving.mulScalar b (1.0 / bNorm))
      | none => none
    { lin with weight, bias }

instance [VectorManifoldCarrier V device] (n : UInt64) :
    NormedModule (ManifoldVectorParam (device := device) V n) where
  norm p := Tensor.l2Norm (VectorManifoldCarrier.toVector (device := device) p.value)
  dualNorm p := Tensor.l2Norm (VectorManifoldCarrier.toVector (device := device) p.value)
  nu _ := 1.0
  mu _ := 1.0
  normalize p :=
    let n' := n
    let v := VectorManifoldCarrier.toVector (device := device) p.value
    let d := Tensor.l2Norm v
    let v' := if d == 0.0 then v else Tensor.Preserving.mulScalar v (1.0 / d)
    { value := VectorManifoldCarrier.project (device := device) n' v' }
  normalizeDual p :=
    let n' := n
    let v := VectorManifoldCarrier.toVector (device := device) p.value
    let d := Tensor.l2Norm v
    let v' := if d == 0.0 then v else Tensor.Preserving.mulScalar v (1.0 / d)
    { value := VectorManifoldCarrier.project (device := device) n' v' }

/-- Convenience alias: manifold-linear with Stiefel-constrained weights. -/
abbrev StiefelLinear (in_dim out_dim : UInt64) (device : Device := .CPU) :=
  ManifoldLinear (device := device) (fun m n => Stiefel m n .Float32 device) in_dim out_dim

/-- Convenience alias: manifold-linear with Grassmann-constrained weights. -/
abbrev GrassmannLinear (in_dim out_dim : UInt64) (device : Device := .CPU) :=
  ManifoldLinear (device := device) (fun m n => Grassmann m n .Float32 device) in_dim out_dim

/-- Convenience alias: manifold-linear with orthogonal square weights. -/
abbrev OrthogonalLinear (dim : UInt64) (device : Device := .CPU) :=
  ManifoldLinear (device := device) (fun m n => OrthogonalMatrix m n device) dim dim

/-- Convenience alias: single hyperbolic vector parameter. -/
abbrev HyperbolicVector (n : UInt64) (device : Device := .CPU) :=
  ManifoldVectorParam (device := device) (fun n => Hyperbolic n .Float32 device) n

/-- Cotangent payload for `ManifoldLinear (device := device)`: weight cotangent plus optional bias cotangent. -/
abbrev ManifoldLinearCotangent (in_dim out_dim : UInt64) :=
  AmbientMatrix (device := device) out_dim in_dim × Option (AmbientVector (device := device) out_dim)

/--
Composable manifold-update interface.

This decouples "what manifold-native step to run" from "how modules are nested",
so product/sequential structures can be updated without dedicated manifold types.
-/
class ManifoldUpdatable (P : Type) where
  Cotangent : P → Type
  applyUpdate : (params : P) → Cotangent params → Float → IO P

namespace ManifoldUpdatable

/-- Apply one manifold-native update step. -/
def step [ManifoldUpdatable P] (params : P) (grad : ManifoldUpdatable.Cotangent params) (lr : Float) : IO P :=
  ManifoldUpdatable.applyUpdate params grad lr

instance [MatrixManifoldCarrier M device] (in_dim out_dim : UInt64) :
    ManifoldUpdatable (ManifoldLinear (device := device) M in_dim out_dim) where
  Cotangent _ := ManifoldLinearCotangent (device := device) in_dim out_dim
  applyUpdate lin g lr := ManifoldLinear.applyDualMapUpdate lin g.1 g.2 lr

instance [VectorManifoldCarrier V device] (n : UInt64) :
    ManifoldUpdatable (ManifoldVectorParam (device := device) V n) where
  Cotangent _ := AmbientVector (device := device) (VectorManifoldCarrier.ambientDim (device := device) (V := V) n)
  applyUpdate p g lr := ManifoldVectorParam.applyDualMapUpdate p g lr

instance [ManifoldUpdatable A] [ManifoldUpdatable B] : ManifoldUpdatable (A × B) where
  Cotangent p := ManifoldUpdatable.Cotangent p.1 × ManifoldUpdatable.Cotangent p.2
  applyUpdate p g lr := do
    let first ← ManifoldUpdatable.applyUpdate p.1 g.1 lr
    let second ← ManifoldUpdatable.applyUpdate p.2 g.2 lr
    pure (first, second)

instance [ManifoldUpdatable M₁] [ManifoldUpdatable M₂] :
    ManifoldUpdatable (Sequential M₁ M₂) where
  Cotangent net := ManifoldUpdatable.Cotangent net.first × ManifoldUpdatable.Cotangent net.second
  applyUpdate net g lr := do
    let first ← ManifoldUpdatable.applyUpdate net.first g.1 lr
    let second ← ManifoldUpdatable.applyUpdate net.second g.2 lr
    pure { first, second }

end ManifoldUpdatable

private def meanOrOne (xs : Array Float) : Float :=
  if xs.isEmpty then 1.0 else xs.foldl (fun acc x => acc + x) 0.0 / xs.size.toFloat

/--
Compile a matrix-group budget multiplier from modular sensitivities of
manifold-native modules and inject it into `DualOptimizer.Config`.
-/
def applyMatrixBudgetFromModules [NormedModule M]
    (optCfg : torch.Optim.DualOptimizer.Config)
    (budgetCfg : BudgetConfig := {})
    (modules : Array M)
    : torch.Optim.DualOptimizer.Config :=
  let scales := sequentialDownstreamScales budgetCfg modules
  let matrixMul := meanOrOne scales
  { optCfg with
    useModularBudget := true
    budget := { optCfg.budget with matrix := matrixMul } }

end Tyr.Modular
