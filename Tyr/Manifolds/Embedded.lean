import Tyr.Manifolds.Basic
import Tyr.Manifolds.Stiefel
import Tyr.Manifolds.Orthogonal
import Tyr.Manifolds.Grassmann
import Tyr.Manifolds.Hyperbolic

namespace Tyr.AD

open torch
open DifferentiableManifold

variable {d : DType} {dev : Device}

/--
Manifolds that admit a distinguished ambient representation together with
projection from ambient directions to the tangent space.
-/
class EmbeddedManifold (M : Type) [DifferentiableManifold M] where
  Ambient : Type
  toAmbient : M → Ambient
  projectAmbientTangent : {x : M} → Ambient → Tangent x

namespace EmbeddedManifold

/-- Project an ambient direction to the tangent space and retract. -/
def retractAmbientStep [DifferentiableManifold M] [EmbeddedManifold M]
    (x : M) (ambientDirection : EmbeddedManifold.Ambient (M := M)) (lr : Float) : M :=
  let tangent := EmbeddedManifold.projectAmbientTangent (x := x) ambientDirection
  let step := scaleTangent (-lr) tangent
  retract x step

end EmbeddedManifold

instance tensorEmbedded {σ : StaticSpec} [DType.ConstantSupported σ.dtype σ.device] : EmbeddedManifold (Tensor σ) where
  Ambient := Tensor σ
  toAmbient := id
  projectAmbientTangent := id

instance stiefelEmbedded (n p : UInt64) : EmbeddedManifold (Stiefel n p d dev) where
  Ambient := Tensor ⟨#[n, p], d, dev⟩
  toAmbient := fun x => x.matrix
  projectAmbientTangent := fun {x} Z => StiefelTangent.project x Z

instance orthogonalEmbedded [DType.ConstantSupported d dev] (n : UInt64) : EmbeddedManifold (Orthogonal n d dev) where
  Ambient := Tensor ⟨#[n, n], d, dev⟩
  toAmbient := fun x => x.matrix
  projectAmbientTangent := fun {x} Z => OrthogonalTangent.fromAmbient x Z

instance grassmannEmbedded (n p : UInt64) : EmbeddedManifold (Grassmann n p d dev) where
  Ambient := Tensor ⟨#[n, p], d, dev⟩
  toAmbient := fun x => x.matrix
  projectAmbientTangent := fun {x} Z => GrassmannTangent.project x Z

instance hyperbolicEmbedded [DType.ConstantSupported d dev] (n : UInt64) : EmbeddedManifold (Hyperbolic n d dev) where
  Ambient := Tensor ⟨#[n + 1], d, dev⟩
  toAmbient := fun x => x.coords
  projectAmbientTangent := fun {x} Z => HyperbolicTangent.project x Z

end Tyr.AD
