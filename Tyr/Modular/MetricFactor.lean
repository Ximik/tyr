import Tyr.TensorStruct

namespace Tyr.Modular

open torch

/-- Euclidean-valued factor for a positive semidefinite pullback metric. -/
structure MetricFactor (rank dim : UInt64) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  matrix : Tensor ⟨#[rank, dim], dtype, device⟩

/-- Diagonal positive mass metric used to regularize a layer metric. -/
structure DiagonalMass (dim : UInt64) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  diag : Tensor ⟨#[dim], dtype, device⟩

namespace DiagonalMass

/-- Identity diagonal mass. -/
def ones (dim : UInt64) (device : Device := .CPU) : DiagonalMass dim .Float32 device :=
  { diag := Tensor.ones #[dim] false device }

/-- Constant diagonal mass. -/
def ofScalar (dim : UInt64) (value : Float) (device : Device := .CPU) : DiagonalMass dim .Float32 device :=
  { diag := Tensor.full #[dim] value false device }

/-- Dense diagonal matrix representation. -/
def toMatrix {dim : UInt64} {d : DType} {dev : Device} (dm : DiagonalMass dim d dev) :
    Tensor ⟨#[dim, dim], d, dev⟩ :=
  Tensor.diagflat dm.diag

/-- Elementwise inverse diagonal entries. -/
def invDiag {dim : UInt64} {d : DType} {dev : Device} (dm : DiagonalMass dim d dev) :
    Tensor ⟨#[dim], d, dev⟩ :=
  Tensor.Preserving.div (Tensor.onesLike dm.diag) dm.diag

/-- Dense inverse of the diagonal mass. -/
def invMatrix {dim : UInt64} {d : DType} {dev : Device} (dm : DiagonalMass dim d dev) :
    Tensor ⟨#[dim, dim], d, dev⟩ :=
  Tensor.diagflat dm.invDiag

/-- Apply the diagonal mass to a vector. -/
def apply {dim : UInt64} {d : DType} {dev : Device} (dm : DiagonalMass dim d dev)
    (x : Tensor ⟨#[dim], d, dev⟩) : Tensor ⟨#[dim], d, dev⟩ :=
  Tensor.Preserving.mul dm.diag x

/-- Apply the inverse diagonal mass to a vector. -/
def applyInv {dim : UInt64} {d : DType} {dev : Device} (dm : DiagonalMass dim d dev)
    (x : Tensor ⟨#[dim], d, dev⟩) : Tensor ⟨#[dim], d, dev⟩ :=
  Tensor.Preserving.mul dm.invDiag x

end DiagonalMass

namespace MetricFactor

/-- Identity metric factor on Euclidean output space. -/
def identity (dim : UInt64) (device : Device := .CPU) : MetricFactor dim dim .Float32 device :=
  { matrix := Tensor.eyeFor .Float32 device }

/-- Construct a factor directly from its matrix representation. -/
def ofMatrix {rank dim : UInt64} {d : DType} {dev : Device}
    (matrix : Tensor ⟨#[rank, dim], d, dev⟩) : MetricFactor rank dim d dev :=
  { matrix }

/-- Pull a metric factor back along a local Jacobian. -/
def pullback {rank outDim inDim : UInt64} {d : DType} {dev : Device}
    (L : MetricFactor rank outDim d dev)
    (A : Tensor ⟨#[outDim, inDim], d, dev⟩) : MetricFactor rank inDim d dev :=
  { matrix := Tensor.mm L.matrix A }

private def scaleColumns {rank dim : UInt64} {d : DType} {dev : Device}
    (matrix : Tensor ⟨#[rank, dim], d, dev⟩)
    (scale : Tensor ⟨#[dim], d, dev⟩) : Tensor ⟨#[rank, dim], d, dev⟩ :=
  let scaleRow : Tensor ⟨#[1, dim], d, dev⟩ :=
    Tensor.reshape scale #[1, dim] (by simp [TensorSpec.numelOfShape])
  let scaleFull : Tensor ⟨#[rank, dim], d, dev⟩ := Tensor.expand scaleRow #[rank, dim]
  Tensor.Preserving.mul matrix scaleFull

private def vectorToColumn {n : UInt64} {d : DType} {dev : Device}
    (v : Tensor ⟨#[n], d, dev⟩) : Tensor ⟨#[n, 1], d, dev⟩ :=
  Tensor.reshape v #[n, 1] (by simp [TensorSpec.numelOfShape])

private def columnToVector {n : UInt64} {d : DType} {dev : Device}
    (v : Tensor ⟨#[n, 1], d, dev⟩) : Tensor ⟨#[n], d, dev⟩ :=
  Tensor.reshape v #[n] (by simp [TensorSpec.numelOfShape])

/-- Dense Gram form `K^T K`. -/
def gram {rank dim : UInt64} {d : DType} {dev : Device} (K : MetricFactor rank dim d dev) :
    Tensor ⟨#[dim, dim], d, dev⟩ :=
  Tensor.mm (Tensor.transpose2d K.matrix) K.matrix

/-- Dense regularized layer metric `D + K^T K`. -/
def denseMetric {rank dim : UInt64} {d : DType} {dev : Device}
    (D : DiagonalMass dim d dev)
    (K : MetricFactor rank dim d dev) : Tensor ⟨#[dim, dim], d, dev⟩ :=
  Tensor.Preserving.add D.toMatrix K.gram

/-- Apply `K` to a vector. -/
def apply {rank dim : UInt64} {d : DType} {dev : Device} (K : MetricFactor rank dim d dev)
    (x : Tensor ⟨#[dim], d, dev⟩) : Tensor ⟨#[rank], d, dev⟩ :=
  columnToVector (Tensor.mm K.matrix (vectorToColumn x))

/-- Apply `K^T` to a vector. -/
def applyTranspose {rank dim : UInt64} {d : DType} {dev : Device}
    (K : MetricFactor rank dim d dev) (x : Tensor ⟨#[rank], d, dev⟩) : Tensor ⟨#[dim], d, dev⟩ :=
  columnToVector (Tensor.mm (Tensor.transpose2d K.matrix) (vectorToColumn x))

/-- Apply the regularized metric `D + K^T K` to a vector. -/
def applyRegularized {rank dim : UInt64} {d : DType} {dev : Device}
    (D : DiagonalMass dim d dev)
    (K : MetricFactor rank dim d dev)
    (x : Tensor ⟨#[dim], d, dev⟩) : Tensor ⟨#[dim], d, dev⟩ :=
  Tensor.Preserving.add (D.apply x) (K.applyTranspose (K.apply x))

/-- Woodbury inner matrix `I + K D^{-1} K^T`. -/
def woodburyInner {rank dim : UInt64} {d : DType} {dev : Device}
    (D : DiagonalMass dim d dev)
    (K : MetricFactor rank dim d dev) : Tensor ⟨#[rank, rank], d, dev⟩ :=
  let kd := scaleColumns K.matrix D.invDiag
  Tensor.Preserving.add (Tensor.eyeFor d dev) (Tensor.mm kd (Tensor.transpose2d K.matrix))

/-- Solve `(D + K^T K) x = g` using the Woodbury identity. -/
def solveWoodbury {rank dim : UInt64} {d : DType} {dev : Device}
    (D : DiagonalMass dim d dev)
    (K : MetricFactor rank dim d dev)
    (g : Tensor ⟨#[dim], d, dev⟩) : Tensor ⟨#[dim], d, dev⟩ :=
  let inner := woodburyInner D K
  let innerInv := Tensor.inv inner
  let base := D.applyInv g
  let correctionArg := K.apply base
  let correctionInner := columnToVector (Tensor.mm innerInv (vectorToColumn correctionArg))
  let correction := D.applyInv (K.applyTranspose correctionInner)
  Tensor.Preserving.sub base correction

end MetricFactor

end Tyr.Modular
