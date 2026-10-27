import Tyr

namespace Tests.ADJaxprLikeElabFixture

open torch
open Lean.IR
open Tyr.AD.Frontend
open Tyr.AD.JaxprLike

def directPadFrontend : FrontendRegistration := {
  jaxpr := {
    invars := #[
      { id := 1, ty := Lean.IR.IRType.object, metaInfo := { shape := some #[2] } },
      { id := 2, ty := Lean.IR.IRType.object, metaInfo := { shape := some #[1] } }
    ]
    eqns := #[
      {
        op := Tyr.AD.JaxprLike.padAliasOpName
        invars := #[
          { id := 1, ty := Lean.IR.IRType.object, metaInfo := { shape := some #[2] } },
          { id := 2, ty := Lean.IR.IRType.object, metaInfo := { shape := some #[1] } }
        ]
        outvars := #[
          { id := 3, ty := Lean.IR.IRType.object, metaInfo := { shape := some #[6] } }
        ]
        params := #[
          Tyr.AD.JaxprLike.OpParam.mkNats .padLow #[1],
          Tyr.AD.JaxprLike.OpParam.mkNats .padHigh #[2],
          Tyr.AD.JaxprLike.OpParam.mkNats .padInterior #[1],
          Tyr.AD.JaxprLike.OpParam.mkName .sourceOp `Graphax.pad_p
        ]
      }
    ]
    outvars := #[
      { id := 3, ty := Lean.IR.IRType.object, metaInfo := { shape := some #[6] } }
    ]
  }
}

def directPadStub (_x _padv : Nat) : Nat := 0

attribute [ad_frontend Tests.ADJaxprLikeElabFixture.directPadFrontend] directPadStub

structure StructuredPadInput where
  base : Tensor { shape := #[2], dtype := .Float32, device := .CPU }
  padv : Tensor { shape := #[1], dtype := .Float32, device := .CPU }
  tag : Static String
  deriving TensorStruct, ToTensorStructSchema, TensorStructFlatten

structure StructuredPadOutput where
  y : Tensor { shape := #[6], dtype := .Float32, device := .CPU }
  label : Static String
  deriving TensorStruct, ToTensorStructSchema, TensorStructFlatten

def sampleStructuredPadInput : StructuredPadInput :=
  { base := Tensor.zeros #[2]
    padv := Tensor.zeros #[1]
    tag := "input" }

def sampleStructuredPadOutput : StructuredPadOutput :=
  { y := Tensor.zeros #[6]
    label := "output" }

def structuredPadFrontendSig : FrontendADSignature := {
  inputs := #[
    FrontendBoundary.ofValue sampleStructuredPadInput .diffOnly
  ]
  outputs := #[
    FrontendBoundary.ofValue sampleStructuredPadOutput .diffOnly
  ]
}

def structuredPadFrontendJaxpr : LeanJaxpr := {
  invars := #[
    { id := 10, ty := Lean.IR.IRType.object, metaInfo := { participation := .diff, dtype := some "Float32", shape := some #[2] } },
    { id := 11, ty := Lean.IR.IRType.object, metaInfo := { participation := .diff, dtype := some "Float32", shape := some #[1] } }
  ]
  eqns := #[
    {
      op := Tyr.AD.JaxprLike.padAliasOpName
      invars := #[
        { id := 10, ty := Lean.IR.IRType.object, metaInfo := { participation := .diff, dtype := some "Float32", shape := some #[2] } },
        { id := 11, ty := Lean.IR.IRType.object, metaInfo := { participation := .diff, dtype := some "Float32", shape := some #[1] } }
      ]
      outvars := #[
        { id := 12, ty := Lean.IR.IRType.object, metaInfo := { participation := .diff, dtype := some "Float32", shape := some #[6] } }
      ]
      params := #[
        Tyr.AD.JaxprLike.OpParam.mkNats .padLow #[1],
        Tyr.AD.JaxprLike.OpParam.mkNats .padHigh #[2],
        Tyr.AD.JaxprLike.OpParam.mkNats .padInterior #[1],
        Tyr.AD.JaxprLike.OpParam.mkName .sourceOp `Graphax.pad_p
      ]
    }
  ]
  outvars := #[
    { id := 12, ty := Lean.IR.IRType.object, metaInfo := { participation := .diff, dtype := some "Float32", shape := some #[6] } }
  ]
}

def structuredPadFrontend : FrontendRegistration := {
  jaxpr := structuredPadFrontendJaxpr
  signature := some structuredPadFrontendSig
}

def directStructuredPadStub (_inp : StructuredPadInput) : StructuredPadOutput :=
  { y := Tensor.zeros #[6]
    label := "stub" }

attribute [ad_frontend Tests.ADJaxprLikeElabFixture.structuredPadFrontend] directStructuredPadStub

structure RuntimePadParams where
  padv : Tensor { shape := #[1], dtype := .Float32, device := .CPU }
  label : Static String
  deriving TensorStruct, ToTensorStructSchema, TensorStructFlatten

structure RuntimePadInput where
  base : Tensor { shape := #[2], dtype := .Float32, device := .CPU }
  tag : Static String
  deriving TensorStruct, ToTensorStructSchema, TensorStructFlatten

structure RuntimePadOutput where
  loss : Tensor { shape := #[], dtype := .Float32, device := .CPU }
  label : Static String
  deriving TensorStruct, ToTensorStructSchema, TensorStructFlatten

def sampleRuntimePadParams : RuntimePadParams :=
  { padv := Tensor.full #[1] 2.0
    label := "params" }

def sampleRuntimePadInput : RuntimePadInput :=
  { base := Tensor.full #[2] 3.0
    tag := "input" }

def sampleRuntimePadOutput : RuntimePadOutput :=
  { loss := Tensor.full #[] 4.0
    label := "output" }

def replacementRuntimePadParams : RuntimePadParams :=
  { padv := Tensor.full #[1] 5.0
    label := "replacement-params" }

def replacementRuntimePadInput : RuntimePadInput :=
  { base := Tensor.full #[2] 6.0
    tag := "replacement-input" }

def replacementRuntimePadOutput : RuntimePadOutput :=
  { loss := Tensor.full #[] 7.0
    label := "replacement-output" }

def runtimePadFrontendSig : FrontendADSignature := {
  params := #[
    FrontendBoundary.ofValue sampleRuntimePadParams .diffOnly
  ]
  inputs := #[
    FrontendBoundary.ofValue sampleRuntimePadInput .diffOnly
  ]
  outputs := #[
    FrontendBoundary.ofValue sampleRuntimePadOutput .diffOnly
  ]
}

def runtimePadFrontendJaxpr : LeanJaxpr := {
  invars := #[
    { id := 20, ty := Lean.IR.IRType.object, metaInfo := { participation := .diff, dtype := some "Float32", shape := some #[1] } },
    { id := 21, ty := Lean.IR.IRType.object, metaInfo := { participation := .diff, dtype := some "Float32", shape := some #[2] } }
  ]
  eqns := #[
    {
      op := `test.runtime_frontend_loss
      invars := #[
        { id := 20, ty := Lean.IR.IRType.object, metaInfo := { participation := .diff, dtype := some "Float32", shape := some #[1] } },
        { id := 21, ty := Lean.IR.IRType.object, metaInfo := { participation := .diff, dtype := some "Float32", shape := some #[2] } }
      ]
      outvars := #[
        { id := 22, ty := Lean.IR.IRType.object, metaInfo := { participation := .diff, dtype := some "Float32", shape := some #[] } }
      ]
      params := #[
        Tyr.AD.JaxprLike.OpParam.mkName .sourceOp `Graphax.runtime_frontend_loss
      ]
    }
  ]
  outvars := #[
    { id := 22, ty := Lean.IR.IRType.object, metaInfo := { participation := .diff, dtype := some "Float32", shape := some #[] } }
  ]
}

def runtimePadFrontendFn : StructuredFrontendFunction RuntimePadParams RuntimePadInput RuntimePadOutput := {
  signature := runtimePadFrontendSig
  evalFlat := fun invarLeaves => do
    if invarLeaves.size != runtimePadFrontendSig.paramLeafCount + runtimePadFrontendSig.inputLeafCount then
      throw <| s!"Expected {runtimePadFrontendSig.paramLeafCount + runtimePadFrontendSig.inputLeafCount} flattened invar leaves, " ++
        s!"got {invarLeaves.size}."
    pure {
      outputTemplate := sampleRuntimePadOutput
      outputLeaves := TensorStructFlatten.flatten replacementRuntimePadOutput .diffOnly
    }
  linearizeFlat := fun invarLeaves => do
    if invarLeaves.size != runtimePadFrontendSig.paramLeafCount + runtimePadFrontendSig.inputLeafCount then
      throw <| s!"Expected {runtimePadFrontendSig.paramLeafCount + runtimePadFrontendSig.inputLeafCount} flattened invar leaves, " ++
        s!"got {invarLeaves.size}."
    pure {
      outputTemplate := sampleRuntimePadOutput
      outputLeaves := TensorStructFlatten.flatten replacementRuntimePadOutput .diffOnly
      pullback := fun outputCotangentLeaves => do
        if outputCotangentLeaves.size != runtimePadFrontendSig.outputLeafCount then
          throw <| s!"Expected {runtimePadFrontendSig.outputLeafCount} flattened output cotangent leaves, " ++
            s!"got {outputCotangentLeaves.size}."
        pure <|
          TensorStructFlatten.flatten replacementRuntimePadParams .diffOnly ++
          TensorStructFlatten.flatten replacementRuntimePadInput .diffOnly
    }
}

def runtimePadFrontend : FrontendRegistration := {
  jaxpr := runtimePadFrontendJaxpr
  signature := some runtimePadFrontendSig
  runtimeFrontend := some `Tests.ADJaxprLikeElabFixture.runtimePadFrontendFn
}

def runtimeStructuredPadStub (_params : RuntimePadParams) (_inp : RuntimePadInput) : RuntimePadOutput :=
  { loss := Tensor.full #[] 0.0
    label := "stub" }

attribute [ad_frontend Tests.ADJaxprLikeElabFixture.runtimePadFrontend] runtimeStructuredPadStub

end Tests.ADJaxprLikeElabFixture
