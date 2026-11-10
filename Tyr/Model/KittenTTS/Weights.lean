/-
  Tyr/Model/KittenTTS/Weights.lean

  Pretrained weight loading for the Lean KittenTTS / Kokoro port.
  Loads converted SafeTensors checkpoints produced from upstream PyTorch
  checkpoints so single-utterance synthesis can use real weights.
-/
import Tyr.Typed.Neural
import Tyr.TensorStruct
import Tyr.Log
import Tyr.Model.KittenTTS.Checkpoint
import Tyr.Model.KittenTTS.Model

namespace torch.kittentts

variable {device : Device}

open torch.Log

private structure LoadedCheckpointWeights where
  bert : KokoroCheckpoint.WeightsBert
  bertEncoder : KokoroCheckpoint.WeightsBertencoder
  predictor : KokoroCheckpoint.WeightsPredictor
  textEncoder : KokoroCheckpoint.WeightsTextencoder
  decoder : KokoroCheckpoint.WeightsDecoder

/-- Move a checkpoint tensor to the model device. -/
private def onDevice {s : Shape} {sourceDevice : Device}
    (tensor : Tensor { shape := s, dtype := .Float32, device := sourceDevice }) :
    IO (Tensor { shape := s, dtype := .Float32, device := device }) := do
  pure (Tensor.ofRaw! (← RawTensor.convertIO s tensor.raw device none))

/-- The configuration of the Kokoro v1 checkpoint (`kokoro_v1.schema.json`). The
    decoder blocks are 1024 channels wide, independent of `max_conv_dim`. -/
def KittenTTSConfig.kokoroV1 : KittenTTSConfig :=
  { maxConvDim := 1024, decoderOutDim := 512, plbert := { intermediateSize := 2048 } }

private abbrev kokoro := KittenTTSConfig.kokoroV1

private def reqGradFalse {s : Shape} (t : Tensor { shape := s, dtype := .Float32, device := device }) : Tensor { shape := s, dtype := .Float32, device := device } :=
  Tensor.detach t

private def getOrThrow {α} (xs : Array α) (idx : Nat) (msg : String) : IO α :=
  match xs[idx]? with
  | some x => pure x
  | none => throw <| IO.userError msg

private def warnShapeOverride (log : Handlers) (field : String) (cfgValue actualValue : Nat) : IO Unit := do
  if cfgValue != actualValue then
    log.onWarn s!"Checkpoint overrides {field}: config={cfgValue}, checkpoint={actualValue}"
  pure ()

private def normDim0_3d {d0 d1 d2 : Nat}
    (x : Tensor { shape := #[d0, d1, d2], dtype := .Float32, device := device })
    : Tensor { shape := #[d0], dtype := .Float32, device := device } :=
  let sq : Tensor { shape := #[d0, d1, d2], dtype := .Float32, device := device } := x * x
  let s2 : Tensor { shape := #[d0, d1, 1], dtype := .Float32, device := device } := Tensor.sumDim sq 2 true
  let s1 : Tensor { shape := #[d0, 1, 1], dtype := .Float32, device := device } := Tensor.sumDim s2 1 true
  Tensor.reshape (Tensor.sqrt (s1 + (1e-12 : Float))) #[d0]

private def materializeWeightNormConv1d {outC inC kernel : Nat}
    (weightV : Tensor { shape := #[outC, inC, kernel], dtype := .Float32, device := device })
    (weightG : Tensor { shape := #[outC], dtype := .Float32, device := device })
    : Tensor { shape := #[outC, inC, kernel], dtype := .Float32, device := device } :=
  let denom : Tensor { shape := #[outC, 1, 1], dtype := .Float32, device := device } := Tensor.reshape (normDim0_3d weightV) #[outC, 1, 1]
  let denomE : Tensor { shape := #[outC, inC, kernel], dtype := .Float32, device := device } := Tensor.broadcastTo denom #[outC, inC, kernel]
  let gain : Tensor { shape := #[outC, 1, 1], dtype := .Float32, device := device } := Tensor.reshape weightG #[outC, 1, 1]
  let gainE : Tensor { shape := #[outC, inC, kernel], dtype := .Float32, device := device } := Tensor.broadcastTo gain #[outC, inC, kernel]
  Tensor.div (weightV * gainE) denomE

private def linearFromTensors {inDim outDim : Nat}
    (weight : Tensor { shape := #[outDim, inDim], dtype := .Float32, device := device })
    (bias : Tensor { shape := #[outDim], dtype := .Float32, device := device })
    : LinearNorm (device := device) inDim outDim := {
      weight := reqGradFalse weight
      bias := reqGradFalse bias
    }

private def biLSTMFromTensors {inputSize hiddenSize : Nat}
    (wxForward : Tensor { shape := #[4 * hiddenSize, inputSize], dtype := .Float32, device := device })
    (whForward : Tensor { shape := #[4 * hiddenSize, hiddenSize], dtype := .Float32, device := device })
    (biasIHForward : Tensor { shape := #[4 * hiddenSize], dtype := .Float32, device := device })
    (biasHHForward : Tensor { shape := #[4 * hiddenSize], dtype := .Float32, device := device })
    (wxBackward : Tensor { shape := #[4 * hiddenSize, inputSize], dtype := .Float32, device := device })
    (whBackward : Tensor { shape := #[4 * hiddenSize, hiddenSize], dtype := .Float32, device := device })
    (biasIHBackward : Tensor { shape := #[4 * hiddenSize], dtype := .Float32, device := device })
    (biasHHBackward : Tensor { shape := #[4 * hiddenSize], dtype := .Float32, device := device })
    : BiLSTM (device := device) inputSize hiddenSize := {
      wxForward := reqGradFalse wxForward
      whForward := reqGradFalse whForward
      biasIHForward := reqGradFalse biasIHForward
      biasHHForward := reqGradFalse biasHHForward
      wxBackward := reqGradFalse wxBackward
      whBackward := reqGradFalse whBackward
      biasIHBackward := reqGradFalse biasIHBackward
      biasHHBackward := reqGradFalse biasHHBackward
    }

private def conv1dFromWeightNorm {inC outC kernel stride padding dilation : Nat}
    (weightV : Tensor { shape := #[outC, inC, kernel], dtype := .Float32, device := device })
    (weightG3 : Tensor { shape := #[outC, 1, 1], dtype := .Float32, device := device })
    (bias : Tensor { shape := #[outC], dtype := .Float32, device := device })
    : Conv1dParams (device := device) inC outC kernel stride padding dilation := {
      weight := reqGradFalse <| materializeWeightNormConv1d weightV (Tensor.reshape weightG3 #[outC])
      bias := reqGradFalse bias
    }

private def wnConv1dFromWeightNorm {inC outC kernel stride padding dilation : Nat}
    (weightV : Tensor { shape := #[outC, inC, kernel], dtype := .Float32, device := device })
    (weightG3 : Tensor { shape := #[outC, 1, 1], dtype := .Float32, device := device })
    (bias : Tensor { shape := #[outC], dtype := .Float32, device := device })
    : WNConv1dParams (device := device) inC outC kernel stride padding dilation := {
      weightV := reqGradFalse weightV
      weightG := reqGradFalse (Tensor.reshape weightG3 #[outC])
      bias := reqGradFalse bias
    }

private def wnDepthwiseTransConv1dFromWeightNorm
    {channels kernel stride padding outputPadding dilation : Nat}
    (weightV : Tensor { shape := #[channels, 1, kernel], dtype := .Float32, device := device })
    (weightG3 : Tensor { shape := #[channels, 1, 1], dtype := .Float32, device := device })
    (bias : Tensor { shape := #[channels], dtype := .Float32, device := device })
    : WNDepthwiseTransConv1dParams (device := device) channels kernel stride padding outputPadding dilation := {
      weightV := reqGradFalse weightV
      weightG := reqGradFalse (Tensor.reshape weightG3 #[channels])
      bias := reqGradFalse bias
    }

private def wnConvTranspose1dFromWeightNorm
    {inC outC kernel stride padding outputPadding dilation : Nat}
    (weightV : Tensor { shape := #[inC, outC, kernel], dtype := .Float32, device := device })
    (weightG3 : Tensor { shape := #[inC, 1, 1], dtype := .Float32, device := device })
    (bias : Tensor { shape := #[outC], dtype := .Float32, device := device })
    : WNConvTranspose1dParams (device := device) inC outC kernel stride padding outputPadding dilation := {
      weightV := reqGradFalse weightV
      weightG := reqGradFalse (Tensor.reshape weightG3 #[inC])
      bias := reqGradFalse bias
    }

private def adaLayerNormFromLinear {styleDim channels : Nat}
    (weight : Tensor { shape := #[2 * channels, styleDim], dtype := .Float32, device := device })
    (bias : Tensor { shape := #[2 * channels], dtype := .Float32, device := device })
    : AdaLayerNorm (device := device) styleDim channels := {
      fc := linearFromTensors weight bias
    }

private def adaIN1dFromLinear {styleDim channels : Nat}
    (weight : Tensor { shape := #[2 * channels, styleDim], dtype := .Float32, device := device })
    (bias : Tensor { shape := #[2 * channels], dtype := .Float32, device := device })
    : IO (AdaIN1d (device := device) styleDim channels) := do
  return {
    fc := linearFromTensors weight bias
    normWeight := Tensor.ones #[channels] (device := device)
    normBias := Tensor.zeros #[channels] (device := device)
  }

private def wnConv1dFromPlain {inC outC kernel stride padding dilation : Nat}
    (weight : Tensor { shape := #[outC, inC, kernel], dtype := .Float32, device := device })
    (bias : Tensor { shape := #[outC], dtype := .Float32, device := device })
    : WNConv1dParams (device := device) inC outC kernel stride padding dilation := {
      weightV := reqGradFalse weight
      weightG := reqGradFalse (normDim0_3d weight)
      bias := reqGradFalse bias
    }

private def loadGenResUnit {α β γ δ ε ζ : Type} {styleDim channels kernel dilation : Nat}
    (convs1 : Array α) (convs2 : Array β) (adain1 : Array γ) (adain2 : Array δ)
    (alpha1 : Array ε) (alpha2 : Array ζ)
    (mkConv1 : α → IO (WNConv1dParams (device := device) channels channels kernel 1 ((kernel * dilation - dilation) / 2) dilation))
    (mkConv2 : β → IO (WNConv1dParams (device := device) channels channels kernel 1 ((kernel - 1) / 2) 1))
    (mkAdaIN1 : γ → IO (AdaIN1d (device := device) styleDim channels))
    (mkAdaIN2 : δ → IO (AdaIN1d (device := device) styleDim channels))
    (mkAlpha1 : ε → IO (Tensor { shape := #[1, channels, 1], dtype := .Float32, device := device }))
    (mkAlpha2 : ζ → IO (Tensor { shape := #[1, channels, 1], dtype := .Float32, device := device }))
    (name : String) (k : Nat)
    : IO (GenResUnit (device := device) styleDim channels kernel dilation) := do
  pure {
    conv1 := ← mkConv1 (← getOrThrow convs1 k s!"Missing {name}.convs1.{k}")
    conv2 := ← mkConv2 (← getOrThrow convs2 k s!"Missing {name}.convs2.{k}")
    norm1 := ← mkAdaIN1 (← getOrThrow adain1 k s!"Missing {name}.adain1.{k}")
    norm2 := ← mkAdaIN2 (← getOrThrow adain2 k s!"Missing {name}.adain2.{k}")
    alpha1 := ← mkAlpha1 (← getOrThrow alpha1 k s!"Missing {name}.alpha1.{k}")
    alpha2 := ← mkAlpha2 (← getOrThrow alpha2 k s!"Missing {name}.alpha2.{k}")
  }

/-- Load one unit of a generator residual block from its checkpoint node arrays. -/
macro "loadGenResUnit%" blk:term "," name:term "," k:term : term => `(
  loadGenResUnit ($blk).convs1 ($blk).convs2 ($blk).adain1 ($blk).adain2 ($blk).alpha1 ($blk).alpha2
    (fun node => do pure (wnConv1dFromWeightNorm (← onDevice node.weight_v) (← onDevice node.weight_g) (← onDevice node.bias)))
    (fun node => do pure (wnConv1dFromWeightNorm (← onDevice node.weight_v) (← onDevice node.weight_g) (← onDevice node.bias)))
    (fun node => do adaIN1dFromLinear (← onDevice node.fc.weight) (← onDevice node.fc.bias))
    (fun node => do adaIN1dFromLinear (← onDevice node.fc.weight) (← onDevice node.fc.bias))
    (fun alpha => do pure (reqGradFalse (← onDevice alpha)))
    (fun alpha => do pure (reqGradFalse (← onDevice alpha)))
    $name $k)

/-- Load a three-unit generator residual block from its checkpoint node arrays. -/
macro "loadGenResBlock%" blk:term "," name:term : term => `(do
  pure {
    u0 := ← loadGenResUnit% $blk, $name, 0
    u1 := ← loadGenResUnit% $blk, $name, 1
    u2 := ← loadGenResUnit% $blk, $name, 2
  })

private def loadKittenAlbertFromCheckpoint
    (weights : KokoroCheckpoint.WeightsBert)
    : IO (KittenAlbert (device := device) kokoro.nToken kokoro.plbert) := do
  let embeddings : AlbertEmbeddings (device := device) kokoro.nToken kokoro.plbert := {
    wordEmbeddings := reqGradFalse (← onDevice weights.embeddings.word_embeddings.weight)
    positionEmbeddings := reqGradFalse (← onDevice weights.embeddings.position_embeddings.weight)
    tokenTypeEmbeddings := reqGradFalse (← onDevice weights.embeddings.token_type_embeddings.weight)
    layerNormWeight := reqGradFalse (← onDevice weights.embeddings.layernorm.weight)
    layerNormBias := reqGradFalse (← onDevice weights.embeddings.layernorm.bias)
  }
  let groups : Array (KittenAlbertLayerGroup (device := device) kokoro.plbert) ←
    weights.encoder.albert_layer_groups.mapM fun group => do
      let layers ← group.albert_layers.mapM fun layer => do pure {
        attention := {
          query := linearFromTensors (← onDevice layer.attention.query.weight) (← onDevice layer.attention.query.bias)
          key := linearFromTensors (← onDevice layer.attention.key.weight) (← onDevice layer.attention.key.bias)
          value := linearFromTensors (← onDevice layer.attention.value.weight) (← onDevice layer.attention.value.bias)
          dense := linearFromTensors (← onDevice layer.attention.dense.weight) (← onDevice layer.attention.dense.bias)
          layerNormWeight := reqGradFalse (← onDevice layer.attention.layernorm.weight)
          layerNormBias := reqGradFalse (← onDevice layer.attention.layernorm.bias)
        }
        fullLayerNormWeight := reqGradFalse (← onDevice layer.full_layer_layer_norm.weight)
        fullLayerNormBias := reqGradFalse (← onDevice layer.full_layer_layer_norm.bias)
        ffn := linearFromTensors (← onDevice layer.ffn.weight) (← onDevice layer.ffn.bias)
        ffnOutput := linearFromTensors (← onDevice layer.ffn_output.weight) (← onDevice layer.ffn_output.bias)
      }
      unless layers.size = kokoro.plbert.innerGroupNum do
        throw <| IO.userError
          s!"KittenTTS checkpoint ALBERT group has {layers.size} layers, config inner_group_num={kokoro.plbert.innerGroupNum}"
      pure { layers }
  let ⟨hGroups⟩ ← if h : groups.size = kokoro.plbert.numHiddenGroups then pure (PLift.up h) else
    throw <| IO.userError
      s!"KittenTTS checkpoint has {groups.size} ALBERT layer groups, config num_hidden_groups={kokoro.plbert.numHiddenGroups}"
  let encoder : KittenAlbertEncoder (device := device) kokoro.plbert := {
    embeddingHiddenMappingIn :=
      linearFromTensors
        (← onDevice weights.encoder.embedding_hidden_mapping_in.weight)
        (← onDevice weights.encoder.embedding_hidden_mapping_in.bias)
    groups := ⟨groups, hGroups⟩
  }
  let pooler : LinearNorm (device := device) kokoro.plbert.hiddenSize kokoro.plbert.hiddenSize :=
    linearFromTensors (← onDevice weights.pooler.weight) (← onDevice weights.pooler.bias)
  pure { embeddings, encoder, pooler }

private def loadTextEncoderFromCheckpoint
    (weights : KokoroCheckpoint.WeightsTextencoder)
    : IO (TextEncoder (device := device) kokoro) := do
  let convs : Array (TextConvLayer (device := device) kokoro) ←
    weights.cnn.mapM fun layer => do pure {
      conv := conv1dFromWeightNorm (← onDevice layer.i0.weight_v) (← onDevice layer.i0.weight_g) (← onDevice layer.i0.bias)
      lnWeight := reqGradFalse (← onDevice layer.i1.gamma)
      lnBias := reqGradFalse (← onDevice layer.i1.beta)
    }
  let lstm :=
    biLSTMFromTensors
      (← onDevice weights.lstm.weight_ih_l0)
      (← onDevice weights.lstm.weight_hh_l0)
      (← onDevice weights.lstm.bias_ih_l0)
      (← onDevice weights.lstm.bias_hh_l0)
      (← onDevice weights.lstm.weight_ih_l0_reverse)
      (← onDevice weights.lstm.weight_hh_l0_reverse)
      (← onDevice weights.lstm.bias_ih_l0_reverse)
      (← onDevice weights.lstm.bias_hh_l0_reverse)
  pure {
    embedding := reqGradFalse (← onDevice weights.embedding.weight)
    convs
    lstm
  }

private def loadDurationEncoderFromCheckpoint
    (weights : KokoroCheckpoint.WeightsPredictorTextencoder)
    : IO (DurationEncoder (device := device) kokoro) := do
  pure {
      lstms := ⟨#[
        biLSTMFromTensors
          (← onDevice weights.lstms.i0.weight_ih_l0)
          (← onDevice weights.lstms.i0.weight_hh_l0)
          (← onDevice weights.lstms.i0.bias_ih_l0)
          (← onDevice weights.lstms.i0.bias_hh_l0)
          (← onDevice weights.lstms.i0.weight_ih_l0_reverse)
          (← onDevice weights.lstms.i0.weight_hh_l0_reverse)
          (← onDevice weights.lstms.i0.bias_ih_l0_reverse)
          (← onDevice weights.lstms.i0.bias_hh_l0_reverse),
        biLSTMFromTensors
          (← onDevice weights.lstms.i2.weight_ih_l0)
          (← onDevice weights.lstms.i2.weight_hh_l0)
          (← onDevice weights.lstms.i2.bias_ih_l0)
          (← onDevice weights.lstms.i2.bias_hh_l0)
          (← onDevice weights.lstms.i2.weight_ih_l0_reverse)
          (← onDevice weights.lstms.i2.weight_hh_l0_reverse)
          (← onDevice weights.lstms.i2.bias_ih_l0_reverse)
          (← onDevice weights.lstms.i2.bias_hh_l0_reverse),
        biLSTMFromTensors
          (← onDevice weights.lstms.i4.weight_ih_l0)
          (← onDevice weights.lstms.i4.weight_hh_l0)
          (← onDevice weights.lstms.i4.bias_ih_l0)
          (← onDevice weights.lstms.i4.bias_hh_l0)
          (← onDevice weights.lstms.i4.weight_ih_l0_reverse)
          (← onDevice weights.lstms.i4.weight_hh_l0_reverse)
          (← onDevice weights.lstms.i4.bias_ih_l0_reverse)
          (← onDevice weights.lstms.i4.bias_hh_l0_reverse)
      ], rfl⟩
      norms := ⟨#[
        adaLayerNormFromLinear (← onDevice weights.lstms.i1.fc.weight) (← onDevice weights.lstms.i1.fc.bias),
        adaLayerNormFromLinear (← onDevice weights.lstms.i3.fc.weight) (← onDevice weights.lstms.i3.fc.bias),
        adaLayerNormFromLinear (← onDevice weights.lstms.i5.fc.weight) (← onDevice weights.lstms.i5.fc.bias)
      ], rfl⟩
    }

private def loadProsodyPredictorFromCheckpoint
    (weights : KokoroCheckpoint.WeightsPredictor)
    (log : Handlers := {})
    : IO (ProsodyPredictor (device := device) kokoro) := do
  log.onInfo "  predictor.text_encoder..."
  let durationEncoder ← loadDurationEncoderFromCheckpoint weights.text_encoder
  log.onInfo "  predictor.lstm..."
  let lstm :=
    biLSTMFromTensors
      (← onDevice weights.lstm.weight_ih_l0)
      (← onDevice weights.lstm.weight_hh_l0)
      (← onDevice weights.lstm.bias_ih_l0)
      (← onDevice weights.lstm.bias_hh_l0)
      (← onDevice weights.lstm.weight_ih_l0_reverse)
      (← onDevice weights.lstm.weight_hh_l0_reverse)
      (← onDevice weights.lstm.bias_ih_l0_reverse)
      (← onDevice weights.lstm.bias_hh_l0_reverse)
  log.onInfo "  predictor.duration_proj..."
  let durationProj : LinearNorm (device := device) kokoro.hiddenDim kokoro.maxDur :=
    linearFromTensors
      (← onDevice weights.duration_proj.linear_layer.weight)
      (← onDevice weights.duration_proj.linear_layer.bias)
  log.onInfo "  predictor.shared..."
  let shared :=
    biLSTMFromTensors
      (← onDevice weights.shared.weight_ih_l0)
      (← onDevice weights.shared.weight_hh_l0)
      (← onDevice weights.shared.bias_ih_l0)
      (← onDevice weights.shared.bias_hh_l0)
      (← onDevice weights.shared.weight_ih_l0_reverse)
      (← onDevice weights.shared.weight_hh_l0_reverse)
      (← onDevice weights.shared.bias_ih_l0_reverse)
      (← onDevice weights.shared.bias_hh_l0_reverse)
  log.onInfo "  predictor.F0 blocks..."
  let f0Blk0 : AdainResBlk1dSame (device := device) kokoro.hiddenDim kokoro.hiddenDim kokoro.styleDim := {
    conv1 := wnConv1dFromWeightNorm (← onDevice weights.f0.i0.conv1.weight_v) (← onDevice weights.f0.i0.conv1.weight_g) (← onDevice weights.f0.i0.conv1.bias)
    conv2 := wnConv1dFromWeightNorm (← onDevice weights.f0.i0.conv2.weight_v) (← onDevice weights.f0.i0.conv2.weight_g) (← onDevice weights.f0.i0.conv2.bias)
    norm1 := ← adaIN1dFromLinear (← onDevice weights.f0.i0.norm1.fc.weight) (← onDevice weights.f0.i0.norm1.fc.bias)
    norm2 := ← adaIN1dFromLinear (← onDevice weights.f0.i0.norm2.fc.weight) (← onDevice weights.f0.i0.norm2.fc.bias)
    shortcut := .identity rfl
  }
  let f0Blk1 : AdainResBlk1dUp (device := device) kokoro.hiddenDim (kokoro.hiddenDim / 2) kokoro.styleDim := {
    pool := wnDepthwiseTransConv1dFromWeightNorm (← onDevice weights.f0.i1.pool.weight_v) (← onDevice weights.f0.i1.pool.weight_g) (← onDevice weights.f0.i1.pool.bias)
    conv1 := wnConv1dFromWeightNorm (← onDevice weights.f0.i1.conv1.weight_v) (← onDevice weights.f0.i1.conv1.weight_g) (← onDevice weights.f0.i1.conv1.bias)
    conv2 := wnConv1dFromWeightNorm (← onDevice weights.f0.i1.conv2.weight_v) (← onDevice weights.f0.i1.conv2.weight_g) (← onDevice weights.f0.i1.conv2.bias)
    norm1 := ← adaIN1dFromLinear (← onDevice weights.f0.i1.norm1.fc.weight) (← onDevice weights.f0.i1.norm1.fc.bias)
    norm2 := ← adaIN1dFromLinear (← onDevice weights.f0.i1.norm2.fc.weight) (← onDevice weights.f0.i1.norm2.fc.bias)
    shortcut := .conv <| wnConv1dFromWeightNorm (← onDevice weights.f0.i1.conv1x1.weight_v) (← onDevice weights.f0.i1.conv1x1.weight_g) (Tensor.zeros #[kokoro.hiddenDim / 2] (device := device))
  }
  let f0Blk2 : AdainResBlk1dSame (device := device) (kokoro.hiddenDim / 2) (kokoro.hiddenDim / 2) kokoro.styleDim := {
    conv1 := wnConv1dFromWeightNorm (← onDevice weights.f0.i2.conv1.weight_v) (← onDevice weights.f0.i2.conv1.weight_g) (← onDevice weights.f0.i2.conv1.bias)
    conv2 := wnConv1dFromWeightNorm (← onDevice weights.f0.i2.conv2.weight_v) (← onDevice weights.f0.i2.conv2.weight_g) (← onDevice weights.f0.i2.conv2.bias)
    norm1 := ← adaIN1dFromLinear (← onDevice weights.f0.i2.norm1.fc.weight) (← onDevice weights.f0.i2.norm1.fc.bias)
    norm2 := ← adaIN1dFromLinear (← onDevice weights.f0.i2.norm2.fc.weight) (← onDevice weights.f0.i2.norm2.fc.bias)
    shortcut := .identity rfl
  }
  log.onInfo "  predictor.N blocks..."
  let nBlk0 : AdainResBlk1dSame (device := device) kokoro.hiddenDim kokoro.hiddenDim kokoro.styleDim := {
    conv1 := wnConv1dFromWeightNorm (← onDevice weights.n.i0.conv1.weight_v) (← onDevice weights.n.i0.conv1.weight_g) (← onDevice weights.n.i0.conv1.bias)
    conv2 := wnConv1dFromWeightNorm (← onDevice weights.n.i0.conv2.weight_v) (← onDevice weights.n.i0.conv2.weight_g) (← onDevice weights.n.i0.conv2.bias)
    norm1 := ← adaIN1dFromLinear (← onDevice weights.n.i0.norm1.fc.weight) (← onDevice weights.n.i0.norm1.fc.bias)
    norm2 := ← adaIN1dFromLinear (← onDevice weights.n.i0.norm2.fc.weight) (← onDevice weights.n.i0.norm2.fc.bias)
    shortcut := .identity rfl
  }
  let nBlk1 : AdainResBlk1dUp (device := device) kokoro.hiddenDim (kokoro.hiddenDim / 2) kokoro.styleDim := {
    pool := wnDepthwiseTransConv1dFromWeightNorm (← onDevice weights.n.i1.pool.weight_v) (← onDevice weights.n.i1.pool.weight_g) (← onDevice weights.n.i1.pool.bias)
    conv1 := wnConv1dFromWeightNorm (← onDevice weights.n.i1.conv1.weight_v) (← onDevice weights.n.i1.conv1.weight_g) (← onDevice weights.n.i1.conv1.bias)
    conv2 := wnConv1dFromWeightNorm (← onDevice weights.n.i1.conv2.weight_v) (← onDevice weights.n.i1.conv2.weight_g) (← onDevice weights.n.i1.conv2.bias)
    norm1 := ← adaIN1dFromLinear (← onDevice weights.n.i1.norm1.fc.weight) (← onDevice weights.n.i1.norm1.fc.bias)
    norm2 := ← adaIN1dFromLinear (← onDevice weights.n.i1.norm2.fc.weight) (← onDevice weights.n.i1.norm2.fc.bias)
    shortcut := .conv <| wnConv1dFromWeightNorm (← onDevice weights.n.i1.conv1x1.weight_v) (← onDevice weights.n.i1.conv1x1.weight_g) (Tensor.zeros #[kokoro.hiddenDim / 2] (device := device))
  }
  let nBlk2 : AdainResBlk1dSame (device := device) (kokoro.hiddenDim / 2) (kokoro.hiddenDim / 2) kokoro.styleDim := {
    conv1 := wnConv1dFromWeightNorm (← onDevice weights.n.i2.conv1.weight_v) (← onDevice weights.n.i2.conv1.weight_g) (← onDevice weights.n.i2.conv1.bias)
    conv2 := wnConv1dFromWeightNorm (← onDevice weights.n.i2.conv2.weight_v) (← onDevice weights.n.i2.conv2.weight_g) (← onDevice weights.n.i2.conv2.bias)
    norm1 := ← adaIN1dFromLinear (← onDevice weights.n.i2.norm1.fc.weight) (← onDevice weights.n.i2.norm1.fc.bias)
    norm2 := ← adaIN1dFromLinear (← onDevice weights.n.i2.norm2.fc.weight) (← onDevice weights.n.i2.norm2.fc.bias)
    shortcut := .identity rfl
  }
  log.onInfo "  predictor output projections..."
  pure {
    durationEncoder
    lstm
    durationProj
    shared
    f0Blk0
    f0Blk1
    f0Blk2
    nBlk0
    nBlk1
    nBlk2
    f0Proj := {
      weight := reqGradFalse (← onDevice weights.f0_proj.weight)
      bias := reqGradFalse (← onDevice weights.f0_proj.bias)
    }
    nProj := {
      weight := reqGradFalse (← onDevice weights.n_proj.weight)
      bias := reqGradFalse (← onDevice weights.n_proj.bias)
    }
  }

private def loadGeneratorFromCheckpoint
    (weights : KokoroCheckpoint.WeightsDecoderGenerator)
    : IO (Generator (device := device) kokoro) := do
  let source : SourceModuleHnNSF (device := device) kokoro := {
    linear := linearFromTensors (← onDevice weights.m_source.l_linear.weight) (← onDevice weights.m_source.l_linear.bias)
  }
  have hStages : kokoro.generator.upsampleRates.size = 2 := rfl
  have hBlocks : kokoro.generator.resblockKernelSizes.size = 3 := rfl
  let stage0 : GeneratorStage (device := device) kokoro 0 := {
    up := wnConvTranspose1dFromWeightNorm (← onDevice weights.ups.i0.weight_v) (← onDevice weights.ups.i0.weight_g) (← onDevice weights.ups.i0.bias)
    noiseConv := wnConv1dFromPlain (← onDevice weights.noise_convs.i0.weight) (← onDevice weights.noise_convs.i0.bias)
    noiseRes := ← loadGenResBlock% weights.noise_res.i0, "decoder.generator.noise_res.0"
    resBlocks :=
      .cons (by omega) (← loadGenResBlock% weights.resblocks.i0, "decoder.generator.resblocks.0")
        (.cons (by omega) (← loadGenResBlock% weights.resblocks.i1, "decoder.generator.resblocks.1")
          (.last (by omega) (← loadGenResBlock% weights.resblocks.i2, "decoder.generator.resblocks.2")))
  }
  let stage1 : GeneratorStage (device := device) kokoro 1 := {
    up := wnConvTranspose1dFromWeightNorm (← onDevice weights.ups.i1.weight_v) (← onDevice weights.ups.i1.weight_g) (← onDevice weights.ups.i1.bias)
    noiseConv := wnConv1dFromPlain (← onDevice weights.noise_convs.i1.weight) (← onDevice weights.noise_convs.i1.bias)
    noiseRes := ← loadGenResBlock% weights.noise_res.i1, "decoder.generator.noise_res.1"
    resBlocks :=
      .cons (by omega) (← loadGenResBlock% weights.resblocks.i3, "decoder.generator.resblocks.3")
        (.cons (by omega) (← loadGenResBlock% weights.resblocks.i4, "decoder.generator.resblocks.4")
          (.last (by omega) (← loadGenResBlock% weights.resblocks.i5, "decoder.generator.resblocks.5")))
  }
  let convPost := wnConv1dFromWeightNorm (← onDevice weights.conv_post.weight_v) (← onDevice weights.conv_post.weight_g) (← onDevice weights.conv_post.bias)
  pure {
    source
    stages := .cons (by omega) stage0 (.last (by omega) stage1)
    convPost
    window := ← onDevice (Tensor.hannWindow kokoro.generator.genIstftNFft)
  }

private def loadDecoderFromCheckpoint
    (weights : KokoroCheckpoint.WeightsDecoder)
    : IO (Decoder (device := device) kokoro) := do
  let asrResNode ← getOrThrow weights.asr_res 0 "Missing decoder.asr_res.0"
  let generator ← loadGeneratorFromCheckpoint weights.generator
  let encode : AdainResBlk1dSame (device := device) (kokoro.hiddenDim + 2) kokoro.maxConvDim kokoro.styleDim := {
    conv1 := wnConv1dFromWeightNorm (← onDevice weights.encode.conv1.weight_v) (← onDevice weights.encode.conv1.weight_g) (← onDevice weights.encode.conv1.bias)
    conv2 := wnConv1dFromWeightNorm (← onDevice weights.encode.conv2.weight_v) (← onDevice weights.encode.conv2.weight_g) (← onDevice weights.encode.conv2.bias)
    norm1 := ← adaIN1dFromLinear (← onDevice weights.encode.norm1.fc.weight) (← onDevice weights.encode.norm1.fc.bias)
    norm2 := ← adaIN1dFromLinear (← onDevice weights.encode.norm2.fc.weight) (← onDevice weights.encode.norm2.fc.bias)
    shortcut := .conv <| wnConv1dFromWeightNorm (← onDevice weights.encode.conv1x1.weight_v) (← onDevice weights.encode.conv1x1.weight_g) (Tensor.zeros #[kokoro.maxConvDim] (device := device))
  }
  let decode0 : AdainResBlk1dSame (device := device) (kokoro.maxConvDim + kokoro.asrResDim + 2) kokoro.maxConvDim kokoro.styleDim := {
    conv1 := wnConv1dFromWeightNorm (← onDevice weights.decode.i0.conv1.weight_v) (← onDevice weights.decode.i0.conv1.weight_g) (← onDevice weights.decode.i0.conv1.bias)
    conv2 := wnConv1dFromWeightNorm (← onDevice weights.decode.i0.conv2.weight_v) (← onDevice weights.decode.i0.conv2.weight_g) (← onDevice weights.decode.i0.conv2.bias)
    norm1 := ← adaIN1dFromLinear (← onDevice weights.decode.i0.norm1.fc.weight) (← onDevice weights.decode.i0.norm1.fc.bias)
    norm2 := ← adaIN1dFromLinear (← onDevice weights.decode.i0.norm2.fc.weight) (← onDevice weights.decode.i0.norm2.fc.bias)
    shortcut := .conv <| wnConv1dFromWeightNorm (← onDevice weights.decode.i0.conv1x1.weight_v) (← onDevice weights.decode.i0.conv1x1.weight_g) (Tensor.zeros #[kokoro.maxConvDim] (device := device))
  }
  let decode1 : AdainResBlk1dSame (device := device) (kokoro.maxConvDim + kokoro.asrResDim + 2) kokoro.maxConvDim kokoro.styleDim := {
    conv1 := wnConv1dFromWeightNorm (← onDevice weights.decode.i1.conv1.weight_v) (← onDevice weights.decode.i1.conv1.weight_g) (← onDevice weights.decode.i1.conv1.bias)
    conv2 := wnConv1dFromWeightNorm (← onDevice weights.decode.i1.conv2.weight_v) (← onDevice weights.decode.i1.conv2.weight_g) (← onDevice weights.decode.i1.conv2.bias)
    norm1 := ← adaIN1dFromLinear (← onDevice weights.decode.i1.norm1.fc.weight) (← onDevice weights.decode.i1.norm1.fc.bias)
    norm2 := ← adaIN1dFromLinear (← onDevice weights.decode.i1.norm2.fc.weight) (← onDevice weights.decode.i1.norm2.fc.bias)
    shortcut := .conv <| wnConv1dFromWeightNorm (← onDevice weights.decode.i1.conv1x1.weight_v) (← onDevice weights.decode.i1.conv1x1.weight_g) (Tensor.zeros #[kokoro.maxConvDim] (device := device))
  }
  let decode2 : AdainResBlk1dSame (device := device) (kokoro.maxConvDim + kokoro.asrResDim + 2) kokoro.maxConvDim kokoro.styleDim := {
    conv1 := wnConv1dFromWeightNorm (← onDevice weights.decode.i2.conv1.weight_v) (← onDevice weights.decode.i2.conv1.weight_g) (← onDevice weights.decode.i2.conv1.bias)
    conv2 := wnConv1dFromWeightNorm (← onDevice weights.decode.i2.conv2.weight_v) (← onDevice weights.decode.i2.conv2.weight_g) (← onDevice weights.decode.i2.conv2.bias)
    norm1 := ← adaIN1dFromLinear (← onDevice weights.decode.i2.norm1.fc.weight) (← onDevice weights.decode.i2.norm1.fc.bias)
    norm2 := ← adaIN1dFromLinear (← onDevice weights.decode.i2.norm2.fc.weight) (← onDevice weights.decode.i2.norm2.fc.bias)
    shortcut := .conv <| wnConv1dFromWeightNorm (← onDevice weights.decode.i2.conv1x1.weight_v) (← onDevice weights.decode.i2.conv1x1.weight_g) (Tensor.zeros #[kokoro.maxConvDim] (device := device))
  }
  let decode3 : AdainResBlk1dUp (device := device) (kokoro.maxConvDim + kokoro.asrResDim + 2) (KittenTTSConfig.decoderChannels kokoro) kokoro.styleDim := {
    pool := wnDepthwiseTransConv1dFromWeightNorm (← onDevice weights.decode.i3.pool.weight_v) (← onDevice weights.decode.i3.pool.weight_g) (← onDevice weights.decode.i3.pool.bias)
    conv1 := wnConv1dFromWeightNorm (← onDevice weights.decode.i3.conv1.weight_v) (← onDevice weights.decode.i3.conv1.weight_g) (← onDevice weights.decode.i3.conv1.bias)
    conv2 := wnConv1dFromWeightNorm (← onDevice weights.decode.i3.conv2.weight_v) (← onDevice weights.decode.i3.conv2.weight_g) (← onDevice weights.decode.i3.conv2.bias)
    norm1 := ← adaIN1dFromLinear (← onDevice weights.decode.i3.norm1.fc.weight) (← onDevice weights.decode.i3.norm1.fc.bias)
    norm2 := ← adaIN1dFromLinear (← onDevice weights.decode.i3.norm2.fc.weight) (← onDevice weights.decode.i3.norm2.fc.bias)
    shortcut := .conv <| wnConv1dFromWeightNorm (← onDevice weights.decode.i3.conv1x1.weight_v) (← onDevice weights.decode.i3.conv1x1.weight_g) (Tensor.zeros #[KittenTTSConfig.decoderChannels kokoro] (device := device))
  }
  pure {
    f0Conv := wnConv1dFromWeightNorm (← onDevice weights.f0_conv.weight_v) (← onDevice weights.f0_conv.weight_g) (← onDevice weights.f0_conv.bias)
    nConv := wnConv1dFromWeightNorm (← onDevice weights.n_conv.weight_v) (← onDevice weights.n_conv.weight_g) (← onDevice weights.n_conv.bias)
    encode
    asrRes := wnConv1dFromWeightNorm (← onDevice asrResNode.weight_v) (← onDevice asrResNode.weight_g) (← onDevice asrResNode.bias)
    decode0
    decode1
    decode2
    decode3
    generator
  }

private def resolveKittenDevice (log : Handlers := {}) : IO Device := do
  let requested := (← IO.getEnv "TYR_DEVICE").map String.toLower
  match requested with
  | some "cpu" => pure Device.CPU
  | some "cuda" =>
    if ← cuda_is_available then
      Device.cuda 0
    else
      log.onWarn "TYR_DEVICE=cuda requested but CUDA is unavailable; falling back to auto."
      getBestDevice
  | some "mps" => Device.mps
  | some "auto" => getBestDevice
  | some _ => getBestDevice
  | none =>
    if ← cuda_is_available then
      Device.cuda 0
    else
      pure Device.CPU

private def loadCheckpointWeights (path : String) (log : Handlers := {}) : IO LoadedCheckpointWeights := do
  let handle ← safetensors.openHandle path
  log.onInfo "Loading ALBERT backbone..."
  let bert ← KokoroCheckpoint.bert.loadFromHandle handle
  log.onInfo "Loading ALBERT projection..."
  let bertEncoder ← KokoroCheckpoint.bert_encoder.loadFromHandle handle
  log.onInfo "Loading prosody predictor..."
  let predictor ← KokoroCheckpoint.predictor.loadFromHandle handle
  log.onInfo "Loading text encoder..."
  let textEncoder ← KokoroCheckpoint.text_encoder.loadFromHandle handle
  log.onInfo "Loading decoder/vocoder..."
  let decoder ← KokoroCheckpoint.decoder.loadFromHandle handle
  pure { bert, bertEncoder, predictor, textEncoder, decoder }

/-- Check a `config.json` configuration once against `KittenTTSConfig.kokoroV1`.
    Sizes the checkpoint layout overrides are warned about; every other value
    must agree. -/
private def checkKokoroV1 (cfg : KittenTTSConfig) (log : Handlers := {}) : IO Unit := do
  let model := kokoro
  for (field, configured, expected) in [
      ("plbert.num_hidden_layers", cfg.plbert.numHiddenLayers, model.plbert.numHiddenLayers),
      ("plbert.num_attention_heads", cfg.plbert.numAttentionHeads, model.plbert.numAttentionHeads),
      ("plbert.hidden_size", cfg.plbert.hiddenSize, model.plbert.hiddenSize),
      ("plbert.intermediate_size", cfg.plbert.intermediateSize, model.plbert.intermediateSize),
      ("plbert.max_position_embeddings", cfg.plbert.maxPositionEmbeddings, model.plbert.maxPositionEmbeddings),
      ("plbert.embedding_size", cfg.plbert.embeddingSize, model.plbert.embeddingSize),
      ("plbert.inner_group_num", cfg.plbert.innerGroupNum, model.plbert.innerGroupNum),
      ("plbert.num_hidden_groups", cfg.plbert.numHiddenGroups, model.plbert.numHiddenGroups),
      ("plbert.type_vocab_size", cfg.plbert.typeVocabSize, model.plbert.typeVocabSize),
      ("istftnet.gen_istft_hop_size", cfg.generator.genIstftHopSize, model.generator.genIstftHopSize),
      ("sample_rate", cfg.sampleRate, model.sampleRate)] do
    unless configured == expected do
      throw <| IO.userError s!"KittenTTS {field}={configured} does not match the Kokoro v1 checkpoint ({expected})"
  for (field, configured, expected) in [
      ("upsample_rates", cfg.generator.upsampleRates, model.generator.upsampleRates),
      ("upsample_kernel_sizes", cfg.generator.upsampleKernelSizes, model.generator.upsampleKernelSizes),
      ("resblock_kernel_sizes", cfg.generator.resblockKernelSizes, model.generator.resblockKernelSizes)] do
    unless configured == expected do
      throw <| IO.userError s!"KittenTTS istftnet {field}={configured} does not match the Kokoro v1 checkpoint ({expected})"
  unless cfg.generator.resblockDilationSizes == model.generator.resblockDilationSizes do
    throw <| IO.userError
      s!"KittenTTS istftnet resblock_dilation_sizes={cfg.generator.resblockDilationSizes} does not match the Kokoro v1 checkpoint ({model.generator.resblockDilationSizes})"
  unless cfg.plbert.layerNormEps == model.plbert.layerNormEps do
    throw <| IO.userError
      s!"KittenTTS plbert.layer_norm_eps={cfg.plbert.layerNormEps} does not match the Kokoro v1 checkpoint ({model.plbert.layerNormEps})"
  warnShapeOverride log "hiddenDim" cfg.hiddenDim model.hiddenDim
  warnShapeOverride log "nToken" cfg.nToken model.nToken
  warnShapeOverride log "textEncoderKernelSize" cfg.textEncoderKernelSize model.textEncoderKernelSize
  warnShapeOverride log "nLayer" cfg.nLayer model.nLayer
  warnShapeOverride log "maxDur" cfg.maxDur model.maxDur
  warnShapeOverride log "styleDim" cfg.styleDim model.styleDim
  warnShapeOverride log "asrResDim" cfg.asrResDim model.asrResDim
  warnShapeOverride log "maxConvDim" cfg.maxConvDim model.maxConvDim
  warnShapeOverride log "decoderOutDim" (KittenTTSConfig.decoderChannels cfg) (KittenTTSConfig.decoderChannels model)
  warnShapeOverride log "generator.upsampleInitialChannel" cfg.generator.upsampleInitialChannel model.generator.upsampleInitialChannel
  warnShapeOverride log "generator.genIstftNFft" cfg.generator.genIstftNFft model.generator.genIstftNFft
  warnShapeOverride log "generator.harmonicCount" cfg.generator.harmonicCount model.generator.harmonicCount

private def buildModelFromCheckpointWeights
    (weights : LoadedCheckpointWeights)
    (log : Handlers := {})
    (device : Device := .CPU)
    : IO (Model (device := device) kokoro) := do
  let bert ← loadKittenAlbertFromCheckpoint weights.bert
  let bertEncoder : LinearNorm (device := device) kokoro.plbert.hiddenSize kokoro.hiddenDim :=
    linearFromTensors (← onDevice weights.bertEncoder.weight) (← onDevice weights.bertEncoder.bias)
  let predictor ← loadProsodyPredictorFromCheckpoint weights.predictor log
  let textEncoder ← loadTextEncoderFromCheckpoint weights.textEncoder
  let decoder ← loadDecoderFromCheckpoint weights.decoder
  log.onInfo s!"KittenTTS target device: {repr device}"
  pure { bert, bertEncoder, predictor, textEncoder, decoder }

namespace Model

structure Loaded where
  cfg : KittenTTSConfig
  device : Device
  model : Model (device := device) cfg

def loadAutoConfig (path : String) (cfg : KittenTTSConfig := {}) (log : Handlers := {})
    : IO Loaded := do
  log.onInfo s!"Loading KittenTTS weights from {path}..."
  checkKokoroV1 cfg log
  let weights ← loadCheckpointWeights path log
  let device ← resolveKittenDevice log
  let model ← buildModelFromCheckpointWeights weights log device
  pure { cfg := kokoro, device, model }

end Model

end torch.kittentts
