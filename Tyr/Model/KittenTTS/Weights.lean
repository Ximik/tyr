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

/-- Check fixed-schema weights against the runtime model dimensions without reshaping them. -/
private def checkedWeight {source target : Shape} {sourceDevice : Device}
    (tensor : Tensor { shape := source, dtype := .Float32, device := sourceDevice }) :
    IO (Tensor { shape := target, dtype := .Float32, device := device }) := do
  let raw ← RawTensor.convertIO source tensor.raw device none
  match Tensor.ofRaw raw with
  | .ok tensor => pure tensor
  | .error message => throw <| IO.userError s!"KittenTTS checkpoint shape: {message}"

private def reqGradFalse {s : Shape} (t : Tensor { shape := s, dtype := .Float32, device := device }) : Tensor { shape := s, dtype := .Float32, device := device } :=
  Tensor.detach t

private def getOrThrow {α} (xs : Array α) (idx : Nat) (msg : String) : IO α :=
  match xs[idx]? with
  | some x => pure x
  | none => throw <| IO.userError msg

private def tensorDimOrThrow {σ : StaticSpec} (t : Tensor σ) (idx : Nat) (name : String) : IO Nat :=
  getOrThrow t.runtimeShape idx s!"Missing dimension {idx} for {name}"

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
  let denomE : Tensor { shape := #[outC, inC, kernel], dtype := .Float32, device := device } := Tensor.expand denom #[outC, inC, kernel]
  let gain : Tensor { shape := #[outC, 1, 1], dtype := .Float32, device := device } := Tensor.reshape weightG #[outC, 1, 1]
  let gainE : Tensor { shape := #[outC, inC, kernel], dtype := .Float32, device := device } := Tensor.expand gain #[outC, inC, kernel]
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
macro "loadGenResUnit!" blk:term "," name:term "," k:term : term => `(
  loadGenResUnit ($blk).convs1 ($blk).convs2 ($blk).adain1 ($blk).adain2 ($blk).alpha1 ($blk).alpha2
    (fun node => do pure (wnConv1dFromWeightNorm (← checkedWeight node.weight_v) (← checkedWeight node.weight_g) (← checkedWeight node.bias)))
    (fun node => do pure (wnConv1dFromWeightNorm (← checkedWeight node.weight_v) (← checkedWeight node.weight_g) (← checkedWeight node.bias)))
    (fun node => do adaIN1dFromLinear (← checkedWeight node.fc.weight) (← checkedWeight node.fc.bias))
    (fun node => do adaIN1dFromLinear (← checkedWeight node.fc.weight) (← checkedWeight node.fc.bias))
    (fun alpha => do pure (reqGradFalse (← checkedWeight alpha)))
    (fun alpha => do pure (reqGradFalse (← checkedWeight alpha)))
    $name $k)

/-- Load a three-unit generator residual block from its checkpoint node arrays. -/
macro "loadGenResBlock!" blk:term "," name:term : term => `(do
  pure {
    u0 := ← loadGenResUnit! $blk, $name, 0
    u1 := ← loadGenResUnit! $blk, $name, 1
    u2 := ← loadGenResUnit! $blk, $name, 2
  })

private def loadKittenAlbertFromCheckpoint
    (weights : KokoroCheckpoint.WeightsBert)
    (cfg : KittenTTSConfig)
    : IO (KittenAlbert (device := device) cfg.nToken cfg.plbert) := do
  let embeddings : AlbertEmbeddings (device := device) cfg.nToken cfg.plbert := {
    wordEmbeddings := reqGradFalse (← checkedWeight weights.embeddings.word_embeddings.weight)
    positionEmbeddings := reqGradFalse (← checkedWeight weights.embeddings.position_embeddings.weight)
    tokenTypeEmbeddings := reqGradFalse (← checkedWeight weights.embeddings.token_type_embeddings.weight)
    layerNormWeight := reqGradFalse (← checkedWeight weights.embeddings.layernorm.weight)
    layerNormBias := reqGradFalse (← checkedWeight weights.embeddings.layernorm.bias)
  }
  let groups : Array (KittenAlbertLayerGroup (device := device) cfg.plbert) ←
    weights.encoder.albert_layer_groups.mapM fun group => do
      let layers ← group.albert_layers.mapM fun layer => do pure {
        attention := {
          query := linearFromTensors (← checkedWeight layer.attention.query.weight) (← checkedWeight layer.attention.query.bias)
          key := linearFromTensors (← checkedWeight layer.attention.key.weight) (← checkedWeight layer.attention.key.bias)
          value := linearFromTensors (← checkedWeight layer.attention.value.weight) (← checkedWeight layer.attention.value.bias)
          dense := linearFromTensors (← checkedWeight layer.attention.dense.weight) (← checkedWeight layer.attention.dense.bias)
          layerNormWeight := reqGradFalse (← checkedWeight layer.attention.layernorm.weight)
          layerNormBias := reqGradFalse (← checkedWeight layer.attention.layernorm.bias)
        }
        fullLayerNormWeight := reqGradFalse (← checkedWeight layer.full_layer_layer_norm.weight)
        fullLayerNormBias := reqGradFalse (← checkedWeight layer.full_layer_layer_norm.bias)
        ffn := linearFromTensors (← checkedWeight layer.ffn.weight) (← checkedWeight layer.ffn.bias)
        ffnOutput := linearFromTensors (← checkedWeight layer.ffn_output.weight) (← checkedWeight layer.ffn_output.bias)
      }
      pure { layers }
  let ⟨hGroups⟩ ← if h : groups.size = cfg.plbert.numHiddenGroups then pure (PLift.up h) else
    throw <| IO.userError
      s!"KittenTTS checkpoint has {groups.size} ALBERT layer groups, config num_hidden_groups={cfg.plbert.numHiddenGroups}"
  let encoder : KittenAlbertEncoder (device := device) cfg.plbert := {
    embeddingHiddenMappingIn :=
      linearFromTensors
        (← checkedWeight weights.encoder.embedding_hidden_mapping_in.weight)
        (← checkedWeight weights.encoder.embedding_hidden_mapping_in.bias)
    groups := ⟨groups, hGroups⟩
  }
  let pooler : LinearNorm (device := device) cfg.plbert.hiddenSize cfg.plbert.hiddenSize :=
    linearFromTensors (← checkedWeight weights.pooler.weight) (← checkedWeight weights.pooler.bias)
  pure { embeddings, encoder, pooler }

private def loadTextEncoderFromCheckpoint
    (weights : KokoroCheckpoint.WeightsTextencoder)
    (cfg : KittenTTSConfig)
    : IO (TextEncoder (device := device) cfg) := do
  let convs : Array (TextConvLayer (device := device) cfg) ←
    weights.cnn.mapM fun layer => do pure {
      conv := conv1dFromWeightNorm (← checkedWeight layer.i0.weight_v) (← checkedWeight layer.i0.weight_g) (← checkedWeight layer.i0.bias)
      lnWeight := reqGradFalse (← checkedWeight layer.i1.gamma)
      lnBias := reqGradFalse (← checkedWeight layer.i1.beta)
    }
  let lstm :=
    biLSTMFromTensors
      (← checkedWeight weights.lstm.weight_ih_l0)
      (← checkedWeight weights.lstm.weight_hh_l0)
      (← checkedWeight weights.lstm.bias_ih_l0)
      (← checkedWeight weights.lstm.bias_hh_l0)
      (← checkedWeight weights.lstm.weight_ih_l0_reverse)
      (← checkedWeight weights.lstm.weight_hh_l0_reverse)
      (← checkedWeight weights.lstm.bias_ih_l0_reverse)
      (← checkedWeight weights.lstm.bias_hh_l0_reverse)
  pure {
    embedding := reqGradFalse (← checkedWeight weights.embedding.weight)
    convs
    lstm
  }

private def loadDurationEncoderFromCheckpoint
    (weights : KokoroCheckpoint.WeightsPredictorTextencoder)
    (cfg : KittenTTSConfig)
    : IO (DurationEncoder (device := device) cfg) := do
  let ⟨hLayers⟩ ← if h : cfg.nLayer = 3 then pure (PLift.up h) else
    throw <| IO.userError
      s!"KittenTTS checkpoint has 3 duration-encoder layers, config n_layer={cfg.nLayer}"
  pure {
      lstms := ⟨#[
        biLSTMFromTensors
          (← checkedWeight weights.lstms.i0.weight_ih_l0)
          (← checkedWeight weights.lstms.i0.weight_hh_l0)
          (← checkedWeight weights.lstms.i0.bias_ih_l0)
          (← checkedWeight weights.lstms.i0.bias_hh_l0)
          (← checkedWeight weights.lstms.i0.weight_ih_l0_reverse)
          (← checkedWeight weights.lstms.i0.weight_hh_l0_reverse)
          (← checkedWeight weights.lstms.i0.bias_ih_l0_reverse)
          (← checkedWeight weights.lstms.i0.bias_hh_l0_reverse),
        biLSTMFromTensors
          (← checkedWeight weights.lstms.i2.weight_ih_l0)
          (← checkedWeight weights.lstms.i2.weight_hh_l0)
          (← checkedWeight weights.lstms.i2.bias_ih_l0)
          (← checkedWeight weights.lstms.i2.bias_hh_l0)
          (← checkedWeight weights.lstms.i2.weight_ih_l0_reverse)
          (← checkedWeight weights.lstms.i2.weight_hh_l0_reverse)
          (← checkedWeight weights.lstms.i2.bias_ih_l0_reverse)
          (← checkedWeight weights.lstms.i2.bias_hh_l0_reverse),
        biLSTMFromTensors
          (← checkedWeight weights.lstms.i4.weight_ih_l0)
          (← checkedWeight weights.lstms.i4.weight_hh_l0)
          (← checkedWeight weights.lstms.i4.bias_ih_l0)
          (← checkedWeight weights.lstms.i4.bias_hh_l0)
          (← checkedWeight weights.lstms.i4.weight_ih_l0_reverse)
          (← checkedWeight weights.lstms.i4.weight_hh_l0_reverse)
          (← checkedWeight weights.lstms.i4.bias_ih_l0_reverse)
          (← checkedWeight weights.lstms.i4.bias_hh_l0_reverse)
      ], by rw [hLayers]; rfl⟩
      norms := ⟨#[
        adaLayerNormFromLinear (← checkedWeight weights.lstms.i1.fc.weight) (← checkedWeight weights.lstms.i1.fc.bias),
        adaLayerNormFromLinear (← checkedWeight weights.lstms.i3.fc.weight) (← checkedWeight weights.lstms.i3.fc.bias),
        adaLayerNormFromLinear (← checkedWeight weights.lstms.i5.fc.weight) (← checkedWeight weights.lstms.i5.fc.bias)
      ], by rw [hLayers]; rfl⟩
    }

private def loadProsodyPredictorFromCheckpoint
    (weights : KokoroCheckpoint.WeightsPredictor)
    (cfg : KittenTTSConfig)
    (log : Handlers := {})
    : IO (ProsodyPredictor (device := device) cfg) := do
  log.onInfo "  predictor.text_encoder..."
  let durationEncoder ← loadDurationEncoderFromCheckpoint weights.text_encoder cfg
  log.onInfo "  predictor.lstm..."
  let lstm :=
    biLSTMFromTensors
      (← checkedWeight weights.lstm.weight_ih_l0)
      (← checkedWeight weights.lstm.weight_hh_l0)
      (← checkedWeight weights.lstm.bias_ih_l0)
      (← checkedWeight weights.lstm.bias_hh_l0)
      (← checkedWeight weights.lstm.weight_ih_l0_reverse)
      (← checkedWeight weights.lstm.weight_hh_l0_reverse)
      (← checkedWeight weights.lstm.bias_ih_l0_reverse)
      (← checkedWeight weights.lstm.bias_hh_l0_reverse)
  log.onInfo "  predictor.duration_proj..."
  let durationProj : LinearNorm (device := device) cfg.hiddenDim cfg.maxDur :=
    linearFromTensors
      (← checkedWeight weights.duration_proj.linear_layer.weight)
      (← checkedWeight weights.duration_proj.linear_layer.bias)
  log.onInfo "  predictor.shared..."
  let shared :=
    biLSTMFromTensors
      (← checkedWeight weights.shared.weight_ih_l0)
      (← checkedWeight weights.shared.weight_hh_l0)
      (← checkedWeight weights.shared.bias_ih_l0)
      (← checkedWeight weights.shared.bias_hh_l0)
      (← checkedWeight weights.shared.weight_ih_l0_reverse)
      (← checkedWeight weights.shared.weight_hh_l0_reverse)
      (← checkedWeight weights.shared.bias_ih_l0_reverse)
      (← checkedWeight weights.shared.bias_hh_l0_reverse)
  log.onInfo "  predictor.F0 blocks..."
  let f0Blk0 : AdainResBlk1dSame (device := device) cfg.hiddenDim cfg.hiddenDim cfg.styleDim := {
    conv1 := wnConv1dFromWeightNorm (← checkedWeight weights.f0.i0.conv1.weight_v) (← checkedWeight weights.f0.i0.conv1.weight_g) (← checkedWeight weights.f0.i0.conv1.bias)
    conv2 := wnConv1dFromWeightNorm (← checkedWeight weights.f0.i0.conv2.weight_v) (← checkedWeight weights.f0.i0.conv2.weight_g) (← checkedWeight weights.f0.i0.conv2.bias)
    norm1 := ← adaIN1dFromLinear (← checkedWeight weights.f0.i0.norm1.fc.weight) (← checkedWeight weights.f0.i0.norm1.fc.bias)
    norm2 := ← adaIN1dFromLinear (← checkedWeight weights.f0.i0.norm2.fc.weight) (← checkedWeight weights.f0.i0.norm2.fc.bias)
    shortcut := .identity rfl
  }
  let f0Blk1 : AdainResBlk1dUp (device := device) cfg.hiddenDim (cfg.hiddenDim / 2) cfg.styleDim := {
    pool := wnDepthwiseTransConv1dFromWeightNorm (← checkedWeight weights.f0.i1.pool.weight_v) (← checkedWeight weights.f0.i1.pool.weight_g) (← checkedWeight weights.f0.i1.pool.bias)
    conv1 := wnConv1dFromWeightNorm (← checkedWeight weights.f0.i1.conv1.weight_v) (← checkedWeight weights.f0.i1.conv1.weight_g) (← checkedWeight weights.f0.i1.conv1.bias)
    conv2 := wnConv1dFromWeightNorm (← checkedWeight weights.f0.i1.conv2.weight_v) (← checkedWeight weights.f0.i1.conv2.weight_g) (← checkedWeight weights.f0.i1.conv2.bias)
    norm1 := ← adaIN1dFromLinear (← checkedWeight weights.f0.i1.norm1.fc.weight) (← checkedWeight weights.f0.i1.norm1.fc.bias)
    norm2 := ← adaIN1dFromLinear (← checkedWeight weights.f0.i1.norm2.fc.weight) (← checkedWeight weights.f0.i1.norm2.fc.bias)
    shortcut := .conv <| wnConv1dFromWeightNorm (← checkedWeight weights.f0.i1.conv1x1.weight_v) (← checkedWeight weights.f0.i1.conv1x1.weight_g) (Tensor.zeros #[cfg.hiddenDim / 2] (device := device))
  }
  let f0Blk2 : AdainResBlk1dSame (device := device) (cfg.hiddenDim / 2) (cfg.hiddenDim / 2) cfg.styleDim := {
    conv1 := wnConv1dFromWeightNorm (← checkedWeight weights.f0.i2.conv1.weight_v) (← checkedWeight weights.f0.i2.conv1.weight_g) (← checkedWeight weights.f0.i2.conv1.bias)
    conv2 := wnConv1dFromWeightNorm (← checkedWeight weights.f0.i2.conv2.weight_v) (← checkedWeight weights.f0.i2.conv2.weight_g) (← checkedWeight weights.f0.i2.conv2.bias)
    norm1 := ← adaIN1dFromLinear (← checkedWeight weights.f0.i2.norm1.fc.weight) (← checkedWeight weights.f0.i2.norm1.fc.bias)
    norm2 := ← adaIN1dFromLinear (← checkedWeight weights.f0.i2.norm2.fc.weight) (← checkedWeight weights.f0.i2.norm2.fc.bias)
    shortcut := .identity rfl
  }
  log.onInfo "  predictor.N blocks..."
  let nBlk0 : AdainResBlk1dSame (device := device) cfg.hiddenDim cfg.hiddenDim cfg.styleDim := {
    conv1 := wnConv1dFromWeightNorm (← checkedWeight weights.n.i0.conv1.weight_v) (← checkedWeight weights.n.i0.conv1.weight_g) (← checkedWeight weights.n.i0.conv1.bias)
    conv2 := wnConv1dFromWeightNorm (← checkedWeight weights.n.i0.conv2.weight_v) (← checkedWeight weights.n.i0.conv2.weight_g) (← checkedWeight weights.n.i0.conv2.bias)
    norm1 := ← adaIN1dFromLinear (← checkedWeight weights.n.i0.norm1.fc.weight) (← checkedWeight weights.n.i0.norm1.fc.bias)
    norm2 := ← adaIN1dFromLinear (← checkedWeight weights.n.i0.norm2.fc.weight) (← checkedWeight weights.n.i0.norm2.fc.bias)
    shortcut := .identity rfl
  }
  let nBlk1 : AdainResBlk1dUp (device := device) cfg.hiddenDim (cfg.hiddenDim / 2) cfg.styleDim := {
    pool := wnDepthwiseTransConv1dFromWeightNorm (← checkedWeight weights.n.i1.pool.weight_v) (← checkedWeight weights.n.i1.pool.weight_g) (← checkedWeight weights.n.i1.pool.bias)
    conv1 := wnConv1dFromWeightNorm (← checkedWeight weights.n.i1.conv1.weight_v) (← checkedWeight weights.n.i1.conv1.weight_g) (← checkedWeight weights.n.i1.conv1.bias)
    conv2 := wnConv1dFromWeightNorm (← checkedWeight weights.n.i1.conv2.weight_v) (← checkedWeight weights.n.i1.conv2.weight_g) (← checkedWeight weights.n.i1.conv2.bias)
    norm1 := ← adaIN1dFromLinear (← checkedWeight weights.n.i1.norm1.fc.weight) (← checkedWeight weights.n.i1.norm1.fc.bias)
    norm2 := ← adaIN1dFromLinear (← checkedWeight weights.n.i1.norm2.fc.weight) (← checkedWeight weights.n.i1.norm2.fc.bias)
    shortcut := .conv <| wnConv1dFromWeightNorm (← checkedWeight weights.n.i1.conv1x1.weight_v) (← checkedWeight weights.n.i1.conv1x1.weight_g) (Tensor.zeros #[cfg.hiddenDim / 2] (device := device))
  }
  let nBlk2 : AdainResBlk1dSame (device := device) (cfg.hiddenDim / 2) (cfg.hiddenDim / 2) cfg.styleDim := {
    conv1 := wnConv1dFromWeightNorm (← checkedWeight weights.n.i2.conv1.weight_v) (← checkedWeight weights.n.i2.conv1.weight_g) (← checkedWeight weights.n.i2.conv1.bias)
    conv2 := wnConv1dFromWeightNorm (← checkedWeight weights.n.i2.conv2.weight_v) (← checkedWeight weights.n.i2.conv2.weight_g) (← checkedWeight weights.n.i2.conv2.bias)
    norm1 := ← adaIN1dFromLinear (← checkedWeight weights.n.i2.norm1.fc.weight) (← checkedWeight weights.n.i2.norm1.fc.bias)
    norm2 := ← adaIN1dFromLinear (← checkedWeight weights.n.i2.norm2.fc.weight) (← checkedWeight weights.n.i2.norm2.fc.bias)
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
      weight := reqGradFalse (← checkedWeight weights.f0_proj.weight)
      bias := reqGradFalse (← checkedWeight weights.f0_proj.bias)
    }
    nProj := {
      weight := reqGradFalse (← checkedWeight weights.n_proj.weight)
      bias := reqGradFalse (← checkedWeight weights.n_proj.bias)
    }
  }

private def loadGeneratorFromCheckpoint
    (weights : KokoroCheckpoint.WeightsDecoderGenerator)
    (cfg : KittenTTSConfig)
    : IO (Generator (device := device) cfg) := do
  let source : SourceModuleHnNSF (device := device) cfg := {
    linear := linearFromTensors (← checkedWeight weights.m_source.l_linear.weight) (← checkedWeight weights.m_source.l_linear.bias)
  }
  -- The Kokoro checkpoint schema has two upsample stages with three residual blocks each.
  let ⟨hStages⟩ ← if h : cfg.generator.upsampleRates.size = 2 then pure (PLift.up h) else
    throw <| IO.userError s!"Kokoro checkpoints have 2 generator upsample stages, config has {cfg.generator.upsampleRates.size}"
  let ⟨hBlocks⟩ ← if h : cfg.generator.resblockKernelSizes.size = 3 then pure (PLift.up h) else
    throw <| IO.userError s!"Kokoro checkpoints have 3 generator residual blocks per stage, config has {cfg.generator.resblockKernelSizes.size}"
  let stage0 : GeneratorStage (device := device) cfg 0 := {
    up := wnConvTranspose1dFromWeightNorm (← checkedWeight weights.ups.i0.weight_v) (← checkedWeight weights.ups.i0.weight_g) (← checkedWeight weights.ups.i0.bias)
    noiseConv := wnConv1dFromPlain (← checkedWeight weights.noise_convs.i0.weight) (← checkedWeight weights.noise_convs.i0.bias)
    noiseRes := ← loadGenResBlock! weights.noise_res.i0, "decoder.generator.noise_res.0"
    resBlocks :=
      .cons (by omega) (← loadGenResBlock! weights.resblocks.i0, "decoder.generator.resblocks.0")
        (.cons (by omega) (← loadGenResBlock! weights.resblocks.i1, "decoder.generator.resblocks.1")
          (.last (by omega) (← loadGenResBlock! weights.resblocks.i2, "decoder.generator.resblocks.2")))
  }
  let stage1 : GeneratorStage (device := device) cfg 1 := {
    up := wnConvTranspose1dFromWeightNorm (← checkedWeight weights.ups.i1.weight_v) (← checkedWeight weights.ups.i1.weight_g) (← checkedWeight weights.ups.i1.bias)
    noiseConv := wnConv1dFromPlain (← checkedWeight weights.noise_convs.i1.weight) (← checkedWeight weights.noise_convs.i1.bias)
    noiseRes := ← loadGenResBlock! weights.noise_res.i1, "decoder.generator.noise_res.1"
    resBlocks :=
      .cons (by omega) (← loadGenResBlock! weights.resblocks.i3, "decoder.generator.resblocks.3")
        (.cons (by omega) (← loadGenResBlock! weights.resblocks.i4, "decoder.generator.resblocks.4")
          (.last (by omega) (← loadGenResBlock! weights.resblocks.i5, "decoder.generator.resblocks.5")))
  }
  let convPost := wnConv1dFromWeightNorm (← checkedWeight weights.conv_post.weight_v) (← checkedWeight weights.conv_post.weight_g) (← checkedWeight weights.conv_post.bias)
  pure {
    source
    stages := .cons (by omega) stage0 (.last (by omega) stage1)
    convPost
    window := ← checkedWeight (Tensor.hannWindow cfg.generator.genIstftNFft)
  }

private def loadDecoderFromCheckpoint
    (weights : KokoroCheckpoint.WeightsDecoder)
    (cfg : KittenTTSConfig)
    : IO (Decoder (device := device) cfg) := do
  let asrResNode ← getOrThrow weights.asr_res 0 "Missing decoder.asr_res.0"
  let generator ← loadGeneratorFromCheckpoint weights.generator cfg
  let encode : AdainResBlk1dSame (device := device) (cfg.hiddenDim + 2) cfg.maxConvDim cfg.styleDim := {
    conv1 := wnConv1dFromWeightNorm (← checkedWeight weights.encode.conv1.weight_v) (← checkedWeight weights.encode.conv1.weight_g) (← checkedWeight weights.encode.conv1.bias)
    conv2 := wnConv1dFromWeightNorm (← checkedWeight weights.encode.conv2.weight_v) (← checkedWeight weights.encode.conv2.weight_g) (← checkedWeight weights.encode.conv2.bias)
    norm1 := ← adaIN1dFromLinear (← checkedWeight weights.encode.norm1.fc.weight) (← checkedWeight weights.encode.norm1.fc.bias)
    norm2 := ← adaIN1dFromLinear (← checkedWeight weights.encode.norm2.fc.weight) (← checkedWeight weights.encode.norm2.fc.bias)
    shortcut := .conv <| wnConv1dFromWeightNorm (← checkedWeight weights.encode.conv1x1.weight_v) (← checkedWeight weights.encode.conv1x1.weight_g) (Tensor.zeros #[cfg.maxConvDim] (device := device))
  }
  let decode0 : AdainResBlk1dSame (device := device) (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim := {
    conv1 := wnConv1dFromWeightNorm (← checkedWeight weights.decode.i0.conv1.weight_v) (← checkedWeight weights.decode.i0.conv1.weight_g) (← checkedWeight weights.decode.i0.conv1.bias)
    conv2 := wnConv1dFromWeightNorm (← checkedWeight weights.decode.i0.conv2.weight_v) (← checkedWeight weights.decode.i0.conv2.weight_g) (← checkedWeight weights.decode.i0.conv2.bias)
    norm1 := ← adaIN1dFromLinear (← checkedWeight weights.decode.i0.norm1.fc.weight) (← checkedWeight weights.decode.i0.norm1.fc.bias)
    norm2 := ← adaIN1dFromLinear (← checkedWeight weights.decode.i0.norm2.fc.weight) (← checkedWeight weights.decode.i0.norm2.fc.bias)
    shortcut := .conv <| wnConv1dFromWeightNorm (← checkedWeight weights.decode.i0.conv1x1.weight_v) (← checkedWeight weights.decode.i0.conv1x1.weight_g) (Tensor.zeros #[cfg.maxConvDim] (device := device))
  }
  let decode1 : AdainResBlk1dSame (device := device) (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim := {
    conv1 := wnConv1dFromWeightNorm (← checkedWeight weights.decode.i1.conv1.weight_v) (← checkedWeight weights.decode.i1.conv1.weight_g) (← checkedWeight weights.decode.i1.conv1.bias)
    conv2 := wnConv1dFromWeightNorm (← checkedWeight weights.decode.i1.conv2.weight_v) (← checkedWeight weights.decode.i1.conv2.weight_g) (← checkedWeight weights.decode.i1.conv2.bias)
    norm1 := ← adaIN1dFromLinear (← checkedWeight weights.decode.i1.norm1.fc.weight) (← checkedWeight weights.decode.i1.norm1.fc.bias)
    norm2 := ← adaIN1dFromLinear (← checkedWeight weights.decode.i1.norm2.fc.weight) (← checkedWeight weights.decode.i1.norm2.fc.bias)
    shortcut := .conv <| wnConv1dFromWeightNorm (← checkedWeight weights.decode.i1.conv1x1.weight_v) (← checkedWeight weights.decode.i1.conv1x1.weight_g) (Tensor.zeros #[cfg.maxConvDim] (device := device))
  }
  let decode2 : AdainResBlk1dSame (device := device) (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim := {
    conv1 := wnConv1dFromWeightNorm (← checkedWeight weights.decode.i2.conv1.weight_v) (← checkedWeight weights.decode.i2.conv1.weight_g) (← checkedWeight weights.decode.i2.conv1.bias)
    conv2 := wnConv1dFromWeightNorm (← checkedWeight weights.decode.i2.conv2.weight_v) (← checkedWeight weights.decode.i2.conv2.weight_g) (← checkedWeight weights.decode.i2.conv2.bias)
    norm1 := ← adaIN1dFromLinear (← checkedWeight weights.decode.i2.norm1.fc.weight) (← checkedWeight weights.decode.i2.norm1.fc.bias)
    norm2 := ← adaIN1dFromLinear (← checkedWeight weights.decode.i2.norm2.fc.weight) (← checkedWeight weights.decode.i2.norm2.fc.bias)
    shortcut := .conv <| wnConv1dFromWeightNorm (← checkedWeight weights.decode.i2.conv1x1.weight_v) (← checkedWeight weights.decode.i2.conv1x1.weight_g) (Tensor.zeros #[cfg.maxConvDim] (device := device))
  }
  let decode3 : AdainResBlk1dUp (device := device) (cfg.maxConvDim + cfg.asrResDim + 2) (KittenTTSConfig.decoderChannels cfg) cfg.styleDim := {
    pool := wnDepthwiseTransConv1dFromWeightNorm (← checkedWeight weights.decode.i3.pool.weight_v) (← checkedWeight weights.decode.i3.pool.weight_g) (← checkedWeight weights.decode.i3.pool.bias)
    conv1 := wnConv1dFromWeightNorm (← checkedWeight weights.decode.i3.conv1.weight_v) (← checkedWeight weights.decode.i3.conv1.weight_g) (← checkedWeight weights.decode.i3.conv1.bias)
    conv2 := wnConv1dFromWeightNorm (← checkedWeight weights.decode.i3.conv2.weight_v) (← checkedWeight weights.decode.i3.conv2.weight_g) (← checkedWeight weights.decode.i3.conv2.bias)
    norm1 := ← adaIN1dFromLinear (← checkedWeight weights.decode.i3.norm1.fc.weight) (← checkedWeight weights.decode.i3.norm1.fc.bias)
    norm2 := ← adaIN1dFromLinear (← checkedWeight weights.decode.i3.norm2.fc.weight) (← checkedWeight weights.decode.i3.norm2.fc.bias)
    shortcut := .conv <| wnConv1dFromWeightNorm (← checkedWeight weights.decode.i3.conv1x1.weight_v) (← checkedWeight weights.decode.i3.conv1x1.weight_g) (Tensor.zeros #[KittenTTSConfig.decoderChannels cfg] (device := device))
  }
  pure {
    f0Conv := wnConv1dFromWeightNorm (← checkedWeight weights.f0_conv.weight_v) (← checkedWeight weights.f0_conv.weight_g) (← checkedWeight weights.f0_conv.bias)
    nConv := wnConv1dFromWeightNorm (← checkedWeight weights.n_conv.weight_v) (← checkedWeight weights.n_conv.weight_g) (← checkedWeight weights.n_conv.bias)
    encode
    asrRes := wnConv1dFromWeightNorm (← checkedWeight asrResNode.weight_v) (← checkedWeight asrResNode.weight_g) (← checkedWeight asrResNode.bias)
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

private def reconcileConfigWithCheckpoint
    (cfg : KittenTTSConfig)
    (weights : LoadedCheckpointWeights)
    (log : Handlers := {})
    : IO KittenTTSConfig := do
  let textConv0 ← getOrThrow weights.textEncoder.cnn 0 "Missing text_encoder.cnn.0"
  let asrRes0 ← getOrThrow weights.decoder.asr_res 0 "Missing decoder.asr_res.0"
  let hiddenDim ← tensorDimOrThrow weights.bertEncoder.bias 0 "bert_encoder.bias"
  let nToken ← tensorDimOrThrow weights.textEncoder.embedding.weight 0 "text_encoder.embedding.weight"
  let textEncoderKernelSize ← tensorDimOrThrow textConv0.i0.weight_v 2 "text_encoder.cnn.0.0.weight_v"
  let maxDur ← tensorDimOrThrow weights.predictor.duration_proj.linear_layer.bias 0 "predictor.duration_proj.linear_layer.bias"
  let styleDim ← tensorDimOrThrow weights.decoder.encode.norm1.fc.weight 1 "decoder.encode.norm1.fc.weight"
  let asrResDim ← tensorDimOrThrow asrRes0.bias 0 "decoder.asr_res.0.bias"
  let decoderBlockDim ← tensorDimOrThrow weights.decoder.encode.conv1.bias 0 "decoder.encode.conv1.bias"
  let decoderOutDim ← tensorDimOrThrow weights.decoder.decode.i3.conv2.bias 0 "decoder.decode.3.conv2.bias"
  let upsampleInitialChannel ← tensorDimOrThrow weights.decoder.generator.ups.i0.weight_g 0 "decoder.generator.ups.0.weight_g"
  let genIstftNFft ← do
    let convPostOut ← tensorDimOrThrow weights.decoder.generator.conv_post.bias 0 "decoder.generator.conv_post.bias"
    if convPostOut < 2 then
      throw <| IO.userError s!"Invalid decoder.generator.conv_post output dimension: {convPostOut}"
    pure (convPostOut - 2)
  let harmonicCount ← do
    let harmonicInputs ← tensorDimOrThrow weights.decoder.generator.m_source.l_linear.weight 1 "decoder.generator.m_source.l_linear.weight"
    if harmonicInputs == 0 then
      throw <| IO.userError "Invalid decoder.generator.m_source.l_linear.weight: expected at least one harmonic input"
    pure (harmonicInputs - 1)

  warnShapeOverride log "hiddenDim" cfg.hiddenDim hiddenDim
  warnShapeOverride log "nToken" cfg.nToken nToken
  warnShapeOverride log "textEncoderKernelSize" cfg.textEncoderKernelSize textEncoderKernelSize
  warnShapeOverride log "nLayer" cfg.nLayer weights.textEncoder.cnn.size
  warnShapeOverride log "maxDur" cfg.maxDur maxDur
  warnShapeOverride log "styleDim" cfg.styleDim styleDim
  warnShapeOverride log "asrResDim" cfg.asrResDim asrResDim
  warnShapeOverride log "maxConvDim" cfg.maxConvDim decoderBlockDim
  warnShapeOverride log "decoderOutDim" (KittenTTSConfig.decoderChannels cfg) decoderOutDim
  warnShapeOverride log "generator.upsampleInitialChannel" cfg.generator.upsampleInitialChannel upsampleInitialChannel
  warnShapeOverride log "generator.genIstftNFft" cfg.generator.genIstftNFft genIstftNFft
  warnShapeOverride log "generator.harmonicCount" cfg.generator.harmonicCount harmonicCount

  let generator : GeneratorConfig := {
    cfg.generator with
    upsampleInitialChannel := upsampleInitialChannel
    genIstftNFft := genIstftNFft
    harmonicCount := harmonicCount
  }
  let ⟨hHidden⟩ ← if h : 2 ∣ hiddenDim then pure (PLift.up h) else
    throw <| IO.userError s!"Checkpoint hidden dimension {hiddenDim} must be even"
  let ⟨hKernel⟩ ← if h : textEncoderKernelSize % 2 = 1 then pure (PLift.up h) else
    throw <| IO.userError s!"Checkpoint text encoder kernel {textEncoderKernelSize} must be odd"
  let ⟨hDecoder⟩ ← if h : (if decoderOutDim == 0 then generator.upsampleInitialChannel else decoderOutDim) =
      generator.upsampleInitialChannel then pure (PLift.up h) else
    throw <| IO.userError
      s!"Checkpoint decoder output {decoderOutDim} must match generator input {generator.upsampleInitialChannel}"
  pure {
    cfg with
    hiddenDim := hiddenDim
    maxConvDim := decoderBlockDim
    maxDur := maxDur
    nLayer := weights.textEncoder.cnn.size
    nToken := nToken
    styleDim := styleDim
    textEncoderKernelSize := textEncoderKernelSize
    asrResDim := asrResDim
    decoderOutDim := decoderOutDim
    generator := generator
    hiddenDimEven := hHidden
    textKernelOdd := hKernel
    decoderMatchesGenerator := hDecoder
  }

private def buildModelFromCheckpointWeights
    (weights : LoadedCheckpointWeights)
    (cfg : KittenTTSConfig)
    (log : Handlers := {})
    (device : Device := .CPU)
    : IO (Model (device := device) cfg) := do
  let bert ← loadKittenAlbertFromCheckpoint weights.bert cfg
  let bertEncoder : LinearNorm (device := device) cfg.plbert.hiddenSize cfg.hiddenDim :=
    linearFromTensors (← checkedWeight weights.bertEncoder.weight) (← checkedWeight weights.bertEncoder.bias)
  let predictor ← loadProsodyPredictorFromCheckpoint weights.predictor cfg log
  let textEncoder ← loadTextEncoderFromCheckpoint weights.textEncoder cfg
  let decoder ← loadDecoderFromCheckpoint weights.decoder cfg
  log.onInfo s!"KittenTTS target device: {repr device}"
  pure { bert, bertEncoder, predictor, textEncoder, decoder }

namespace Model

structure Loaded where
  cfg : KittenTTSConfig
  device : Device
  model : Model (device := device) cfg

/-- Load on an explicitly selected device, retained in the result type. -/
def loadAt (path : String) (device : Device) (cfg : KittenTTSConfig := {})
    (log : Handlers := {}) : IO (Model (device := device) cfg) := do
  log.onInfo s!"Loading KittenTTS weights from {path}..."
  let weights ← loadCheckpointWeights path log
  buildModelFromCheckpointWeights weights cfg log device

/-- Resolve the configured placement once and return it with the indexed model. -/
def load (path : String) (cfg : KittenTTSConfig := {}) (log : Handlers := {})
    : IO (Sigma fun device => Model (device := device) cfg) := do
  let device ← resolveKittenDevice log
  return ⟨device, ← loadAt path device cfg log⟩

def loadAutoConfig (path : String) (cfg : KittenTTSConfig := {}) (log : Handlers := {})
    : IO Loaded := do
  log.onInfo s!"Loading KittenTTS weights from {path}..."
  let weights ← loadCheckpointWeights path log
  let cfg' ← reconcileConfigWithCheckpoint cfg weights log
  let device ← resolveKittenDevice log
  let model ← buildModelFromCheckpointWeights weights cfg' log device
  pure { cfg := cfg', device, model }

end Model

end torch.kittentts
