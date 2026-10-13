/-
  Tyr/Model/KittenTTS/Weights.lean

  Pretrained weight loading for the Lean KittenTTS / Kokoro port.
  Loads converted SafeTensors checkpoints produced from upstream PyTorch
  checkpoints so single-utterance synthesis can use real weights.
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Log
import Tyr.Model.KittenTTS.Checkpoint
import Tyr.Model.KittenTTS.Model

namespace torch.kittentts

open torch.Log

-- Transitional boundary: generated checkpoint records are indexed Tensor values,
-- while KittenTTS.Model still uses the legacy API. Keep this conversion local
-- to loading; remove it when the model and its layers migrate to Tensor.
local instance {σ : StaticSpec} {s : Shape} (tensor : Tensor σ) :
    CoeDep (Tensor σ) tensor (T s) where
  coe := tensor.raw

private structure LoadedCheckpointWeights where
  bert : KokoroCheckpoint.WeightsBert
  bertEncoder : KokoroCheckpoint.WeightsBertencoder
  predictor : KokoroCheckpoint.WeightsPredictor
  textEncoder : KokoroCheckpoint.WeightsTextencoder
  decoder : KokoroCheckpoint.WeightsDecoder

private def reqGradFalse {s : Shape} (t : T s) : T s :=
  autograd.set_requires_grad t false

private def getOrThrow {α} (xs : Array α) (idx : Nat) (msg : String) : IO α :=
  match xs[idx]? with
  | some x => pure x
  | none => throw <| IO.userError msg

private def tensorDimOrThrow {σ : StaticSpec} (t : Tensor σ) (idx : Nat) (name : String) : IO UInt64 :=
  getOrThrow t.runtimeShape idx s!"Missing dimension {idx} for {name}"

private def warnShapeOverride (log : Handlers) (field : String) (cfgValue actualValue : UInt64) : IO Unit := do
  if cfgValue != actualValue then
    log.onWarn s!"Checkpoint overrides {field}: config={cfgValue}, checkpoint={actualValue}"
  pure ()

private def normDim0_3d {d0 d1 d2 : UInt64}
    (x : T #[d0, d1, d2])
    : T #[d0] :=
  let sq : T #[d0, d1, d2] := x * x
  let s2 : T #[d0, d1, 1] := nn.sumDim sq 2 true
  let s1 : T #[d0, 1, 1] := nn.sumDim s2 1 true
  reshape (nn.sqrt (s1 + (1e-12 : Float))) #[d0]

private def materializeWeightNormConv1d {outC inC kernel : UInt64}
    (weightV : T #[outC, inC, kernel])
    (weightG : T #[outC])
    : T #[outC, inC, kernel] :=
  let denom : T #[outC, 1, 1] := reshape (normDim0_3d weightV) #[outC, 1, 1]
  let denomE : T #[outC, inC, kernel] := nn.expand denom #[outC, inC, kernel]
  let gain : T #[outC, 1, 1] := reshape weightG #[outC, 1, 1]
  let gainE : T #[outC, inC, kernel] := nn.expand gain #[outC, inC, kernel]
  nn.div (weightV * gainE) denomE

private def linearFromTensors {inDim outDim : UInt64}
    (weight : T #[outDim, inDim])
    (bias : T #[outDim])
    : LinearNorm inDim outDim := {
      weight := reqGradFalse weight
      bias := reqGradFalse bias
    }

private def biLSTMFromTensors {inputSize hiddenSize : UInt64}
    (wxForward : T #[4 * hiddenSize, inputSize])
    (whForward : T #[4 * hiddenSize, hiddenSize])
    (biasIHForward : T #[4 * hiddenSize])
    (biasHHForward : T #[4 * hiddenSize])
    (wxBackward : T #[4 * hiddenSize, inputSize])
    (whBackward : T #[4 * hiddenSize, hiddenSize])
    (biasIHBackward : T #[4 * hiddenSize])
    (biasHHBackward : T #[4 * hiddenSize])
    : BiLSTM inputSize hiddenSize := {
      wxForward := reqGradFalse wxForward
      whForward := reqGradFalse whForward
      biasIHForward := reqGradFalse biasIHForward
      biasHHForward := reqGradFalse biasHHForward
      wxBackward := reqGradFalse wxBackward
      whBackward := reqGradFalse whBackward
      biasIHBackward := reqGradFalse biasIHBackward
      biasHHBackward := reqGradFalse biasHHBackward
    }

private def conv1dFromWeightNorm {inC outC kernel stride padding dilation : UInt64}
    (weightV : T #[outC, inC, kernel])
    (weightG3 : T #[outC, 1, 1])
    (bias : T #[outC])
    : Conv1dParams inC outC kernel stride padding dilation := {
      weight := reqGradFalse <| materializeWeightNormConv1d weightV (reshape weightG3 #[outC])
      bias := reqGradFalse bias
    }

private def wnConv1dFromWeightNorm {inC outC kernel stride padding dilation : UInt64}
    (weightV : T #[outC, inC, kernel])
    (weightG3 : T #[outC, 1, 1])
    (bias : T #[outC])
    : WNConv1dParams inC outC kernel stride padding dilation := {
      weightV := reqGradFalse weightV
      weightG := reqGradFalse (reshape weightG3 #[outC])
      bias := reqGradFalse bias
    }

private def wnDepthwiseTransConv1dFromWeightNorm
    {channels kernel stride padding outputPadding dilation : UInt64}
    (weightV : T #[channels, 1, kernel])
    (weightG3 : T #[channels, 1, 1])
    (bias : T #[channels])
    : WNDepthwiseTransConv1dParams channels kernel stride padding outputPadding dilation := {
      weightV := reqGradFalse weightV
      weightG := reqGradFalse (reshape weightG3 #[channels])
      bias := reqGradFalse bias
    }

private def wnConvTranspose1dFromWeightNorm
    {inC outC kernel stride padding outputPadding dilation : UInt64}
    (weightV : T #[inC, outC, kernel])
    (weightG3 : T #[inC, 1, 1])
    (bias : T #[outC])
    : WNConvTranspose1dParams inC outC kernel stride padding outputPadding dilation := {
      weightV := reqGradFalse weightV
      weightG := reqGradFalse (reshape weightG3 #[inC])
      bias := reqGradFalse bias
    }

private def adaLayerNormFromLinear {styleDim channels : UInt64}
    (weight : T #[2 * channels, styleDim])
    (bias : T #[2 * channels])
    : AdaLayerNorm styleDim channels := {
      fc := linearFromTensors weight bias
    }

private def adaIN1dFromLinear {styleDim channels : UInt64}
    (weight : T #[2 * channels, styleDim])
    (bias : T #[2 * channels])
    : AdaIN1d styleDim channels := {
      fc := linearFromTensors weight bias
      normWeight := castLike weight (reqGradFalse (torch.ones #[channels] false Device.CPU))
      normBias := castLike weight (reqGradFalse (torch.zeros #[channels] false Device.CPU))
    }

private def adaIN1dDynFromLinear {styleDim channels : UInt64}
    (weight : T #[2 * channels, styleDim])
    (bias : T #[2 * channels])
    : AdaIN1dDyn styleDim := {
      fcWeight := nn.eraseShape (reqGradFalse weight)
      fcBias := nn.eraseShape (reqGradFalse bias)
      normWeight := nn.eraseShape (castLike weight (torch.ones #[channels] false Device.CPU))
      normBias := nn.eraseShape (castLike weight (torch.zeros #[channels] false Device.CPU))
    }

private def wnConv1dDynFromWeightNorm {inC outC kernel : UInt64}
    (weightV : T #[outC, inC, kernel])
    (weightG3 : T #[outC, 1, 1])
    (bias : T #[outC])
    (stride padding dilation : UInt64)
    : WNConv1dDyn := {
      weightV := nn.eraseShape (reqGradFalse weightV)
      weightG := nn.eraseShape (reqGradFalse (reshape weightG3 #[outC]))
      bias := nn.eraseShape (reqGradFalse bias)
      kernel := kernel
      stride := stride
      padding := padding
      dilation := dilation
    }

private def wnConv1dDynFromPlain {inC outC kernel : UInt64}
    (weight : T #[outC, inC, kernel])
    (bias : T #[outC])
    (stride padding dilation : UInt64)
    : WNConv1dDyn := {
      weightV := nn.eraseShape (reqGradFalse weight)
      weightG := nn.eraseShape (reqGradFalse (normDim0_3d weight))
      bias := nn.eraseShape (reqGradFalse bias)
      kernel := kernel
      stride := stride
      padding := padding
      dilation := dilation
    }

private def wnConvTranspose1dDynFromWeightNorm {inC outC kernel : UInt64}
    (weightV : T #[inC, outC, kernel])
    (weightG3 : T #[inC, 1, 1])
    (bias : T #[outC])
    (stride padding outputPadding dilation : UInt64)
    : WNConvTranspose1dDyn := {
      weightV := nn.eraseShape (reqGradFalse weightV)
      weightG := nn.eraseShape (reqGradFalse (reshape weightG3 #[inC]))
      bias := nn.eraseShape (reqGradFalse bias)
      kernel := kernel
      stride := stride
      padding := padding
      outputPadding := outputPadding
      dilation := dilation
    }

private def alphaDyn {σ : StaticSpec} (x : Tensor σ) : T #[] :=
  nn.eraseShape (reqGradFalse (s := σ.shape) x.raw)

private def generatorAdaINResBlockFromParts {styleDim : UInt64}
    (convs1 : Array WNConv1dDyn)
    (convs2 : Array WNConv1dDyn)
    (norms1 : Array (AdaIN1dDyn styleDim))
    (norms2 : Array (AdaIN1dDyn styleDim))
    (alpha1 : Array (T #[]))
    (alpha2 : Array (T #[]))
    : GeneratorAdaINResBlock styleDim := {
      convs1, convs2, norms1, norms2, alpha1, alpha2
    }

private def mapArrayM {α β} (xs : Array α) (f : α → IO β) : IO (Array β) := do
  let mut ys : Array β := #[]
  for x in xs do
    ys := ys.push (← f x)
  pure ys

private def mapArrayIdxM {α β} (xs : Array α) (f : Nat → α → IO β) : IO (Array β) := do
  let mut ys : Array β := #[]
  for i in [:xs.size] do
    if let some x := xs[i]? then
      ys := ys.push (← f i x)
  pure ys

private def loadGeneratorAdaINResBlockFromCheckpointArrays
    {styleDim : UInt64}
    (convs1 : Array α)
    (convs2 : Array β)
    (adain1 : Array γ)
    (adain2 : Array δ)
    (alpha1 : Array ε)
    (alpha2 : Array ζ)
    (kernel : UInt64)
    (dilations : Array UInt64)
    (mkConv1 : α → UInt64 → UInt64 → WNConv1dDyn)
    (mkConv2 : β → WNConv1dDyn)
    (mkAdaIN1 : γ → AdaIN1dDyn styleDim)
    (mkAdaIN2 : δ → AdaIN1dDyn styleDim)
    (mkAlpha1 : ε → T #[])
    (mkAlpha2 : ζ → T #[])
    : IO (GeneratorAdaINResBlock styleDim) := do
  let convs1 ←
    mapArrayIdxM convs1 fun i node => do
      let dilation := dilations.getD i 1
      let padding := (kernel * dilation - dilation) / 2
      pure (mkConv1 node padding dilation)
  let convs2 ← mapArrayM convs2 fun node => pure (mkConv2 node)
  let norms1 ← mapArrayM adain1 fun node => pure (mkAdaIN1 node)
  let norms2 ← mapArrayM adain2 fun node => pure (mkAdaIN2 node)
  pure <|
    generatorAdaINResBlockFromParts
      convs1
      convs2
      norms1
      norms2
      (alpha1.map mkAlpha1)
      (alpha2.map mkAlpha2)

private def loadKittenAlbertFromCheckpoint
    (weights : KokoroCheckpoint.WeightsBert)
    (cfg : KittenTTSConfig)
    : IO (KittenAlbert cfg.nToken cfg.plbert) := do
  let embeddings : AlbertEmbeddings cfg.nToken cfg.plbert := {
    wordEmbeddings := reqGradFalse weights.embeddings.word_embeddings.weight
    positionEmbeddings := reqGradFalse weights.embeddings.position_embeddings.weight
    tokenTypeEmbeddings := reqGradFalse weights.embeddings.token_type_embeddings.weight
    layerNormWeight := reqGradFalse weights.embeddings.layernorm.weight
    layerNormBias := reqGradFalse weights.embeddings.layernorm.bias
  }
  let groups : Array (KittenAlbertLayerGroup cfg.plbert) :=
    weights.encoder.albert_layer_groups.map fun group => {
      layers := group.albert_layers.map fun layer => {
        attention := {
          query := linearFromTensors layer.attention.query.weight layer.attention.query.bias
          key := linearFromTensors layer.attention.key.weight layer.attention.key.bias
          value := linearFromTensors layer.attention.value.weight layer.attention.value.bias
          dense := linearFromTensors layer.attention.dense.weight layer.attention.dense.bias
          layerNormWeight := reqGradFalse layer.attention.layernorm.weight
          layerNormBias := reqGradFalse layer.attention.layernorm.bias
        }
        fullLayerNormWeight := reqGradFalse layer.full_layer_layer_norm.weight
        fullLayerNormBias := reqGradFalse layer.full_layer_layer_norm.bias
        ffn := linearFromTensors layer.ffn.weight layer.ffn.bias
        ffnOutput := linearFromTensors layer.ffn_output.weight layer.ffn_output.bias
      }
    }
  let encoder : KittenAlbertEncoder cfg.plbert := {
    embeddingHiddenMappingIn :=
      linearFromTensors
        weights.encoder.embedding_hidden_mapping_in.weight
        weights.encoder.embedding_hidden_mapping_in.bias
    groups := groups
  }
  let pooler : LinearNorm cfg.plbert.hiddenSize cfg.plbert.hiddenSize :=
    linearFromTensors weights.pooler.weight weights.pooler.bias
  pure { embeddings, encoder, pooler }

private def loadTextEncoderFromCheckpoint
    (weights : KokoroCheckpoint.WeightsTextencoder)
    (cfg : KittenTTSConfig)
    : IO (TextEncoder cfg) := do
  let convs : Array (TextConvLayer cfg) :=
    weights.cnn.map fun layer => {
      conv := conv1dFromWeightNorm layer.i0.weight_v layer.i0.weight_g layer.i0.bias
      lnWeight := reqGradFalse layer.i1.gamma
      lnBias := reqGradFalse layer.i1.beta
    }
  let lstm :=
    biLSTMFromTensors
      weights.lstm.weight_ih_l0
      weights.lstm.weight_hh_l0
      weights.lstm.bias_ih_l0
      weights.lstm.bias_hh_l0
      weights.lstm.weight_ih_l0_reverse
      weights.lstm.weight_hh_l0_reverse
      weights.lstm.bias_ih_l0_reverse
      weights.lstm.bias_hh_l0_reverse
  pure {
    embedding := reqGradFalse weights.embedding.weight
    convs
    lstm
  }

private def loadDurationEncoderFromCheckpoint
    (weights : KokoroCheckpoint.WeightsPredictorTextencoder)
    (cfg : KittenTTSConfig)
    : DurationEncoder cfg := {
      lstms := #[
        biLSTMFromTensors
          weights.lstms.i0.weight_ih_l0
          weights.lstms.i0.weight_hh_l0
          weights.lstms.i0.bias_ih_l0
          weights.lstms.i0.bias_hh_l0
          weights.lstms.i0.weight_ih_l0_reverse
          weights.lstms.i0.weight_hh_l0_reverse
          weights.lstms.i0.bias_ih_l0_reverse
          weights.lstms.i0.bias_hh_l0_reverse,
        biLSTMFromTensors
          weights.lstms.i2.weight_ih_l0
          weights.lstms.i2.weight_hh_l0
          weights.lstms.i2.bias_ih_l0
          weights.lstms.i2.bias_hh_l0
          weights.lstms.i2.weight_ih_l0_reverse
          weights.lstms.i2.weight_hh_l0_reverse
          weights.lstms.i2.bias_ih_l0_reverse
          weights.lstms.i2.bias_hh_l0_reverse,
        biLSTMFromTensors
          weights.lstms.i4.weight_ih_l0
          weights.lstms.i4.weight_hh_l0
          weights.lstms.i4.bias_ih_l0
          weights.lstms.i4.bias_hh_l0
          weights.lstms.i4.weight_ih_l0_reverse
          weights.lstms.i4.weight_hh_l0_reverse
          weights.lstms.i4.bias_ih_l0_reverse
          weights.lstms.i4.bias_hh_l0_reverse
      ]
      norms := #[
        adaLayerNormFromLinear weights.lstms.i1.fc.weight weights.lstms.i1.fc.bias,
        adaLayerNormFromLinear weights.lstms.i3.fc.weight weights.lstms.i3.fc.bias,
        adaLayerNormFromLinear weights.lstms.i5.fc.weight weights.lstms.i5.fc.bias
      ]
    }

private def loadProsodyPredictorFromCheckpoint
    (weights : KokoroCheckpoint.WeightsPredictor)
    (cfg : KittenTTSConfig)
    (log : Handlers := {})
    : IO (ProsodyPredictor cfg) := do
  log.onInfo "  predictor.text_encoder..."
  let durationEncoder := loadDurationEncoderFromCheckpoint weights.text_encoder cfg
  log.onInfo "  predictor.lstm..."
  let lstm :=
    biLSTMFromTensors
      weights.lstm.weight_ih_l0
      weights.lstm.weight_hh_l0
      weights.lstm.bias_ih_l0
      weights.lstm.bias_hh_l0
      weights.lstm.weight_ih_l0_reverse
      weights.lstm.weight_hh_l0_reverse
      weights.lstm.bias_ih_l0_reverse
      weights.lstm.bias_hh_l0_reverse
  log.onInfo "  predictor.duration_proj..."
  let durationProj : LinearNorm cfg.hiddenDim cfg.maxDur :=
    linearFromTensors
      weights.duration_proj.linear_layer.weight
      weights.duration_proj.linear_layer.bias
  log.onInfo "  predictor.shared..."
  let shared :=
    biLSTMFromTensors
      weights.shared.weight_ih_l0
      weights.shared.weight_hh_l0
      weights.shared.bias_ih_l0
      weights.shared.bias_hh_l0
      weights.shared.weight_ih_l0_reverse
      weights.shared.weight_hh_l0_reverse
      weights.shared.bias_ih_l0_reverse
      weights.shared.bias_hh_l0_reverse
  log.onInfo "  predictor.F0 blocks..."
  let f0Blk0 : AdainResBlk1dSame cfg.hiddenDim cfg.hiddenDim cfg.styleDim := {
    conv1 := wnConv1dFromWeightNorm weights.f0.i0.conv1.weight_v weights.f0.i0.conv1.weight_g weights.f0.i0.conv1.bias
    conv2 := wnConv1dFromWeightNorm weights.f0.i0.conv2.weight_v weights.f0.i0.conv2.weight_g weights.f0.i0.conv2.bias
    norm1 := adaIN1dFromLinear weights.f0.i0.norm1.fc.weight weights.f0.i0.norm1.fc.bias
    norm2 := adaIN1dFromLinear weights.f0.i0.norm2.fc.weight weights.f0.i0.norm2.fc.bias
    shortcut := none
  }
  let f0Blk1 : AdainResBlk1dUp cfg.hiddenDim (cfg.hiddenDim / 2) cfg.styleDim := {
    pool := wnDepthwiseTransConv1dFromWeightNorm weights.f0.i1.pool.weight_v weights.f0.i1.pool.weight_g weights.f0.i1.pool.bias
    conv1 := wnConv1dFromWeightNorm weights.f0.i1.conv1.weight_v weights.f0.i1.conv1.weight_g weights.f0.i1.conv1.bias
    conv2 := wnConv1dFromWeightNorm weights.f0.i1.conv2.weight_v weights.f0.i1.conv2.weight_g weights.f0.i1.conv2.bias
    norm1 := adaIN1dFromLinear weights.f0.i1.norm1.fc.weight weights.f0.i1.norm1.fc.bias
    norm2 := adaIN1dFromLinear weights.f0.i1.norm2.fc.weight weights.f0.i1.norm2.fc.bias
    shortcut := some <| wnConv1dFromWeightNorm weights.f0.i1.conv1x1.weight_v weights.f0.i1.conv1x1.weight_g (castLike (sRef := weights.f0.i1.conv1x1.weight_v.shape) weights.f0.i1.conv1x1.weight_v (torch.zeros #[cfg.hiddenDim / 2] false Device.CPU))
  }
  let f0Blk2 : AdainResBlk1dSame (cfg.hiddenDim / 2) (cfg.hiddenDim / 2) cfg.styleDim := {
    conv1 := wnConv1dFromWeightNorm weights.f0.i2.conv1.weight_v weights.f0.i2.conv1.weight_g weights.f0.i2.conv1.bias
    conv2 := wnConv1dFromWeightNorm weights.f0.i2.conv2.weight_v weights.f0.i2.conv2.weight_g weights.f0.i2.conv2.bias
    norm1 := adaIN1dFromLinear weights.f0.i2.norm1.fc.weight weights.f0.i2.norm1.fc.bias
    norm2 := adaIN1dFromLinear weights.f0.i2.norm2.fc.weight weights.f0.i2.norm2.fc.bias
    shortcut := none
  }
  log.onInfo "  predictor.N blocks..."
  let nBlk0 : AdainResBlk1dSame cfg.hiddenDim cfg.hiddenDim cfg.styleDim := {
    conv1 := wnConv1dFromWeightNorm weights.n.i0.conv1.weight_v weights.n.i0.conv1.weight_g weights.n.i0.conv1.bias
    conv2 := wnConv1dFromWeightNorm weights.n.i0.conv2.weight_v weights.n.i0.conv2.weight_g weights.n.i0.conv2.bias
    norm1 := adaIN1dFromLinear weights.n.i0.norm1.fc.weight weights.n.i0.norm1.fc.bias
    norm2 := adaIN1dFromLinear weights.n.i0.norm2.fc.weight weights.n.i0.norm2.fc.bias
    shortcut := none
  }
  let nBlk1 : AdainResBlk1dUp cfg.hiddenDim (cfg.hiddenDim / 2) cfg.styleDim := {
    pool := wnDepthwiseTransConv1dFromWeightNorm weights.n.i1.pool.weight_v weights.n.i1.pool.weight_g weights.n.i1.pool.bias
    conv1 := wnConv1dFromWeightNorm weights.n.i1.conv1.weight_v weights.n.i1.conv1.weight_g weights.n.i1.conv1.bias
    conv2 := wnConv1dFromWeightNorm weights.n.i1.conv2.weight_v weights.n.i1.conv2.weight_g weights.n.i1.conv2.bias
    norm1 := adaIN1dFromLinear weights.n.i1.norm1.fc.weight weights.n.i1.norm1.fc.bias
    norm2 := adaIN1dFromLinear weights.n.i1.norm2.fc.weight weights.n.i1.norm2.fc.bias
    shortcut := some <| wnConv1dFromWeightNorm weights.n.i1.conv1x1.weight_v weights.n.i1.conv1x1.weight_g (castLike (sRef := weights.n.i1.conv1x1.weight_v.shape) weights.n.i1.conv1x1.weight_v (torch.zeros #[cfg.hiddenDim / 2] false Device.CPU))
  }
  let nBlk2 : AdainResBlk1dSame (cfg.hiddenDim / 2) (cfg.hiddenDim / 2) cfg.styleDim := {
    conv1 := wnConv1dFromWeightNorm weights.n.i2.conv1.weight_v weights.n.i2.conv1.weight_g weights.n.i2.conv1.bias
    conv2 := wnConv1dFromWeightNorm weights.n.i2.conv2.weight_v weights.n.i2.conv2.weight_g weights.n.i2.conv2.bias
    norm1 := adaIN1dFromLinear weights.n.i2.norm1.fc.weight weights.n.i2.norm1.fc.bias
    norm2 := adaIN1dFromLinear weights.n.i2.norm2.fc.weight weights.n.i2.norm2.fc.bias
    shortcut := none
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
      weight := reqGradFalse weights.f0_proj.weight
      bias := reqGradFalse weights.f0_proj.bias
    }
    nProj := {
      weight := reqGradFalse weights.n_proj.weight
      bias := reqGradFalse weights.n_proj.bias
    }
  }

private def loadGeneratorFromCheckpoint
    (weights : KokoroCheckpoint.WeightsDecoderGenerator)
    (cfg : KittenTTSConfig)
    : IO (Generator cfg) := do
  let source : SourceModuleHnNSF cfg := {
    linear := linearFromTensors weights.m_source.l_linear.weight weights.m_source.l_linear.bias
  }
  let noiseRes0Dilations : Array UInt64 := #[1, 3, 5]
  let noiseRes1Dilations : Array UInt64 := #[1, 3, 5]
  let stage0Out := GeneratorConfig.stageOutChannels cfg.generator 0
  let stage1Out := GeneratorConfig.stageOutChannels cfg.generator 1
  let stage0Stride ← getOrThrow cfg.generator.upsampleRates 0 "Missing upsample rate for stage 0"
  let stage0Kernel ← getOrThrow cfg.generator.upsampleKernelSizes 0 "Missing upsample kernel for stage 0"
  let stage1Stride ← getOrThrow cfg.generator.upsampleRates 1 "Missing upsample rate for stage 1"
  let stage1Kernel ← getOrThrow cfg.generator.upsampleKernelSizes 1 "Missing upsample kernel for stage 1"
  let resblockKernel0 ← getOrThrow cfg.generator.resblockKernelSizes 0 "Missing generator resblock kernel 0"
  let resblockKernel1 ← getOrThrow cfg.generator.resblockKernelSizes 1 "Missing generator resblock kernel 1"
  let resblockKernel2 ← getOrThrow cfg.generator.resblockKernelSizes 2 "Missing generator resblock kernel 2"
  let resblockDilations0 ← getOrThrow cfg.generator.resblockDilationSizes 0 "Missing generator dilation set 0"
  let resblockDilations1 ← getOrThrow cfg.generator.resblockDilationSizes 1 "Missing generator dilation set 1"
  let resblockDilations2 ← getOrThrow cfg.generator.resblockDilationSizes 2 "Missing generator dilation set 2"
  let noiseRes0 ←
    loadGeneratorAdaINResBlockFromCheckpointArrays
      weights.noise_res.i0.convs1
      weights.noise_res.i0.convs2
      weights.noise_res.i0.adain1
      weights.noise_res.i0.adain2
      weights.noise_res.i0.alpha1
      weights.noise_res.i0.alpha2
      (if 0 + 1 < cfg.generator.upsampleRates.size then 7 else 11)
      noiseRes0Dilations
      (fun node padding dilation =>
        wnConv1dDynFromWeightNorm (inC := node.weight_v.shape[1]!) (outC := node.weight_v.shape[0]!) (kernel := node.weight_v.shape[2]!) node.weight_v node.weight_g node.bias 1 padding dilation)
      (fun node =>
        wnConv1dDynFromWeightNorm (inC := node.weight_v.shape[1]!) (outC := node.weight_v.shape[0]!) (kernel := node.weight_v.shape[2]!)
          node.weight_v
          node.weight_g
          node.bias
          1
          (((if 0 + 1 < cfg.generator.upsampleRates.size then 7 else 11) - 1) / 2)
          1)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage0Out) node.fc.weight node.fc.bias)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage0Out) node.fc.weight node.fc.bias)
      alphaDyn
      alphaDyn
  let noiseRes1 ←
    loadGeneratorAdaINResBlockFromCheckpointArrays
      weights.noise_res.i1.convs1
      weights.noise_res.i1.convs2
      weights.noise_res.i1.adain1
      weights.noise_res.i1.adain2
      weights.noise_res.i1.alpha1
      weights.noise_res.i1.alpha2
      (if 1 + 1 < cfg.generator.upsampleRates.size then 7 else 11)
      noiseRes1Dilations
      (fun node padding dilation =>
        wnConv1dDynFromWeightNorm (inC := node.weight_v.shape[1]!) (outC := node.weight_v.shape[0]!) (kernel := node.weight_v.shape[2]!) node.weight_v node.weight_g node.bias 1 padding dilation)
      (fun node =>
        wnConv1dDynFromWeightNorm (inC := node.weight_v.shape[1]!) (outC := node.weight_v.shape[0]!) (kernel := node.weight_v.shape[2]!)
          node.weight_v
          node.weight_g
          node.bias
          1
          (((if 1 + 1 < cfg.generator.upsampleRates.size then 7 else 11) - 1) / 2)
          1)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage1Out) node.fc.weight node.fc.bias)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage1Out) node.fc.weight node.fc.bias)
      alphaDyn
      alphaDyn
  let res0 ←
    loadGeneratorAdaINResBlockFromCheckpointArrays
      weights.resblocks.i0.convs1
      weights.resblocks.i0.convs2
      weights.resblocks.i0.adain1
      weights.resblocks.i0.adain2
      weights.resblocks.i0.alpha1
      weights.resblocks.i0.alpha2
      resblockKernel0
      resblockDilations0
      (fun node padding dilation =>
        wnConv1dDynFromWeightNorm (inC := node.weight_v.shape[1]!) (outC := node.weight_v.shape[0]!) (kernel := node.weight_v.shape[2]!) node.weight_v node.weight_g node.bias 1 padding dilation)
      (fun node =>
        wnConv1dDynFromWeightNorm (inC := node.weight_v.shape[1]!) (outC := node.weight_v.shape[0]!) (kernel := node.weight_v.shape[2]!) node.weight_v node.weight_g node.bias 1 ((resblockKernel0 - 1) / 2) 1)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage0Out) node.fc.weight node.fc.bias)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage0Out) node.fc.weight node.fc.bias)
      alphaDyn
      alphaDyn
  let res1 ←
    loadGeneratorAdaINResBlockFromCheckpointArrays
      weights.resblocks.i1.convs1
      weights.resblocks.i1.convs2
      weights.resblocks.i1.adain1
      weights.resblocks.i1.adain2
      weights.resblocks.i1.alpha1
      weights.resblocks.i1.alpha2
      resblockKernel1
      resblockDilations1
      (fun node padding dilation =>
        wnConv1dDynFromWeightNorm (inC := node.weight_v.shape[1]!) (outC := node.weight_v.shape[0]!) (kernel := node.weight_v.shape[2]!) node.weight_v node.weight_g node.bias 1 padding dilation)
      (fun node =>
        wnConv1dDynFromWeightNorm (inC := node.weight_v.shape[1]!) (outC := node.weight_v.shape[0]!) (kernel := node.weight_v.shape[2]!) node.weight_v node.weight_g node.bias 1 ((resblockKernel1 - 1) / 2) 1)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage0Out) node.fc.weight node.fc.bias)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage0Out) node.fc.weight node.fc.bias)
      alphaDyn
      alphaDyn
  let res2 ←
    loadGeneratorAdaINResBlockFromCheckpointArrays
      weights.resblocks.i2.convs1
      weights.resblocks.i2.convs2
      weights.resblocks.i2.adain1
      weights.resblocks.i2.adain2
      weights.resblocks.i2.alpha1
      weights.resblocks.i2.alpha2
      resblockKernel2
      resblockDilations2
      (fun node padding dilation =>
        wnConv1dDynFromWeightNorm (inC := node.weight_v.shape[1]!) (outC := node.weight_v.shape[0]!) (kernel := node.weight_v.shape[2]!) node.weight_v node.weight_g node.bias 1 padding dilation)
      (fun node =>
        wnConv1dDynFromWeightNorm (inC := node.weight_v.shape[1]!) (outC := node.weight_v.shape[0]!) (kernel := node.weight_v.shape[2]!) node.weight_v node.weight_g node.bias 1 ((resblockKernel2 - 1) / 2) 1)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage0Out) node.fc.weight node.fc.bias)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage0Out) node.fc.weight node.fc.bias)
      alphaDyn
      alphaDyn
  let res3 ←
    loadGeneratorAdaINResBlockFromCheckpointArrays
      weights.resblocks.i3.convs1
      weights.resblocks.i3.convs2
      weights.resblocks.i3.adain1
      weights.resblocks.i3.adain2
      weights.resblocks.i3.alpha1
      weights.resblocks.i3.alpha2
      resblockKernel0
      resblockDilations0
      (fun node padding dilation =>
        wnConv1dDynFromWeightNorm (inC := node.weight_v.shape[1]!) (outC := node.weight_v.shape[0]!) (kernel := node.weight_v.shape[2]!) node.weight_v node.weight_g node.bias 1 padding dilation)
      (fun node =>
        wnConv1dDynFromWeightNorm (inC := node.weight_v.shape[1]!) (outC := node.weight_v.shape[0]!) (kernel := node.weight_v.shape[2]!) node.weight_v node.weight_g node.bias 1 ((resblockKernel0 - 1) / 2) 1)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage1Out) node.fc.weight node.fc.bias)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage1Out) node.fc.weight node.fc.bias)
      alphaDyn
      alphaDyn
  let res4 ←
    loadGeneratorAdaINResBlockFromCheckpointArrays
      weights.resblocks.i4.convs1
      weights.resblocks.i4.convs2
      weights.resblocks.i4.adain1
      weights.resblocks.i4.adain2
      weights.resblocks.i4.alpha1
      weights.resblocks.i4.alpha2
      resblockKernel1
      resblockDilations1
      (fun node padding dilation =>
        wnConv1dDynFromWeightNorm (inC := node.weight_v.shape[1]!) (outC := node.weight_v.shape[0]!) (kernel := node.weight_v.shape[2]!) node.weight_v node.weight_g node.bias 1 padding dilation)
      (fun node =>
        wnConv1dDynFromWeightNorm (inC := node.weight_v.shape[1]!) (outC := node.weight_v.shape[0]!) (kernel := node.weight_v.shape[2]!) node.weight_v node.weight_g node.bias 1 ((resblockKernel1 - 1) / 2) 1)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage1Out) node.fc.weight node.fc.bias)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage1Out) node.fc.weight node.fc.bias)
      alphaDyn
      alphaDyn
  let res5 ←
    loadGeneratorAdaINResBlockFromCheckpointArrays
      weights.resblocks.i5.convs1
      weights.resblocks.i5.convs2
      weights.resblocks.i5.adain1
      weights.resblocks.i5.adain2
      weights.resblocks.i5.alpha1
      weights.resblocks.i5.alpha2
      resblockKernel2
      resblockDilations2
      (fun node padding dilation =>
        wnConv1dDynFromWeightNorm (inC := node.weight_v.shape[1]!) (outC := node.weight_v.shape[0]!) (kernel := node.weight_v.shape[2]!) node.weight_v node.weight_g node.bias 1 padding dilation)
      (fun node =>
        wnConv1dDynFromWeightNorm (inC := node.weight_v.shape[1]!) (outC := node.weight_v.shape[0]!) (kernel := node.weight_v.shape[2]!) node.weight_v node.weight_g node.bias 1 ((resblockKernel2 - 1) / 2) 1)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage1Out) node.fc.weight node.fc.bias)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage1Out) node.fc.weight node.fc.bias)
      alphaDyn
      alphaDyn
  let stage0 : GeneratorStage cfg.styleDim := {
    up :=
      wnConvTranspose1dDynFromWeightNorm (inC := weights.ups.i0.weight_v.shape[0]!) (outC := weights.ups.i0.weight_v.shape[1]!) (kernel := weights.ups.i0.weight_v.shape[2]!)
        weights.ups.i0.weight_v
        weights.ups.i0.weight_g
        weights.ups.i0.bias
        stage0Stride
        ((stage0Kernel - stage0Stride) / 2)
        0
        1
    noiseConv :=
      wnConv1dDynFromPlain (inC := weights.noise_convs.i0.weight.shape[1]!) (outC := weights.noise_convs.i0.weight.shape[0]!) (kernel := weights.noise_convs.i0.weight.shape[2]!)
        weights.noise_convs.i0.weight
        weights.noise_convs.i0.bias
        (GeneratorConfig.noiseStride cfg.generator 0)
        (GeneratorConfig.noisePadding cfg.generator 0)
        1
    noiseRes := noiseRes0
    resBlocks := #[res0, res1, res2]
  }
  let stage1 : GeneratorStage cfg.styleDim := {
    up :=
      wnConvTranspose1dDynFromWeightNorm (inC := weights.ups.i1.weight_v.shape[0]!) (outC := weights.ups.i1.weight_v.shape[1]!) (kernel := weights.ups.i1.weight_v.shape[2]!)
        weights.ups.i1.weight_v
        weights.ups.i1.weight_g
        weights.ups.i1.bias
        stage1Stride
        ((stage1Kernel - stage1Stride) / 2)
        0
        1
    noiseConv :=
      wnConv1dDynFromPlain (inC := weights.noise_convs.i1.weight.shape[1]!) (outC := weights.noise_convs.i1.weight.shape[0]!) (kernel := weights.noise_convs.i1.weight.shape[2]!)
        weights.noise_convs.i1.weight
        weights.noise_convs.i1.bias
        (GeneratorConfig.noiseStride cfg.generator 1)
        (GeneratorConfig.noisePadding cfg.generator 1)
        1
    noiseRes := noiseRes1
    resBlocks := #[res3, res4, res5]
  }
  let convPost := wnConv1dDynFromWeightNorm (inC := weights.conv_post.weight_v.shape[1]!) (outC := weights.conv_post.weight_v.shape[0]!) (kernel := weights.conv_post.weight_v.shape[2]!) weights.conv_post.weight_v weights.conv_post.weight_g weights.conv_post.bias 1 3 1
  pure {
    source
    stages := #[stage0, stage1]
    convPost
    window := signal.hannWindow cfg.generator.genIstftNFft
  }

private def loadDecoderFromCheckpoint
    (weights : KokoroCheckpoint.WeightsDecoder)
    (cfg : KittenTTSConfig)
    : IO (Decoder cfg) := do
  let asrResNode ← getOrThrow weights.asr_res 0 "Missing decoder.asr_res.0"
  let generator ← loadGeneratorFromCheckpoint weights.generator cfg
  let encode : AdainResBlk1dSame (cfg.hiddenDim + 2) cfg.maxConvDim cfg.styleDim := {
    conv1 := wnConv1dFromWeightNorm weights.encode.conv1.weight_v weights.encode.conv1.weight_g weights.encode.conv1.bias
    conv2 := wnConv1dFromWeightNorm weights.encode.conv2.weight_v weights.encode.conv2.weight_g weights.encode.conv2.bias
    norm1 := adaIN1dFromLinear weights.encode.norm1.fc.weight weights.encode.norm1.fc.bias
    norm2 := adaIN1dFromLinear weights.encode.norm2.fc.weight weights.encode.norm2.fc.bias
    shortcut := some <| wnConv1dFromWeightNorm weights.encode.conv1x1.weight_v weights.encode.conv1x1.weight_g (castLike (sRef := weights.encode.conv1x1.weight_v.shape) weights.encode.conv1x1.weight_v (torch.zeros #[cfg.maxConvDim] false Device.CPU))
  }
  let decode0 : AdainResBlk1dSame (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim := {
    conv1 := wnConv1dFromWeightNorm weights.decode.i0.conv1.weight_v weights.decode.i0.conv1.weight_g weights.decode.i0.conv1.bias
    conv2 := wnConv1dFromWeightNorm weights.decode.i0.conv2.weight_v weights.decode.i0.conv2.weight_g weights.decode.i0.conv2.bias
    norm1 := adaIN1dFromLinear weights.decode.i0.norm1.fc.weight weights.decode.i0.norm1.fc.bias
    norm2 := adaIN1dFromLinear weights.decode.i0.norm2.fc.weight weights.decode.i0.norm2.fc.bias
    shortcut := some <| wnConv1dFromWeightNorm weights.decode.i0.conv1x1.weight_v weights.decode.i0.conv1x1.weight_g (castLike (sRef := weights.decode.i0.conv1x1.weight_v.shape) weights.decode.i0.conv1x1.weight_v (torch.zeros #[cfg.maxConvDim] false Device.CPU))
  }
  let decode1 : AdainResBlk1dSame (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim := {
    conv1 := wnConv1dFromWeightNorm weights.decode.i1.conv1.weight_v weights.decode.i1.conv1.weight_g weights.decode.i1.conv1.bias
    conv2 := wnConv1dFromWeightNorm weights.decode.i1.conv2.weight_v weights.decode.i1.conv2.weight_g weights.decode.i1.conv2.bias
    norm1 := adaIN1dFromLinear weights.decode.i1.norm1.fc.weight weights.decode.i1.norm1.fc.bias
    norm2 := adaIN1dFromLinear weights.decode.i1.norm2.fc.weight weights.decode.i1.norm2.fc.bias
    shortcut := some <| wnConv1dFromWeightNorm weights.decode.i1.conv1x1.weight_v weights.decode.i1.conv1x1.weight_g (castLike (sRef := weights.decode.i1.conv1x1.weight_v.shape) weights.decode.i1.conv1x1.weight_v (torch.zeros #[cfg.maxConvDim] false Device.CPU))
  }
  let decode2 : AdainResBlk1dSame (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim := {
    conv1 := wnConv1dFromWeightNorm weights.decode.i2.conv1.weight_v weights.decode.i2.conv1.weight_g weights.decode.i2.conv1.bias
    conv2 := wnConv1dFromWeightNorm weights.decode.i2.conv2.weight_v weights.decode.i2.conv2.weight_g weights.decode.i2.conv2.bias
    norm1 := adaIN1dFromLinear weights.decode.i2.norm1.fc.weight weights.decode.i2.norm1.fc.bias
    norm2 := adaIN1dFromLinear weights.decode.i2.norm2.fc.weight weights.decode.i2.norm2.fc.bias
    shortcut := some <| wnConv1dFromWeightNorm weights.decode.i2.conv1x1.weight_v weights.decode.i2.conv1x1.weight_g (castLike (sRef := weights.decode.i2.conv1x1.weight_v.shape) weights.decode.i2.conv1x1.weight_v (torch.zeros #[cfg.maxConvDim] false Device.CPU))
  }
  let decode3 : AdainResBlk1dUp (cfg.maxConvDim + cfg.asrResDim + 2) (KittenTTSConfig.decoderChannels cfg) cfg.styleDim := {
    pool := wnDepthwiseTransConv1dFromWeightNorm weights.decode.i3.pool.weight_v weights.decode.i3.pool.weight_g weights.decode.i3.pool.bias
    conv1 := wnConv1dFromWeightNorm weights.decode.i3.conv1.weight_v weights.decode.i3.conv1.weight_g weights.decode.i3.conv1.bias
    conv2 := wnConv1dFromWeightNorm weights.decode.i3.conv2.weight_v weights.decode.i3.conv2.weight_g weights.decode.i3.conv2.bias
    norm1 := adaIN1dFromLinear weights.decode.i3.norm1.fc.weight weights.decode.i3.norm1.fc.bias
    norm2 := adaIN1dFromLinear weights.decode.i3.norm2.fc.weight weights.decode.i3.norm2.fc.bias
    shortcut := some <| wnConv1dFromWeightNorm weights.decode.i3.conv1x1.weight_v weights.decode.i3.conv1x1.weight_g (castLike (sRef := weights.decode.i3.conv1x1.weight_v.shape) weights.decode.i3.conv1x1.weight_v (torch.zeros #[KittenTTSConfig.decoderChannels cfg] false Device.CPU))
  }
  pure {
    f0Conv := wnConv1dFromWeightNorm weights.f0_conv.weight_v weights.f0_conv.weight_g weights.f0_conv.bias
    nConv := wnConv1dFromWeightNorm weights.n_conv.weight_v weights.n_conv.weight_g weights.n_conv.bias
    encode
    asrRes := wnConv1dFromWeightNorm asrResNode.weight_v asrResNode.weight_g asrResNode.bias
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
      pure (Device.CUDA 0)
    else
      log.onWarn "TYR_DEVICE=cuda requested but CUDA is unavailable; falling back to auto."
      getBestDevice
  | some "mps" => pure Device.MPS
  | some "auto" => getBestDevice
  | some _ => getBestDevice
  | none =>
    if ← cuda_is_available then
      pure (Device.CUDA 0)
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
  warnShapeOverride log "nLayer" cfg.nLayer weights.textEncoder.cnn.size.toUInt64
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
  pure {
    cfg with
    hiddenDim := hiddenDim
    maxConvDim := decoderBlockDim
    maxDur := maxDur
    nLayer := weights.textEncoder.cnn.size.toUInt64
    nToken := nToken
    styleDim := styleDim
    textEncoderKernelSize := textEncoderKernelSize
    asrResDim := asrResDim
    decoderOutDim := decoderOutDim
    generator := generator
  }

private def buildModelFromCheckpointWeights
    (weights : LoadedCheckpointWeights)
    (cfg : KittenTTSConfig)
    (log : Handlers := {})
    : IO (Model cfg) := do
  let bert ← loadKittenAlbertFromCheckpoint weights.bert cfg
  let bertEncoder : LinearNorm cfg.plbert.hiddenSize cfg.hiddenDim :=
    linearFromTensors weights.bertEncoder.weight weights.bertEncoder.bias
  let predictor ← loadProsodyPredictorFromCheckpoint weights.predictor cfg log
  let textEncoder ← loadTextEncoderFromCheckpoint weights.textEncoder cfg
  let decoder ← loadDecoderFromCheckpoint weights.decoder cfg
  log.onInfo "Resolving target device..."
  let targetDevice ← resolveKittenDevice log
  let baseModel : Model cfg := { bert, bertEncoder, predictor, textEncoder, decoder }
  let model : Model cfg :=
    match targetDevice with
    | Device.CPU => baseModel
    | _ => TensorStruct.map (fun t => t.to targetDevice) baseModel
  log.onInfo s!"KittenTTS target device: {repr targetDevice}"
  pure model

namespace Model

structure Loaded where
  cfg : KittenTTSConfig
  model : Model cfg

def load (path : String) (cfg : KittenTTSConfig := {}) (log : Handlers := {})
    : IO (Model cfg) := do
  log.onInfo s!"Loading KittenTTS weights from {path}..."
  let weights ← loadCheckpointWeights path log
  buildModelFromCheckpointWeights weights cfg log

def loadAutoConfig (path : String) (cfg : KittenTTSConfig := {}) (log : Handlers := {})
    : IO Loaded := do
  log.onInfo s!"Loading KittenTTS weights from {path}..."
  let weights ← loadCheckpointWeights path log
  let cfg' ← reconcileConfigWithCheckpoint cfg weights log
  let model ← buildModelFromCheckpointWeights weights cfg' log
  pure { cfg := cfg', model := model }

end Model

end torch.kittentts
