// Copyright © 2025 Resemble AI (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/resemble-ai/chatterbox
// License: licenses/chatterbox.txt

// Token-To-Token (T3) TTS model using GPT2 as backbone for Chatterbox Turbo

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom

// MARK: - T3 Turbo Config

/// Configuration for T3 Turbo model
struct T3TurboConfig: Codable, Sendable {
  // Text tokens
  var startTextToken: Int = 255
  var stopTextToken: Int = 0
  var textTokensDictSize: Int = 50276
  var maxTextTokens: Int = 2048

  // Speech tokens
  var startSpeechToken: Int = 6561
  var stopSpeechToken: Int = 6562
  var speechTokensDictSize: Int = 6563
  var maxSpeechTokens: Int = 4096

  // Model architecture
  var llamaConfigName: String = "GPT2_medium"
  var inputPosEmb: String? = nil // Turbo doesn't use learned pos emb
  var speechCondPromptLen: Int = 375

  // Conditioning
  var encoderType: String = "voice_encoder"
  var speakerEmbedSize: Int = 256
  var usePerceiverResampler: Bool = false // Turbo doesn't use perceiver
  var emotionAdv: Bool = false // Turbo doesn't use emotion

  /// Get hidden size from GPT2 config
  var nChannels: Int { 1024 }

  enum CodingKeys: String, CodingKey {
    case startTextToken = "start_text_token"
    case stopTextToken = "stop_text_token"
    case textTokensDictSize = "text_tokens_dict_size"
    case maxTextTokens = "max_text_tokens"
    case startSpeechToken = "start_speech_token"
    case stopSpeechToken = "stop_speech_token"
    case speechTokensDictSize = "speech_tokens_dict_size"
    case maxSpeechTokens = "max_speech_tokens"
    case llamaConfigName = "llama_config_name"
    case inputPosEmb = "input_pos_emb"
    case speechCondPromptLen = "speech_cond_prompt_len"
    case encoderType = "encoder_type"
    case speakerEmbedSize = "speaker_embed_size"
    case usePerceiverResampler = "use_perceiver_resampler"
    case emotionAdv = "emotion_adv"
  }

  init() {}

  /// Create default Turbo configuration
  static func turbo() -> T3TurboConfig {
    T3TurboConfig()
  }
}

// MARK: - T3 Turbo Cond

/// Container for T3 Turbo conditioning information
struct T3TurboCond: @unchecked Sendable {
  /// Speaker embedding from voice encoder (B, speaker_dim)
  var speakerEmb: MLXArray

  /// Optional speech token prompt (B, T)
  var condPromptSpeechTokens: MLXArray?

  /// Optional embedded speech prompt (B, T, D)
  var condPromptSpeechEmb: MLXArray?

  init(
    speakerEmb: MLXArray,
    condPromptSpeechTokens: MLXArray? = nil,
    condPromptSpeechEmb: MLXArray? = nil
  ) {
    self.speakerEmb = speakerEmb
    self.condPromptSpeechTokens = condPromptSpeechTokens
    self.condPromptSpeechEmb = condPromptSpeechEmb
  }
}

// MARK: - T3 Turbo CondEnc

/// Conditioning encoder for T3 Turbo model
/// Handles speaker embeddings and prompt speech tokens (no emotion/CLAP in Turbo)
class T3TurboCondEnc: Module {
  let config: T3TurboConfig

  @ModuleInfo(key: "spkr_enc") var spkrEnc: Linear

  init(config: T3TurboConfig) {
    self.config = config

    // Speaker embedding projection
    _spkrEnc.wrappedValue = Linear(config.speakerEmbedSize, config.nChannels)

    super.init()
  }

  /// Process conditioning inputs into a single conditioning tensor
  func callAsFunction(_ cond: T3TurboCond) -> MLXArray {
    // Validate
    let hasTokens = cond.condPromptSpeechTokens != nil
    let hasEmb = cond.condPromptSpeechEmb != nil
    precondition(
      hasTokens == hasEmb,
      "condPromptSpeechTokens and condPromptSpeechEmb must both be provided or both be nil"
    )

    // Speaker embedding projection (B, speaker_dim) -> (B, 1, D)
    let B = cond.speakerEmb.shape[0]
    var condSpkr = spkrEnc(cond.speakerEmb.reshaped([B, config.speakerEmbedSize]))
    condSpkr = condSpkr.expandedDimensions(axis: 1) // (B, 1, D)

    let dim = condSpkr.shape[2]

    // Empty placeholder for unused conditioning
    let empty = MLXArray.zeros([B, 0, dim])

    // Conditional prompt speech embeddings
    let condPromptSpeechEmb = cond.condPromptSpeechEmb ?? empty

    // Turbo doesn't use CLAP or emotion, so those are empty
    let condClap = empty
    let condEmotionAdv = empty

    // Concatenate all conditioning signals
    let condEmbeds = MLX.concatenated([
      condSpkr,
      condClap,
      condPromptSpeechEmb,
      condEmotionAdv,
    ], axis: 1)

    return condEmbeds
  }
}

// MARK: - T3 Turbo

/// Token-To-Token (T3) TTS model using GPT2 as backbone
/// Marked `@unchecked Sendable` for async streaming operations
class T3Turbo: Module, @unchecked Sendable {
  let config: T3TurboConfig
  let gpt2Config: GPT2Config
  let dim: Int

  @ModuleInfo(key: "tfmr") var tfmr: GPT2Model
  @ModuleInfo(key: "cond_enc") var condEnc: T3TurboCondEnc
  @ModuleInfo(key: "text_emb") var textEmb: Embedding
  @ModuleInfo(key: "speech_emb") var speechEmb: Embedding
  @ModuleInfo(key: "text_head") var textHead: Linear
  @ModuleInfo(key: "speech_head") var speechHead: Linear

  init(config: T3TurboConfig? = nil) {
    let hp = config ?? T3TurboConfig.turbo()
    self.config = hp

    // Create GPT2 config
    gpt2Config = GPT2Config.gpt2Medium()
    dim = gpt2Config.hiddenSize

    // GPT2 transformer backbone
    _tfmr.wrappedValue = GPT2Model(gpt2Config)

    // Conditioning encoder
    _condEnc.wrappedValue = T3TurboCondEnc(config: hp)

    // Text and speech token embeddings
    _textEmb.wrappedValue = Embedding(embeddingCount: hp.textTokensDictSize, dimensions: dim)
    _speechEmb.wrappedValue = Embedding(embeddingCount: hp.speechTokensDictSize, dimensions: dim)

    // Output projection heads
    _textHead.wrappedValue = Linear(dim, hp.textTokensDictSize, bias: false)
    _speechHead.wrappedValue = Linear(dim, hp.speechTokensDictSize, bias: true)

    super.init()
  }

  /// Prepare conditioning embeddings from T3TurboCond
  func prepareConditioning(_ t3Cond: inout T3TurboCond) -> MLXArray {
    // Embed speech prompt tokens if provided
    if t3Cond.condPromptSpeechTokens != nil, t3Cond.condPromptSpeechEmb == nil {
      t3Cond.condPromptSpeechEmb = speechEmb(t3Cond.condPromptSpeechTokens!)
    }

    return condEnc(t3Cond)
  }

  /// Prepare input embeddings for the transformer
  func prepareInputEmbeds(
    t3Cond: inout T3TurboCond,
    textTokens: MLXArray,
    speechTokens: MLXArray
  ) -> (MLXArray, Int) {
    // Prepare conditioning embeddings
    var condEmb = prepareConditioning(&t3Cond)

    // Text embeddings
    let textEmbeddings = textEmb(textTokens)

    // Speech embeddings
    let speechEmbeddings = speechEmb(speechTokens)

    let lenCond = condEmb.shape[1]

    // Broadcast conditioning if batch sizes don't match
    if condEmb.shape[0] != textEmbeddings.shape[0] {
      condEmb = MLX.broadcast(
        condEmb,
        to: [textEmbeddings.shape[0]] + Array(condEmb.shape.dropFirst())
      )
    }

    // Concatenate: [conditioning | text | speech]
    let embeds = MLX.concatenated([condEmb, textEmbeddings, speechEmbeddings], axis: 1)

    return (embeds, lenCond)
  }

  /// Generate speech tokens from text tokens (Turbo inference)
  func inferenceTurbo(
    t3Cond: inout T3TurboCond,
    textTokens: MLXArray,
    temperature: Float = 0.8,
    topK: Int = 1000,
    topP: Float = 0.95,
    repetitionPenalty: Float = 1.2,
    maxGenLen: Int = 1000
  ) -> MLXArray {
    // Ensure batch dimension
    var tokens = textTokens
    if tokens.ndim == 1 {
      tokens = tokens.expandedDimensions(axis: 0)
    }

    let B = tokens.shape[0]

    // Initial speech token (BOS)
    let speechStartToken = MLXArray.full([B, 1], values: MLXArray(Int32(config.startSpeechToken)))

    // Prepare initial embeddings
    let (embeds, _) = prepareInputEmbeds(
      t3Cond: &t3Cond,
      textTokens: tokens,
      speechTokens: speechStartToken
    )

    // Create KV cache
    let cache = tfmr.newCache()

    // Initial forward pass
    var hiddenStates = tfmr(inputsEmbeds: embeds, cache: cache)

    // Get first speech prediction (last position, keeping dimension)
    let speechHidden = hiddenStates[0..., (hiddenStates.shape[1] - 1) ..< hiddenStates.shape[1], 0...]
    var speechLogits = speechHead(speechHidden)

    // Sample first token
    var nextSpeechToken = sampleTokenTurbo(
      logits: speechLogits[0..., -1, 0...],
      temperature: temperature,
      topK: topK,
      topP: topP,
      generatedTokens: nil,
      repetitionPenalty: repetitionPenalty
    )

    // Track generated tokens (pre-allocate to avoid reallocations)
    var generatedIds: [Int32] = []
    generatedIds.reserveCapacity(maxGenLen + 1)
    asyncEval(nextSpeechToken, cache)
    let firstTokenId = nextSpeechToken.item(Int32.self)
    generatedIds.append(firstTokenId)

    var currentSpeechToken = nextSpeechToken.reshaped([B, 1])

    // Generation loop
    for _ in 0 ..< maxGenLen {
      // Check for cancellation
      if Task.isCancelled {
        break
      }

      // Get embedding for current token
      let currentSpeechEmbed = speechEmb(currentSpeechToken)

      // Forward pass with cache
      hiddenStates = tfmr(inputsEmbeds: currentSpeechEmbed, cache: cache)

      // Get logits
      speechLogits = speechHead(hiddenStates)

      // Sample next token
      nextSpeechToken = sampleTokenTurbo(
        logits: speechLogits[0..., -1, 0...],
        temperature: temperature,
        topK: topK,
        topP: topP,
        generatedTokens: MLXArray(generatedIds),
        repetitionPenalty: repetitionPenalty
      )

      // Start async eval before extracting token - GPU works while we prepare
      asyncEval(nextSpeechToken, cache)
      let nextTokenId = nextSpeechToken.item(Int32.self)

      // Check for EOS
      if nextTokenId == Int32(config.stopSpeechToken) {
        break
      }

      generatedIds.append(nextTokenId)
      currentSpeechToken = nextSpeechToken.reshaped([B, 1])
    }

    return MLXArray(generatedIds).reshaped([1, -1])
  }

  /// Generate speech tokens with streaming (yields chunks)
  func inferenceTurboStream(
    t3Cond: inout T3TurboCond,
    textTokens: MLXArray,
    temperature: Float = 0.8,
    topK: Int = 1000,
    topP: Float = 0.95,
    repetitionPenalty: Float = 1.2,
    maxGenLen: Int = 1000,
    chunkSize: Int = 40
  ) -> AsyncThrowingStream<(MLXArray, Bool), Error> {
    // Create generator state
    let state = T3TurboStreamState(
      model: self,
      t3Cond: t3Cond,
      textTokens: textTokens,
      temperature: temperature,
      topK: topK,
      topP: topP,
      repetitionPenalty: repetitionPenalty,
      maxGenLen: maxGenLen,
      chunkSize: chunkSize
    )

    return AsyncThrowingStream(unfolding: { try await state.next() })
  }
}

// MARK: - Stream State

/// Encapsulates streaming generation state for T3Turbo
private final class T3TurboStreamState: @unchecked Sendable {
  let model: T3Turbo
  var t3Cond: T3TurboCond
  let temperature: Float
  let topK: Int
  let topP: Float
  let repetitionPenalty: Float
  let maxGenLen: Int
  let chunkSize: Int

  // Generation state
  var isInitialized = false
  var isFinished = false
  var B: Int = 1
  var cache: [any KVCache]?
  var currentSpeechToken: MLXArray?
  var generatedIds: [Int32] = []
  var chunkTokens: [Int32] = []
  var stepCount = 0

  init(
    model: T3Turbo,
    t3Cond: T3TurboCond,
    textTokens: MLXArray,
    temperature: Float,
    topK: Int,
    topP: Float,
    repetitionPenalty: Float,
    maxGenLen: Int,
    chunkSize: Int
  ) {
    self.model = model
    self.t3Cond = t3Cond
    self.temperature = temperature
    self.topK = topK
    self.topP = topP
    self.repetitionPenalty = repetitionPenalty
    self.maxGenLen = maxGenLen
    self.chunkSize = chunkSize

    // Initialize generation
    var tokens = textTokens
    if tokens.ndim == 1 {
      tokens = tokens.expandedDimensions(axis: 0)
    }
    B = tokens.shape[0]

    // Initial speech token (BOS)
    let speechStartToken = MLXArray.full([B, 1], values: MLXArray(Int32(model.config.startSpeechToken)))

    // Prepare initial embeddings
    let (embeds, _) = model.prepareInputEmbeds(
      t3Cond: &self.t3Cond,
      textTokens: tokens,
      speechTokens: speechStartToken
    )

    // Create KV cache
    cache = model.tfmr.newCache()

    // Initial forward pass
    let hiddenStates = model.tfmr(inputsEmbeds: embeds, cache: cache)

    // Get first speech prediction (last position, keeping dimension)
    let speechHidden = hiddenStates[0..., (hiddenStates.shape[1] - 1) ..< hiddenStates.shape[1], 0...]
    let speechLogits = model.speechHead(speechHidden)

    // Sample first token
    let nextSpeechToken = sampleTokenTurbo(
      logits: speechLogits[0..., -1, 0...],
      temperature: temperature,
      topK: topK,
      topP: topP,
      generatedTokens: nil,
      repetitionPenalty: repetitionPenalty
    )

    // Pre-allocate arrays to avoid reallocations
    generatedIds.reserveCapacity(maxGenLen + 1)
    chunkTokens.reserveCapacity(chunkSize + 1)

    if let c = cache { asyncEval(nextSpeechToken, c) } else { asyncEval(nextSpeechToken) }
    let firstTokenId = nextSpeechToken.item(Int32.self)
    generatedIds.append(firstTokenId)
    chunkTokens.append(firstTokenId)

    currentSpeechToken = nextSpeechToken.reshaped([B, 1])
    isInitialized = true
  }

  func next() async throws -> (MLXArray, Bool)? {
    guard !isFinished else { return nil }
    try Task.checkCancellation()

    // Check if we have a chunk ready to yield
    if chunkTokens.count >= chunkSize {
      let chunk = MLXArray(chunkTokens).reshaped([1, -1])
      chunkTokens = []
      return (chunk, false)
    }

    // Generate more tokens until we have a chunk or hit EOS
    while stepCount < maxGenLen {
      try Task.checkCancellation()
      stepCount += 1

      guard let token = currentSpeechToken, let cache else { break }

      // Get embedding for current token
      let currentSpeechEmbed = model.speechEmb(token)

      // Forward pass with cache
      let hiddenStates = model.tfmr(inputsEmbeds: currentSpeechEmbed, cache: cache)

      // Get logits
      let speechLogits = model.speechHead(hiddenStates)

      // Sample next token
      let nextSpeechToken = sampleTokenTurbo(
        logits: speechLogits[0..., -1, 0...],
        temperature: temperature,
        topK: topK,
        topP: topP,
        generatedTokens: MLXArray(generatedIds),
        repetitionPenalty: repetitionPenalty
      )

      // Start async eval before extracting token - GPU works while we prepare
      asyncEval(nextSpeechToken, cache)
      let nextTokenId = nextSpeechToken.item(Int32.self)

      // Check for EOS
      if nextTokenId == Int32(model.config.stopSpeechToken) {
        isFinished = true
        if !chunkTokens.isEmpty {
          let chunk = MLXArray(chunkTokens).reshaped([1, -1])
          chunkTokens = []
          return (chunk, true)
        }
        return nil
      }

      generatedIds.append(nextTokenId)
      chunkTokens.append(nextTokenId)
      currentSpeechToken = nextSpeechToken.reshaped([B, 1])

      // Yield chunk if we've accumulated enough tokens
      if chunkTokens.count >= chunkSize {
        let chunk = MLXArray(chunkTokens).reshaped([1, -1])
        chunkTokens = []
        return (chunk, false)
      }
    }

    // Yield any remaining tokens at end
    isFinished = true
    if !chunkTokens.isEmpty {
      let chunk = MLXArray(chunkTokens).reshaped([1, -1])
      chunkTokens = []
      return (chunk, true)
    }
    return nil
  }
}

// MARK: - Sampling Helpers

/// Sample token with temperature, top-k, top-p, and repetition penalty
private func sampleTokenTurbo(
  logits: MLXArray,
  temperature: Float,
  topK: Int,
  topP: Float,
  generatedTokens: MLXArray?,
  repetitionPenalty: Float
) -> MLXArray {
  var processedLogits = logits

  // Apply repetition penalty
  if let generated = generatedTokens, repetitionPenalty != 1.0 {
    processedLogits = applyRepetitionPenaltyTurbo(
      logits: processedLogits,
      generatedTokens: generated,
      penalty: repetitionPenalty
    )
  }

  // Apply temperature
  if temperature > 0, temperature != 1.0 {
    processedLogits = processedLogits / temperature
  }

  // Apply top-k
  if topK > 0 {
    processedLogits = topKFilteringTurbo(processedLogits, topK: topK)
  }

  // Apply top-p
  if topP < 1.0 {
    processedLogits = topPFilteringTurbo(processedLogits, topP: topP)
  }

  // Sample from categorical distribution
  return MLXRandom.categorical(processedLogits)
}

/// Apply repetition penalty to logits (vectorized)
private func applyRepetitionPenaltyTurbo(
  logits: MLXArray,
  generatedTokens: MLXArray,
  penalty: Float
) -> MLXArray {
  if penalty == 1.0 {
    return logits
  }

  let vocabSize = logits.shape[logits.ndim - 1]

  // Get unique generated tokens
  let flatTokens = generatedTokens.flattened()
  let tokensArray = flatTokens.asArray(Int32.self)
  let uniqueTokens = Array(Set(tokensArray))

  // Create mask for tokens that should be penalized (vectorized)
  var tokenMaskArray = [Float](repeating: 0.0, count: vocabSize)
  for token in uniqueTokens where token >= 0 && token < Int32(vocabSize) {
    tokenMaskArray[Int(token)] = 1.0
  }
  let tokenMask = MLXArray(tokenMaskArray).expandedDimensions(axis: 0)

  // Apply penalty: divide positive scores, multiply negative scores
  let penaltyScalar = MLXArray(penalty)
  let penalized = MLX.where(
    logits .< 0,
    logits * penaltyScalar,
    logits / penaltyScalar
  )

  // Only apply penalty to tokens that have been generated
  return MLX.where(tokenMask .> 0, penalized, logits)
}

/// Filter logits to only keep top-k values
private func topKFilteringTurbo(_ logits: MLXArray, topK: Int) -> MLXArray {
  if topK <= 0 {
    return logits
  }

  let k = min(topK, logits.shape[logits.ndim - 1])

  // Find the k-th largest value as threshold
  let partitioned = MLX.argPartition(logits, kth: -k, axis: -1)
  let vocabSize = partitioned.shape[partitioned.ndim - 1]
  let kthIndices = partitioned[0..., (vocabSize - k) ..< (vocabSize - k + 1)]
  let kthValues = MLX.takeAlong(logits, kthIndices, axis: -1)

  // Mask: keep values >= kth value
  let mask = logits .>= kthValues

  // Apply mask (set non-top-k to -inf)
  return MLX.where(mask, logits, MLXArray(-Float.infinity))
}

/// Filter logits using nucleus (top-p) sampling
private func topPFilteringTurbo(_ logits: MLXArray, topP: Float) -> MLXArray {
  if topP >= 1.0 {
    return logits
  }

  // Sort logits in descending order
  let sortedIndices = MLX.argSort(-logits, axis: -1)
  let sortedLogits = MLX.takeAlong(logits, sortedIndices, axis: -1)

  // Compute cumulative probabilities
  let sortedProbs = softmax(sortedLogits, axis: -1)
  let cumulativeProbs = MLX.cumsum(sortedProbs, axis: -1)

  // Remove tokens with cumulative probability above threshold
  let sortedIndicesToRemove = cumulativeProbs .> topP

  // Shift right to keep first token above threshold
  let zeros = MLXArray.zeros([logits.shape[0], 1], dtype: .bool)
  let lastDim = sortedIndicesToRemove.shape[sortedIndicesToRemove.ndim - 1]
  let shiftedMask = MLX.concatenated([zeros, sortedIndicesToRemove[0..., 0 ..< (lastDim - 1)]], axis: -1)

  // Set removed tokens to -inf
  let maskedSortedLogits = MLX.where(shiftedMask, MLXArray(-Float.infinity), sortedLogits)

  // Scatter back to original order
  let inverseIndices = MLX.argSort(sortedIndices, axis: -1)
  return MLX.takeAlong(maskedSortedLogits, inverseIndices, axis: -1)
}
