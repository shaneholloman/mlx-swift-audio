// Copyright © 2025 FunASR (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/modelscope/FunASR
// License: licenses/funasr.txt

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Main Fun-ASR model combining audio encoder, adaptor, and LLM decoder
class FunASRModel: Module {
  let config: FunASRConfig

  @ModuleInfo(key: "audio_encoder") var audioEncoder: SenseVoiceEncoder
  @ModuleInfo(key: "audio_adaptor") var audioAdaptor: AudioAdaptor
  var llm: Qwen3ForCausalLM

  // Path to the model directory (for loading tokenizer)
  var modelDirectory: URL?

  init(config: FunASRConfig) {
    self.config = config

    // Audio encoder
    _audioEncoder.wrappedValue = SenseVoiceEncoder(config: config.encoder)

    // Audio adaptor
    _audioAdaptor.wrappedValue = AudioAdaptor(config: config.adaptor)

    // LLM decoder
    llm = Qwen3ForCausalLM(config: config.llm)
  }

  /// Encode audio to embeddings, sliced to the audio-token count the LLM
  /// expects.
  ///
  /// Matches the upstream PyTorch path: when `use_low_frame_rate=true` the
  /// model only consumes the first `fake_token_len` frames of the adaptor
  /// output (three successive 2× downsamples on the LFR frame count).
  /// Inserting all `T` frames — as a naive port would — gives the LLM ~8× the
  /// embeddings it was trained on, which manifests as repeated tokens / high
  /// CER on long audio.
  ///
  /// - Parameter audio: Audio waveform (T,) at 16kHz
  /// - Returns: Audio embeddings (1, fakeTokenLen, llmDim)
  func encodeAudio(_ audio: MLXArray) -> MLXArray {
    // Preprocess audio. CMVN is intentionally disabled by default to match the
    // upstream config (`cmvn_file: null`); enabling per-utterance CMVN
    // shifts/scales the features in a way the model wasn't trained for.
    let features = preprocessAudio(
      audio,
      nMels: config.nMels,
      lfrM: config.lfrM,
      lfrN: config.lfrN,
      applyNormalization: false,
      dither: 1.0,
    )

    // Add batch dimension
    var batchedFeatures = features
    if features.ndim == 2 {
      batchedFeatures = features.expandedDimensions(axis: 0)
    }

    // Encode
    let (encoderOut, lengths) = audioEncoder(batchedFeatures, lengths: nil)

    // Adapt to LLM space
    let (adapted, _) = audioAdaptor(encoderOut, lengths: lengths)

    // Slice the adaptor output to the audio-token count the LLM expects.
    let lfrFrameCount = features.shape[0]
    let audioTokenLen: Int = if config.adaptor.useLowFrameRate {
      Self.fakeTokenLen(lfrFrameCount: lfrFrameCount)
    } else {
      adapted.shape[1]
    }
    let truncatedLen = min(audioTokenLen, adapted.shape[1])
    return adapted[0..., 0 ..< truncatedLen, 0...]
  }

  /// Audio-token count from the LFR-processed frame count.
  ///
  /// Mirrors the upstream PyTorch formula in `fun_asr_nano/model.py`:
  ///
  /// ```python
  /// olens = 1 + (T - 3 + 2 * 1) // 2
  /// olens = 1 + (olens - 3 + 2 * 1) // 2
  /// fake_token_len = (olens - 1) // 2 + 1
  /// ```
  ///
  /// Three successive 2× downsamples (kernel=3, stride=2, padding=1).
  static func fakeTokenLen(lfrFrameCount T: Int) -> Int {
    let olens1 = 1 + (T - 1) / 2
    let olens2 = 1 + (olens1 - 1) / 2
    return (olens2 - 1) / 2 + 1
  }

  /// Splice the audio embeddings between the pre-audio and post-audio prompt
  /// tokens.
  ///
  /// The LLM sees `embed(preTokens) | audioEmbeddings | embed(postTokens)`.
  /// This matches PyTorch's `data_load_speech` / `inference_prepare` flow
  /// without ever putting the literal speech-marker BPE pieces into the
  /// prompt.
  ///
  /// - Parameters:
  ///   - preTokens: Tokens before the audio (system + user instruction)
  ///   - audioEmbeddings: Audio-token embeddings, shape `(audioLen, llmDim)`
  ///     or `(1, audioLen, llmDim)`
  ///   - postTokens: Tokens after the audio (`<|im_end|><|im_start|>assistant\n`)
  /// - Returns: Combined embeddings `(1, total, llmDim)`
  func spliceEmbeddings(
    preTokens: [Int],
    audioEmbeddings: MLXArray,
    postTokens: [Int],
  ) -> MLXArray {
    let preIds = MLXArray(preTokens.map { Int32($0) })
    let postIds = MLXArray(postTokens.map { Int32($0) })
    let preEmb = llm.getInputEmbeddings()(preIds)
    let postEmb = llm.getInputEmbeddings()(postIds)
    let audio = audioEmbeddings.ndim == 3 ? audioEmbeddings.squeezed(axis: 0) : audioEmbeddings
    let combined = MLX.concatenated([preEmb, audio, postEmb], axis: 0)
    return combined.expandedDimensions(axis: 0)
  }

  /// Sample next token from logits
  ///
  /// - Parameters:
  ///   - logits: Logits from model (batch, seq, vocab)
  ///   - temperature: Sampling temperature (0 for greedy)
  ///   - topP: Top-p (nucleus) sampling threshold
  ///   - topK: Top-k sampling (0 to disable)
  /// - Returns: Sampled token IDs (batch,)
  func sampleNextToken(
    _ logits: MLXArray,
    temperature: Float = 0.0,
    topP: Float = 0.95,
    topK: Int = 0,
  ) -> MLXArray {
    // Get logits for last position - flatten to 1D for single batch
    var lastLogits = logits[0, -1, 0...]

    if temperature == 0 {
      // Greedy decoding
      return MLX.argMax(lastLogits, axis: -1)
    }

    // Apply temperature
    lastLogits = lastLogits / temperature

    // Apply top-k using argPartition
    let effectiveTopK = topK > 0 ? topK : 50

    // Get top-k indices
    let topKIndices = MLX.argPartition(-lastLogits, kth: effectiveTopK - 1)[0 ..< effectiveTopK]
    let topKLogits = lastLogits[topKIndices]

    // Apply top-p within top-k if specified
    var finalLogits = topKLogits
    var finalIndices = topKIndices

    if topP < 1.0 {
      let probs = MLX.softmax(topKLogits)
      let sortedIndices = MLX.argSort(-probs)
      let sortedProbs = probs[sortedIndices]
      let cumsumProbs = MLX.cumsum(sortedProbs)

      // Find cutoff
      let belowThreshold = cumsumProbs .< topP
      let nTokens = max(1, Int(MLX.sum(belowThreshold).item(Int32.self)) + 1)

      let selectedSortedIndices = sortedIndices[0 ..< nTokens]
      finalIndices = topKIndices[selectedSortedIndices]
      finalLogits = lastLogits[finalIndices]
    }

    // Sample from filtered distribution
    let probs = MLX.softmax(finalLogits)
    let idx = MLXRandom.categorical(MLX.log(probs + 1e-10))

    return finalIndices[idx]
  }

  /// Sanitize weights for loading
  ///
  /// Handles Conv1d weight transposition and key remapping.
  ///
  /// - Parameter weights: Raw weights dictionary
  /// - Returns: Sanitized weights
  static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var sanitized: [String: MLXArray] = [:]

    for (k, var v) in weights {
      // Handle FSMN conv weights (PyTorch: [out, 1, kernel] -> MLX: [out, kernel, 1])
      if k.contains("fsmn_block"), k.contains("weight") {
        // Check if transposition is needed
        if v.ndim == 3, v.shape[1] == 1 {
          // Squeeze middle dimension and add at end: (out, 1, kernel) -> (out, kernel, 1)
          v = v.squeezed(axis: 1).expandedDimensions(axis: -1)
        }
      }
      // Handle other conv weights that might need transposition
      else if k.contains("conv"), k.contains("weight") {
        if v.ndim == 3 {
          // Check shape to determine if transposition needed
          if v.shape[2] < v.shape[1] {
            v = v.swappedAxes(-1, -2)
          }
        }
      }

      sanitized[k] = v
    }

    return sanitized
  }

  /// Load Fun-ASR model from a local directory
  ///
  /// - Parameters:
  ///   - variant: Model variant to load
  ///   - progressHandler: Optional callback for download progress
  /// - Returns: Initialized model with loaded weights
  /// Load Fun-ASR model from a local directory
  static func load(
    from directory: URL,
    variant: FunASRModelVariant = .nano4bit,
  ) throws -> FunASRModel {
    // Load config
    let configURL = directory.appending(path: "config.json")
    let config: FunASRConfig = if FileManager.default.fileExists(atPath: configURL.path) {
      try FunASRConfig.load(from: configURL)
    } else {
      FunASRConfig()
    }

    // Initialize model
    let model = FunASRModel(config: config)

    // Find and load weights
    let weightFiles = try FileManager.default.contentsOfDirectory(
      at: directory,
      includingPropertiesForKeys: nil,
    ).filter { $0.pathExtension == "safetensors" }

    guard !weightFiles.isEmpty else {
      throw STTError.modelUnavailable("No safetensors files found in \(directory.path)")
    }

    var allWeights: [String: MLXArray] = [:]
    for weightFile in weightFiles {
      let weights = try MLX.loadArrays(url: weightFile)
      allWeights.merge(weights) { _, new in new }
    }

    // Sanitize weights (handle conv transposition)
    let sanitizedWeights = sanitize(allWeights)

    // Apply quantization if weights are quantized and quantization level specifies bits
    let isQuantized = sanitizedWeights.keys.contains { $0.contains(".scales") }
    if isQuantized, let bits = variant.quantization.bits {
      Log.model.info("Detected quantized Fun-ASR model weights (\(bits)-bit)")
      quantize(model: model) { path, _ in
        sanitizedWeights["\(path).scales"] != nil ? (64, bits, .affine) : nil
      }
    }

    // Load weights into model
    let parameters = ModuleParameters.unflattened(sanitizedWeights)
    try model.update(parameters: parameters, verify: [.noUnusedKeys])

    // Set to eval mode
    model.train(false)

    // Evaluate to ensure weights are loaded
    eval(model)

    // Store model directory for tokenizer
    model.modelDirectory = directory

    Log.model.info("Fun-ASR model loaded successfully")

    return model
  }

  /// Download and load Fun-ASR model
  static func load(
    variant: FunASRModelVariant = .nano4bit,
    from downloader: any Downloader,
    progressHandler: @escaping @Sendable (Progress) -> Void = { _ in },
  ) async throws -> FunASRModel {
    let repoId = variant.repoId
    Log.model.info("Loading Fun-ASR from \(repoId)...")

    let modelDirectory = try await downloader.download(
      id: repoId,
      revision: nil,
      matching: [
        "*.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
      ],
      useLatest: false,
      progressHandler: progressHandler,
    )

    return try load(from: modelDirectory, variant: variant)
  }
}
