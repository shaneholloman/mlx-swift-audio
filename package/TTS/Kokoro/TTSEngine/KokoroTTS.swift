// Copyright © Hexgrad (original model implementation)
// Ported to MLX from https://github.com/hexgrad/kokoro
// Copyright © 2024 Prince Canuma and contributors to Blaizzy/mlx-audio
// Copyright © Anthony DePasquale
// License: licenses/kokoro.txt

import Foundation
import MLX
import MLXAudio
import MLXLMCommon
import MLXNN
import Synchronization

/// Kokoro TTS actor providing thread-safe text-to-speech generation.
///
/// Use the static `load()` factory method to create an initialized instance.
actor KokoroTTS {
  enum KokoroTTSError: LocalizedError {
    case tooManyTokens
    case sentenceSplitError
    case audioGenerationError
    case voiceLoadFailed

    var errorDescription: String? {
      switch self {
        case .tooManyTokens:
          "Input text exceeds maximum token limit"
        case .sentenceSplitError:
          "Failed to split text into sentences"
        case .audioGenerationError:
          "Failed to generate audio"
        case .voiceLoadFailed:
          "Failed to load voice"
      }
    }
  }

  // MARK: - Constants

  /// Kokoro model's absolute maximum token capacity
  private static let maxTokenCount = 510
  /// Threshold for proactive splitting before hitting the hard limit.
  ///
  /// Using 450 (60 tokens below max) provides buffer for:
  /// - Phonemization variance: same text can produce slightly different token counts
  /// - Edge cases in token boundary handling
  /// - Ensuring we never accidentally exceed maxTokenCount after a split
  private static let safeTokenThreshold = 450
  private static let sampleRate = 24000

  // MARK: - Properties

  private let model: KokoroModel
  private let eSpeakEngine: ESpeakNGEngine
  private let kokoroTokenizer: KokoroTokenizer

  // MARK: - Initialization

  private init(
    model: KokoroModel,
    eSpeakEngine: ESpeakNGEngine,
    kokoroTokenizer: KokoroTokenizer,
  ) {
    self.model = model
    self.eSpeakEngine = eSpeakEngine
    self.kokoroTokenizer = kokoroTokenizer
  }

  /// Load and initialize a KokoroTTS instance from a local directory.
  static func load(
    from directory: URL
  ) async throws -> KokoroTTS {
    // Initialize text processing components
    let eSpeakEngine = try ESpeakNGEngine()
    let kokoroTokenizer = KokoroTokenizer(engine: eSpeakEngine)

    // Load lexicons from GitHub (cached on disk)
    async let usLexicon = LexiconLoader.loadUSLexicon()
    async let gbLexicon = LexiconLoader.loadGBLexicon()
    try await kokoroTokenizer.setLexicons(us: usLexicon, gb: gbLexicon)

    // Load weights from local directory
    let weights = try KokoroWeightLoader.loadWeights(from: directory)

    // Create model and load weights
    let model = KokoroModel()
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: .noUnusedKeys)

    return KokoroTTS(
      model: model,
      eSpeakEngine: eSpeakEngine,
      kokoroTokenizer: kokoroTokenizer,
    )
  }

  /// Download and load a KokoroTTS instance.
  static func load(
    id: String = KokoroWeightLoader.defaultRepoId,
    from downloader: any Downloader,
    progressHandler: @escaping @Sendable (Progress) -> Void = { _ in },
  ) async throws -> KokoroTTS {
    let directory = try await downloader.download(
      id: id,
      revision: nil,
      matching: [KokoroWeightLoader.defaultWeightsFilename],
      useLatest: false,
      progressHandler: progressHandler
    )

    return try await load(from: directory)
  }

  // MARK: - Public API

  /// Set the language for text processing based on the voice.
  func setLanguage(for voice: KokoroEngine.Voice) throws {
    try kokoroTokenizer.setLanguage(for: voice)
  }

  func generate(
    text: String,
    voiceData: MLXArray,
    speed: Float = 1.0,
  ) async throws -> TTSGenerationResult {
    let startTime = CFAbsoluteTimeGetCurrent()

    let sentences = SentenceTokenizer.splitIntoSentences(text: text)
    if sentences.isEmpty {
      throw KokoroTTSError.sentenceSplitError
    }

    var allAudio: [Float] = []
    for sentence in sentences {
      // Check for cancellation between sentences
      try Task.checkCancellation()

      let audioChunks = try generateAudioChunks(text: sentence, voiceData: voiceData, speed: speed)
      for chunk in audioChunks {
        allAudio.append(contentsOf: chunk)
        MLXMemory.clearCache()
      }
    }

    let processingTime = CFAbsoluteTimeGetCurrent() - startTime
    return TTSGenerationResult(
      audio: allAudio,
      sampleRate: Self.sampleRate,
      processingTime: processingTime,
    )
  }

  func generateStreaming(
    text: String,
    voiceData: MLXArray,
    speed: Float = 1.0,
  ) async throws -> AsyncThrowingStream<[Float], Error> {
    let sentences = SentenceTokenizer.splitIntoSentences(text: text)
    if sentences.isEmpty {
      throw KokoroTTSError.sentenceSplitError
    }

    let sentenceIndex = Atomic<Int>(0)
    let pendingChunks = Mutex<[[Float]]>([])

    return AsyncThrowingStream {
      // Check if we have pending chunks from a previous split
      let nextPending = pendingChunks.withLock { chunks -> [Float]? in
        if !chunks.isEmpty {
          return chunks.removeFirst()
        }
        return nil
      }
      if let chunk = nextPending {
        MLXMemory.clearCache()
        return chunk
      }

      // Get next sentence
      let i = sentenceIndex.wrappingAdd(1, ordering: .relaxed).oldValue
      guard i < sentences.count else { return nil }

      // Check for cancellation before generating each sentence
      try Task.checkCancellation()

      let audioChunks = try await self.generateAudioChunks(text: sentences[i], voiceData: voiceData, speed: speed)

      guard !audioChunks.isEmpty else { return nil }

      // If multiple chunks, store extras for subsequent iterations
      if audioChunks.count > 1 {
        pendingChunks.withLock { pending in
          pending.append(contentsOf: audioChunks.dropFirst())
        }
      }

      MLXMemory.clearCache()
      return audioChunks[0]
    }
  }

  // MARK: - Private Methods

  /// Generate audio chunks for text, splitting if necessary.
  ///
  /// Returns an array of audio chunks to support proper streaming when text is split.
  private func generateAudioChunks(
    text: String,
    voiceData: MLXArray,
    speed: Float,
  ) throws -> [[Float]] {
    if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
      return [[0.0]]
    }

    let phonemizedResult = try kokoroTokenizer.phonemize(text)

    let inputIds = PhonemeTokenizer.tokenize(phonemizedText: phonemizedResult.phonemes)

    // If tokens exceed the safe threshold, split the text and process recursively
    if inputIds.count > Self.safeTokenThreshold {
      Log.tts.debug("KokoroTTS: Text exceeds safe token threshold (\(inputIds.count) > \(Self.safeTokenThreshold)), splitting...")

      if let (firstHalf, secondHalf) = TextSplitter.splitAtPunctuationBoundary(text) {
        Log.tts.debug("KokoroTTS: Split into '\(firstHalf.prefix(30))...' and '\(secondHalf.prefix(30))...'")

        let firstChunks = try generateAudioChunks(text: firstHalf, voiceData: voiceData, speed: speed)
        let secondChunks = try generateAudioChunks(text: secondHalf, voiceData: voiceData, speed: speed)

        return firstChunks + secondChunks
      } else {
        guard inputIds.count <= Self.maxTokenCount else {
          Log.tts.error("KokoroTTS: Text too long and cannot be split (\(inputIds.count) tokens)")
          throw KokoroTTSError.tooManyTokens
        }
        Log.tts.warning("KokoroTTS: Could not split text, proceeding with \(inputIds.count) tokens")
      }
    }

    return try [generateAudioForTokens(inputIds: inputIds, voiceData: voiceData, speed: speed)]
  }

  private func generateAudioForTokens(
    inputIds: [Int],
    voiceData: MLXArray,
    speed: Float,
  ) throws -> [Float] {
    let paddedInputIdsBase = [0] + inputIds + [0]
    let paddedInputIds = MLXArray(paddedInputIdsBase).expandedDimensions(axes: [0])
    paddedInputIds.eval()

    let inputLengths = MLXArray(paddedInputIds.dim(-1))
    inputLengths.eval()

    let inputLengthMax: Int = MLX.max(inputLengths).item()
    var textMask = MLXArray(0 ..< inputLengthMax)
    textMask.eval()

    textMask = textMask + 1 .> inputLengths
    textMask.eval()

    textMask = textMask.expandedDimensions(axes: [0])
    textMask.eval()

    let swiftTextMask: [Bool] = textMask.asArray(Bool.self)
    let swiftTextMaskInt = swiftTextMask.map { !$0 ? 1 : 0 }
    let attentionMask = MLXArray(swiftTextMaskInt).reshaped(textMask.shape)
    attentionMask.eval()

    let (bertDur, _) = model.bert(paddedInputIds, attentionMask: attentionMask)
    bertDur.eval()

    let dEn = model.bertEncoder(bertDur).transposed(0, 2, 1)
    dEn.eval()

    // Voice shape is [510, 1, 256], index by phoneme length to get [1, 256]
    let voiceIdx = min(inputIds.count - 1, voiceData.shape[0] - 1)
    let refS = voiceData[voiceIdx]
    refS.eval()

    // Extract style vector: columns 128+ for duration/prosody prediction
    let s = refS[0..., 128...]
    s.eval()

    let d = model.predictor.textEncoder(dEn, style: s, textLengths: inputLengths, m: textMask)
    d.eval()

    let (x, _) = model.predictor.lstm(d)
    x.eval()

    let duration = model.predictor.durationProj(x)
    duration.eval()

    let durationSigmoid = MLX.sigmoid(duration).sum(axis: -1) / speed
    durationSigmoid.eval()

    let predDur = MLX.clip(durationSigmoid.round(), min: 1).asType(.int32)[0]
    predDur.eval()

    // Index and matrix generation
    // Build indices in chunks to reduce memory
    var allIndices: [MLXArray] = []
    let chunkSize = 50

    for startIdx in stride(from: 0, to: predDur.shape[0], by: chunkSize) {
      let endIdx = min(startIdx + chunkSize, predDur.shape[0])
      let chunkIndices = predDur[startIdx ..< endIdx]

      let indices = MLX.concatenated(
        chunkIndices.enumerated().map { i, n in
          let nSize: Int = n.item()
          let arrayIndex = MLXArray([i + startIdx])
          arrayIndex.eval()
          let repeated = MLX.repeated(arrayIndex, count: nSize)
          repeated.eval()
          return repeated
        },
      )
      indices.eval()
      allIndices.append(indices)
    }

    let indices = MLX.concatenated(allIndices)
    indices.eval()

    allIndices.removeAll()

    let indicesShape = indices.shape[0]
    let inputIdsShape = paddedInputIds.shape[1]

    // Create sparse matrix using COO format
    var rowIndices: [Int] = []
    var colIndices: [Int] = []

    // Reserve capacity to avoid reallocations
    let estimatedNonZeros = min(indicesShape, inputIdsShape * 5)
    rowIndices.reserveCapacity(estimatedNonZeros)
    colIndices.reserveCapacity(estimatedNonZeros)

    // Process in batches
    let batchSize = 256
    for startIdx in stride(from: 0, to: indicesShape, by: batchSize) {
      let endIdx = min(startIdx + batchSize, indicesShape)
      for i in startIdx ..< endIdx {
        let indiceValue: Int = indices[i].item()
        if indiceValue < inputIdsShape {
          rowIndices.append(indiceValue)
          colIndices.append(i)
        }
      }
    }

    // Create dense matrix from COO data
    var swiftPredAlnTrg = [Float](repeating: 0.0, count: inputIdsShape * indicesShape)
    let matrixBatchSize = 1000
    for startIdx in stride(from: 0, to: rowIndices.count, by: matrixBatchSize) {
      let endIdx = min(startIdx + matrixBatchSize, rowIndices.count)
      for i in startIdx ..< endIdx {
        let row = rowIndices[i]
        let col = colIndices[i]
        if row < inputIdsShape, col < indicesShape {
          swiftPredAlnTrg[row * indicesShape + col] = 1.0
        }
      }
    }

    // Create MLXArray from the dense matrix
    let predAlnTrg = MLXArray(swiftPredAlnTrg).reshaped([inputIdsShape, indicesShape])
    predAlnTrg.eval()

    // Clear Swift arrays
    swiftPredAlnTrg = []
    rowIndices = []
    colIndices = []

    let predAlnTrgBatched = predAlnTrg.expandedDimensions(axis: 0)
    predAlnTrgBatched.eval()

    let en = d.transposed(0, 2, 1).matmul(predAlnTrgBatched)
    en.eval()

    let (F0Pred, NPred) = model.predictor.F0NTrain(x: en, s: s)
    F0Pred.eval()
    NPred.eval()

    let tEn = model.textEncoder(paddedInputIds, inputLengths: inputLengths, m: textMask)
    tEn.eval()

    let asr = MLX.matmul(tEn, predAlnTrg)
    asr.eval()

    // Extract style vector: columns 0-127 for decoder
    let voiceS = refS[0..., ..<128]
    voiceS.eval()

    let audio = model.decoder(asr: asr, F0Curve: F0Pred, N: NPred, s: voiceS)[0]
    audio.eval()

    let audioShape = audio.shape

    // Check if the audio shape is valid
    let totalSamples: Int = if audioShape.count == 1 {
      audioShape[0]
    } else if audioShape.count == 2 {
      audioShape[1]
    } else {
      0
    }

    if totalSamples <= 1 {
      Log.tts.error("KokoroTTS: Invalid audio shape - totalSamples: \(totalSamples), shape: \(audioShape)")
      throw KokoroTTSError.audioGenerationError
    }

    return audio.asArray(Float.self)
  }
}
