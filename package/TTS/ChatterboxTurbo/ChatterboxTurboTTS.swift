// Copyright © 2025 Resemble AI (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/resemble-ai/chatterbox
// License: licenses/chatterbox.txt

import Foundation
import MLX
import Synchronization

/// Actor wrapper for ChatterboxTurboModel that provides thread-safe generation
actor ChatterboxTurboTTS {
  // MARK: - Properties

  // Model is nonisolated(unsafe) because it contains non-Sendable types (MLXArray)
  // but is only accessed within the actor's methods
  private nonisolated(unsafe) let model: ChatterboxTurboModel

  // MARK: - Constants

  /// Maximum character count for a single chunk before splitting.
  ///
  /// Chatterbox Turbo's T3 model tends to hit its 1000-token generation limit with long text,
  /// resulting in truncated audio. Empirically, 250 characters keeps generation well
  /// within limits while maintaining natural speech flow.
  private static let maxChunkCharacters = 250

  // MARK: - Initialization

  private init(model: ChatterboxTurboModel) {
    self.model = model
  }

  /// Load ChatterboxTurboTTS from Hugging Face Hub
  ///
  /// - Parameters:
  ///   - quantization: Quantization level (fp16, 8bit, 4bit). Default is 4bit.
  ///   - progressHandler: Optional callback for download progress
  /// - Returns: Initialized ChatterboxTurboTTS instance
  static func load(
    quantization: ChatterboxTurboQuantization = .q4,
    progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
  ) async throws -> ChatterboxTurboTTS {
    let model = try await ChatterboxTurboModel.load(
      quantization: quantization,
      progressHandler: progressHandler
    )
    return ChatterboxTurboTTS(model: model)
  }

  // MARK: - Conditionals

  /// Prepare conditioning from reference audio
  ///
  /// Returns the pre-computed conditionals that can be reused across multiple generation calls.
  /// This is the expensive operation that extracts voice characteristics from reference audio.
  ///
  /// Note: Chatterbox Turbo does not support emotion exaggeration.
  ///
  /// - Parameters:
  ///   - refWav: Reference audio waveform
  ///   - refSr: Sample rate of the reference audio
  /// - Returns: Pre-computed conditionals for generation
  func prepareConditionals(
    refWav: MLXArray,
    refSr: Int
  ) -> ChatterboxTurboConditionals {
    model.prepareConditionals(
      refWav: refWav,
      refSr: refSr
    )
  }

  // MARK: - Generation

  /// Generate audio from text using pre-computed conditionals
  ///
  /// This runs on the actor's background executor, not blocking the main thread.
  /// Long text is automatically split into chunks and processed separately to avoid truncation.
  ///
  /// Note: Unlike the original Chatterbox, Turbo does not support exaggeration or cfgWeight.
  ///
  /// - Parameters:
  ///   - text: Text to synthesize
  ///   - conditionals: Pre-computed reference audio conditionals
  ///   - temperature: Sampling temperature
  ///   - repetitionPenalty: Penalty for repeated tokens
  ///   - topP: Top-p sampling threshold
  ///   - topK: Top-k sampling value
  ///   - maxNewTokens: Maximum tokens to generate per chunk
  /// - Returns: Generated audio result
  func generate(
    text: String,
    conditionals: ChatterboxTurboConditionals,
    temperature: Float = 0.8,
    repetitionPenalty: Float = 1.2,
    topP: Float = 0.95,
    topK: Int = 1000,
    maxNewTokens: Int = 1000
  ) -> TTSGenerationResult {
    let startTime = CFAbsoluteTimeGetCurrent()

    // Split text into manageable chunks
    let chunks = splitTextForGeneration(text)

    var allAudio: [Float] = []

    for chunk in chunks {
      // Check for cancellation between chunks
      if Task.isCancelled {
        break
      }

      let audioArray = model.generate(
        text: chunk,
        conds: conditionals,
        temperature: temperature,
        repetitionPenalty: repetitionPenalty,
        topP: topP,
        topK: topK,
        maxNewTokens: maxNewTokens
      )

      // Ensure computation is complete
      audioArray.eval()

      allAudio.append(contentsOf: audioArray.asArray(Float.self))

      // Clear GPU memory between chunks
      MLXMemory.clearCache()
    }

    let processingTime = CFAbsoluteTimeGetCurrent() - startTime

    return TTSGenerationResult(
      audio: allAudio,
      sampleRate: sampleRate,
      processingTime: processingTime
    )
  }

  /// Output sample rate
  var sampleRate: Int {
    ChatterboxTurboS3GenSr
  }

  // MARK: - Streaming Generation

  /// Generate audio from text as a stream of chunks.
  ///
  /// Text is split into sentences, then oversized sentences are further split.
  /// Each chunk is yielded as it's generated for streaming playback.
  ///
  /// - Parameters:
  ///   - text: Text to synthesize
  ///   - conditionals: Pre-computed reference audio conditionals
  ///   - temperature: Sampling temperature
  ///   - repetitionPenalty: Penalty for repeated tokens
  ///   - topP: Top-p sampling threshold
  ///   - topK: Top-k sampling value
  ///   - maxNewTokens: Maximum tokens to generate per chunk
  /// - Returns: An async stream of audio sample chunks
  func generateStreaming(
    text: String,
    conditionals: ChatterboxTurboConditionals,
    temperature: Float = 0.8,
    repetitionPenalty: Float = 1.2,
    topP: Float = 0.95,
    topK: Int = 1000,
    maxNewTokens: Int = 1000
  ) -> AsyncThrowingStream<[Float], Error> {
    let chunks = splitTextForGeneration(text)
    let chunkIndex = Atomic<Int>(0)

    return AsyncThrowingStream {
      let i = chunkIndex.wrappingAdd(1, ordering: .relaxed).oldValue
      guard i < chunks.count else { return nil }

      try Task.checkCancellation()

      let audioArray = self.model.generate(
        text: chunks[i],
        conds: conditionals,
        temperature: temperature,
        repetitionPenalty: repetitionPenalty,
        topP: topP,
        topK: topK,
        maxNewTokens: maxNewTokens
      )

      audioArray.eval()
      let samples = audioArray.asArray(Float.self)
      MLXMemory.clearCache()
      return samples
    }
  }

  // MARK: - Private Methods

  /// Split text into chunks suitable for generation.
  ///
  /// First splits into sentences using SentenceTokenizer, then splits any
  /// oversized sentences using TextSplitter.
  private func splitTextForGeneration(_ text: String) -> [String] {
    let sentences = SentenceTokenizer.splitIntoSentences(text: text)
    var result: [String] = []

    for sentence in sentences {
      let chunks = TextSplitter.splitToMaxLength(sentence, maxCharacters: Self.maxChunkCharacters)
      result.append(contentsOf: chunks)
    }

    return result.isEmpty ? [text] : result
  }
}
