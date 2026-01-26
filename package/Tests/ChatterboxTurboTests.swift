// Copyright © Anthony DePasquale

import AVFoundation
import Foundation
import Hub
import MLX
import MLXNN
import Testing

@testable import MLXAudio

// MARK: - Test Helper

/// Global shared model for ChatterboxTurbo tests
@MainActor
enum ChatterboxTurboTestHelper {
  /// Shared model instance - loaded once and reused
  private static var _sharedModel: ChatterboxTurboModel?

  /// Get or load the shared model (loads only once)
  static func getOrLoadModel() async throws -> ChatterboxTurboModel {
    if let model = _sharedModel {
      return model
    }
    print("[ChatterboxTurboTestHelper] Loading shared model (first time)...")
    let model = try await ChatterboxTurboModel.load()
    eval(model)
    _sharedModel = model
    print("[ChatterboxTurboTestHelper] Shared model loaded and cached")
    return model
  }

  /// Clear cached resources
  static func clearCache() {
    _sharedModel = nil
    print("[ChatterboxTurboTestHelper] Cache cleared")
  }
}

// MARK: - End-to-End Tests

@Suite(.serialized)
struct ChatterboxTurboTests {
  /// Reference audio from LJ Speech dataset (public domain)
  static let referenceAudioURL = URL(string: "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav")!

  /// Output directory for generated audio
  static let outputDir = URL(fileURLWithPath: "/tmp/chatterbox-turbo-test")

  /// Download audio from URL and return as MLXArray
  static func downloadAudio(from url: URL) async throws -> (audio: MLXArray, sampleRate: Int) {
    let cacheURL = try await TestAudioCache.downloadToFile(from: url)
    return try loadAudioFile(at: cacheURL)
  }

  /// Load audio file and return as MLXArray with its sample rate
  static func loadAudioFile(at url: URL) throws -> (audio: MLXArray, sampleRate: Int) {
    let file = try AVAudioFile(forReading: url)

    guard let buffer = AVAudioPCMBuffer(
      pcmFormat: file.processingFormat,
      frameCapacity: AVAudioFrameCount(file.length)
    ) else {
      throw TestError(message: "Failed to create buffer")
    }

    try file.read(into: buffer)

    guard let floatData = buffer.floatChannelData else {
      throw TestError(message: "No float data in buffer")
    }

    let frameCount = Int(buffer.frameLength)
    let channelCount = Int(buffer.format.channelCount)

    // Convert to mono if stereo
    var samples = [Float](repeating: 0, count: frameCount)
    if channelCount == 1 {
      for i in 0 ..< frameCount {
        samples[i] = floatData[0][i]
      }
    } else {
      // Average channels for mono
      for i in 0 ..< frameCount {
        var sum: Float = 0
        for ch in 0 ..< channelCount {
          sum += floatData[ch][i]
        }
        samples[i] = sum / Float(channelCount)
      }
    }

    let sampleRate = Int(file.fileFormat.sampleRate)
    return (audio: MLXArray(samples), sampleRate: sampleRate)
  }

  /// Save audio to WAV file
  static func saveAudio(_ samples: [Float], to url: URL, sampleRate: Int = 24000) throws {
    // Create output directory if needed
    try FileManager.default.createDirectory(
      at: url.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )

    let format = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate), channels: 1)!
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
    buffer.frameLength = AVAudioFrameCount(samples.count)

    for (i, sample) in samples.enumerated() {
      buffer.floatChannelData![0][i] = sample
    }

    let file = try AVAudioFile(forWriting: url, settings: format.settings)
    try file.write(from: buffer)
  }

  /// Test model loading
  @Test @MainActor func testModelLoading() async throws {
    print("=== ChatterboxTurbo Model Loading Test ===\n")

    let loadStart = CFAbsoluteTimeGetCurrent()
    let model = try await ChatterboxTurboTestHelper.getOrLoadModel()
    let loadTime = CFAbsoluteTimeGetCurrent() - loadStart

    print("Model loaded in \(String(format: "%.2f", loadTime))s")

    // Verify components
    #expect(model.textTokenizer != nil, "Text tokenizer should be loaded")
    #expect(model.s3Tokenizer != nil, "S3 tokenizer should be loaded")

    // Check if pre-computed conditionals are available
    if model.conds != nil {
      print("Pre-computed conditionals available")
    }

    print("\nChatterboxTurbo model loading test passed!")
  }

  /// End-to-end audio generation test
  @Test @MainActor func testAudioGeneration() async throws {
    print("=== ChatterboxTurbo End-to-End Audio Generation Test ===\n")

    // Load model
    print("Step 1: Loading model...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let model = try await ChatterboxTurboTestHelper.getOrLoadModel()
    print("  Model loaded in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - loadStart))s")

    // Download reference audio
    print("\nStep 2: Downloading reference audio (LJ Speech)...")
    let (refAudio, refSampleRate) = try await Self.downloadAudio(from: Self.referenceAudioURL)
    let refDuration = Float(refAudio.shape[0]) / Float(refSampleRate)
    print("  Reference audio: \(refAudio.shape[0]) samples at \(refSampleRate)Hz (\(String(format: "%.1f", refDuration))s)")

    // Prepare conditionals from reference audio
    print("\nStep 3: Preparing voice conditionals...")
    let condStart = CFAbsoluteTimeGetCurrent()
    let conditionals = model.prepareConditionals(refWav: refAudio, refSr: refSampleRate)
    eval(conditionals.t3.speakerEmb)
    eval(conditionals.gen.embedding)
    print("  Conditionals prepared in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - condStart))s")

    // Create output directory
    try FileManager.default.createDirectory(at: Self.outputDir, withIntermediateDirectories: true)

    // Generate audio
    let testText = "Hello, this is a test of the Chatterbox Turbo text to speech system."
    print("\nStep 4: Generating speech...")
    print("  Text: \"\(testText)\"")

    let genStart = CFAbsoluteTimeGetCurrent()
    let audio = model.generate(
      text: testText,
      conds: conditionals,
      temperature: 0.8,
      repetitionPenalty: 1.2,
      topP: 0.95,
      topK: 1000
    )
    eval(audio)
    let genTime = CFAbsoluteTimeGetCurrent() - genStart

    // Get audio samples
    let audioSamples = audio.asArray(Float.self)
    let audioDuration = Float(audioSamples.count) / Float(model.sampleRate)
    let rtf = Float(genTime) / audioDuration

    print("  Generated \(audioSamples.count) samples (\(String(format: "%.2f", audioDuration))s)")
    print("  Generation time: \(String(format: "%.2f", genTime))s")
    print("  RTF: \(String(format: "%.2fx", rtf))")

    // Save audio
    let outputURL = Self.outputDir.appendingPathComponent("chatterbox_turbo_test.wav")
    try Self.saveAudio(audioSamples, to: outputURL, sampleRate: model.sampleRate)
    print("\nStep 5: Audio saved to: \(outputURL.path)")
    print("Open with: open \"\(outputURL.path)\"")

    // Assertions
    #expect(audioSamples.count > 0, "Audio should have samples")
    #expect(audioDuration > 0.5, "Audio should be at least 0.5 seconds")
    #expect(rtf < 10.0, "RTF should be under 10x (got \(String(format: "%.1f", rtf)))")

    print("\nChatterboxTurbo end-to-end test passed!")
  }

  /// Test multiple generations with cached conditionals
  @Test @MainActor func testMultipleGenerations() async throws {
    print("=== ChatterboxTurbo Multiple Generations Test ===\n")

    // Load model
    let model = try await ChatterboxTurboTestHelper.getOrLoadModel()

    // Download and prepare conditionals
    print("Preparing voice conditionals...")
    let (refAudio, refSampleRate) = try await Self.downloadAudio(from: Self.referenceAudioURL)
    let conditionals = model.prepareConditionals(refWav: refAudio, refSr: refSampleRate)
    eval(conditionals.t3.speakerEmb)
    eval(conditionals.gen.embedding)

    // Create output directory
    try FileManager.default.createDirectory(at: Self.outputDir, withIntermediateDirectories: true)

    // Generate multiple sentences
    let sentences = [
      "First sentence with cached conditionals.",
      "Second sentence should be faster.",
      "Third and final test sentence.",
    ]

    for (i, text) in sentences.enumerated() {
      print("\nGenerating sentence \(i + 1): \"\(text)\"")
      let start = CFAbsoluteTimeGetCurrent()

      let audio = model.generate(text: text, conds: conditionals)
      eval(audio)

      let genTime = CFAbsoluteTimeGetCurrent() - start
      let samples = audio.asArray(Float.self)
      let duration = Float(samples.count) / Float(model.sampleRate)
      let rtf = Float(genTime) / duration

      print("  Duration: \(String(format: "%.2f", duration))s, Time: \(String(format: "%.2f", genTime))s, RTF: \(String(format: "%.2fx", rtf))")

      // Save each audio file
      let outputURL = Self.outputDir.appendingPathComponent("chatterbox_turbo_multi_\(i + 1).wav")
      try Self.saveAudio(samples, to: outputURL, sampleRate: model.sampleRate)
      print("  Saved: \(outputURL.path)")

      #expect(samples.count > 0, "Sentence \(i + 1) should have audio")
    }

    print("\nMultiple generations test passed!")
  }

  /// Test q8 model loading
  @Test @MainActor func testQ8ModelLoading() async throws {
    print("=== ChatterboxTurbo Q8 Model Loading Test ===\n")

    let loadStart = CFAbsoluteTimeGetCurrent()
    let model = try await ChatterboxTurboModel.load(quantization: .q8)
    let loadTime = CFAbsoluteTimeGetCurrent() - loadStart

    print("Q8 model loaded in \(String(format: "%.2f", loadTime))s")

    // Verify components
    #expect(model.textTokenizer != nil, "Text tokenizer should be loaded")
    #expect(model.s3Tokenizer != nil, "S3 tokenizer should be loaded")

    print("\nChatterboxTurbo q8 model loading test passed!")
  }
}
