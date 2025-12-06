import Foundation
import MLX
import Testing

@testable import MLXAudio

struct ChatterboxTests {
  @Test @MainActor func chatterboxEngineInitializes() async {
    let engine = ChatterboxEngine()
    #expect(engine.isLoaded == false)
    #expect(engine.isGenerating == false)
    #expect(engine.provider == .chatterbox)
  }

  @Test @MainActor func chatterboxEngineLoadsModel() async throws {
    let engine = ChatterboxEngine()

    try await engine.load { progress in
      print("Loading: \(Int(progress.fractionCompleted * 100))%")
    }

    #expect(engine.isLoaded == true)
    print("Chatterbox model loaded successfully")
  }

  @Test @MainActor func chatterboxGeneratesAudio() async throws {
    let engine = ChatterboxEngine()

    // Load model
    print("Loading Chatterbox model...")
    try await engine.load()
    #expect(engine.isLoaded == true)

    // Prepare reference audio (using default sample)
    print("Preparing reference audio...")
    let referenceAudio = try await engine.prepareDefaultReferenceAudio()
    print("Reference audio prepared: \(referenceAudio.description)")

    // Generate speech
    let testText = "Hello, this is a test of the Chatterbox text to speech system."
    print("Generating speech for: \"\(testText)\"")

    let result = try await engine.generate(testText, referenceAudio: referenceAudio)

    switch result {
      case let .samples(data, sampleRate, processingTime):
        print("Generated \(data.count) samples at \(sampleRate)Hz in \(String(format: "%.2f", processingTime)) sec.")
        #expect(data.count > 0)
        #expect(sampleRate == ChatterboxS3GenSr)

        let audioDuration = Double(data.count) / Double(sampleRate)
        let rtf = processingTime / audioDuration
        print("Audio duration: \(String(format: "%.2f", audioDuration)) sec., RTF: \(String(format: "%.2f", rtf))")

      case .file:
        Issue.record("Expected samples result, got file")
    }

    #expect(engine.generationTime > 0)
    print("Chatterbox generation test passed!")
  }

  @Test @MainActor func chatterboxMultipleSpeakers() async throws {
    let engine = ChatterboxEngine()

    // Load model
    print("Loading Chatterbox model...")
    try await engine.load()

    // Prepare default reference audio
    print("Preparing reference audio...")
    let speaker = try await engine.prepareDefaultReferenceAudio()

    // Generate with the same speaker multiple times (should be fast after first)
    print("Generating with speaker...")
    let result1 = try await engine.generate("First sentence.", referenceAudio: speaker)
    let result2 = try await engine.generate("Second sentence.", referenceAudio: speaker)

    // Verify both generated successfully
    if case let .samples(data1, _, _) = result1 {
      #expect(data1.count > 0)
    }
    if case let .samples(data2, _, _) = result2 {
      #expect(data2.count > 0)
    }

    print("Multiple speaker test passed!")
  }

  @Test func chatterboxModelDirectLoad() async throws {
    print("Testing direct ChatterboxModel.load()...")

    let model = try await ChatterboxModel.load { progress in
      if progress.fractionCompleted.truncatingRemainder(dividingBy: 0.1) < 0.01 {
        print("Loading: \(Int(progress.fractionCompleted * 100))%")
      }
    }

    print("Model loaded successfully")

    // Check model components are initialized
    // (t3, s3gen, ve are non-optional Module properties, so just access them to verify)
    _ = model.t3
    _ = model.s3gen
    _ = model.ve
    #expect(model.textTokenizer != nil)

    print("ChatterboxModel direct load test passed!")
  }
}
