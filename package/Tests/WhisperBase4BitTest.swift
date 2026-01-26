// Copyright © Anthony DePasquale

import Foundation
import MLX
import Testing

@testable import MLXAudio

@Suite(.serialized)
struct WhisperBase4BitTest {
  @Test @MainActor func whisperBaseCompare() async throws {
    print("Comparing Whisper base fp16 vs 4-bit...")

    // Download test audio (cached)
    let audioURL = URL(string: "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav")!
    let testAudioURL = try await TestAudioCache.downloadToFile(from: audioURL)

    // Test fp16 first
    print("\n=== Testing base fp16 ===")
    let engineFp16 = STT.whisper(model: .base, quantization: .fp16)
    try await engineFp16.load()
    let resultFp16 = try await engineFp16.transcribe(testAudioURL, language: .english)
    print("FP16 Transcription: \(resultFp16.text)")
    print("FP16 RTF: \(String(format: "%.2fx", resultFp16.realTimeFactor))")
    await engineFp16.unload()

    // Test 4bit
    print("\n=== Testing base 4-bit ===")
    let engine4Bit = STT.whisper(model: .base, quantization: .q4)
    try await engine4Bit.load()
    let result4Bit = try await engine4Bit.transcribe(testAudioURL, language: .english)
    print("4-bit Transcription: \(result4Bit.text)")
    print("4-bit RTF: \(String(format: "%.2fx", result4Bit.realTimeFactor))")
    await engine4Bit.unload()

    // Test 8bit
    print("\n=== Testing base 8-bit ===")
    let engine8Bit = STT.whisper(model: .base, quantization: .q8)
    try await engine8Bit.load()
    let result8Bit = try await engine8Bit.transcribe(testAudioURL, language: .english)
    print("8-bit Transcription: \(result8Bit.text)")
    print("8-bit RTF: \(String(format: "%.2fx", result8Bit.realTimeFactor))")
    await engine8Bit.unload()

    // Verify all work
    #expect(!resultFp16.text.isEmpty, "FP16 should produce transcription")
    #expect(!result4Bit.text.isEmpty, "4-bit should produce transcription")
    #expect(!result8Bit.text.isEmpty, "8-bit should produce transcription")
  }
}
