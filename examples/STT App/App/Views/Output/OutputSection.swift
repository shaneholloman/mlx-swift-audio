// Copyright © Anthony DePasquale

import MLXAudio
import SwiftUI

/// Combined output section showing status, transcription, and segments
struct OutputSection: View {
  @Environment(AppState.self) private var appState

  private var transcriptionDisplayText: String {
    // Fun-ASR streaming tokens
    if appState.selectedProvider == .funASR, appState.isStreamingTokens {
      return appState.streamingText
    }
    // Whisper recording with periodic transcription
    if appState.isRecording, !appState.streamingSegments.isEmpty {
      return appState.streamingSegments.map { $0.text }.joined(separator: " ")
    }
    // Final result or Fun-ASR streaming text
    if appState.selectedProvider == .funASR, !appState.streamingText.isEmpty,
       appState.lastResult == nil
    {
      return appState.streamingText
    }
    return appState.lastResult?.text ?? ""
  }

  private var isStreamingTranscription: Bool {
    // Fun-ASR token streaming
    if appState.isStreamingTokens {
      return true
    }
    // Whisper periodic transcription during recording
    if appState.isRecording, !appState.streamingSegments.isEmpty {
      return true
    }
    // General transcribing state
    return appState.isTranscribing
  }

  var body: some View {
    VStack(spacing: 16) {
      // Status banner
      StatusBanner(
        message: appState.statusMessage,
        isLoading: appState.isTranscribing || appState.isModelLoading,
        isError: appState.error != nil,
        isStreaming: appState.isStreamingTokens
      )

      // Language detection result (Whisper only)
      if let result = appState.detectedLanguageResult {
        LanguageDetectionResult(language: result.language, confidence: result.confidence)
      }

      // Transcription text
      if appState.selectedTask != .detectLanguage {
        TranscriptionTextView(
          text: transcriptionDisplayText,
          isStreaming: isStreamingTranscription
        )

        // Segment list (Whisper only - Fun-ASR doesn't have timestamps)
        if appState.selectedProvider == .whisper {
          // Show streaming segments during recording
          if appState.isRecording, !appState.streamingSegments.isEmpty {
            SegmentListView(segments: appState.streamingSegments)
          }
          // Or show final segments after transcription
          else if let segments = appState.lastResult?.segments, !segments.isEmpty {
            SegmentListView(segments: segments)
          }
        }

        // Statistics (only for final result)
        if let result = appState.lastResult {
          TranscriptionStats(result: result, provider: appState.selectedProvider)
        }
      }
    }
    .padding()
  }
}

/// Display detected language with confidence
struct LanguageDetectionResult: View {
  let language: Language
  let confidence: Float

  var body: some View {
    VStack(spacing: 12) {
      HStack {
        Image(systemName: "globe")
          .font(.title)
          .foregroundStyle(.blue)

        VStack(alignment: .leading) {
          Text("Detected Language")
            .font(.caption)
            .foregroundStyle(.secondary)
          Text(language.displayName)
            .font(.title2)
            .fontWeight(.semibold)
        }

        Spacer()

        // Confidence indicator
        VStack(alignment: .trailing) {
          Text("Confidence")
            .font(.caption)
            .foregroundStyle(.secondary)
          Text("\(Int(confidence * 100))%")
            .font(.title3)
            .fontWeight(.medium)
            .foregroundStyle(confidenceColor)
        }
      }

      // Confidence bar
      GeometryReader { geometry in
        ZStack(alignment: .leading) {
          RoundedRectangle(cornerRadius: 4)
            .fill(Color.secondary.opacity(0.2))
            .frame(height: 8)

          RoundedRectangle(cornerRadius: 4)
            .fill(confidenceColor)
            .frame(width: geometry.size.width * CGFloat(confidence), height: 8)
        }
      }
      .frame(height: 8)
    }
    .padding()
    .background(
      RoundedRectangle(cornerRadius: 12)
        .fill(Color.blue.opacity(0.05))
    )
    .overlay(
      RoundedRectangle(cornerRadius: 12)
        .stroke(Color.blue.opacity(0.2), lineWidth: 1)
    )
  }

  private var confidenceColor: Color {
    if confidence >= 0.8 {
      .green
    } else if confidence >= 0.5 {
      .orange
    } else {
      .red
    }
  }
}

/// Transcription statistics display
struct TranscriptionStats: View {
  let result: TranscriptionResult
  let provider: STTProvider

  var body: some View {
    HStack(spacing: 16) {
      // Duration - only show if available (Whisper has it, Fun-ASR may not)
      if result.duration > 0 {
        StatItem(label: "Duration", value: formatDuration(result.duration))
        Divider().frame(height: 24)
      }

      StatItem(label: "Processing", value: formatDuration(result.processingTime))

      // RTF - only meaningful if duration is available
      if result.duration > 0, result.realTimeFactor > 0 {
        Divider().frame(height: 24)
        StatItem(label: "RTF", value: String(format: "%.2fx", result.realTimeFactor))
      }

      // Language - only show if detected
      if !result.language.isEmpty, result.language != "unknown" {
        Divider().frame(height: 24)
        StatItem(label: "Language", value: result.language.uppercased())
      }

      // Provider indicator
      Divider().frame(height: 24)
      StatItem(label: "Provider", value: provider.displayName)
    }
    .padding()
    .background(
      RoundedRectangle(cornerRadius: 8)
        .fill(Color.secondary.opacity(0.05))
    )
  }

  private func formatDuration(_ duration: TimeInterval) -> String {
    if duration < 60 {
      return String(format: "%.1fs", duration)
    }
    let minutes = Int(duration) / 60
    let seconds = Int(duration) % 60
    return String(format: "%d:%02d", minutes, seconds)
  }
}

/// Single stat item
struct StatItem: View {
  let label: String
  let value: String

  var body: some View {
    VStack(spacing: 2) {
      Text(label)
        .font(.caption)
        .foregroundStyle(.secondary)
      Text(value)
        .font(.subheadline)
        .fontWeight(.medium)
        .monospacedDigit()
    }
  }
}
