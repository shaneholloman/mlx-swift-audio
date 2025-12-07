import Foundation

/// Available TTS providers
public enum TTSProvider: String, CaseIterable, Identifiable, Sendable {
  case kokoro
  case orpheus
  case marvis
  case outetts
  case chatterbox

  public var id: String { rawValue }

  // MARK: - Display Properties

  /// Human-readable name for UI display
  public var displayName: String {
    switch self {
      case .outetts: "OuteTTS"
      case .chatterbox: "Chatterbox"
      default: rawValue.capitalized
    }
  }

  /// Description of the provider's capabilities
  public var description: String {
    switch self {
      case .kokoro:
        "Fast, lightweight TTS with many voices"
      case .orpheus:
        "High quality with emotional expressions"
      case .marvis:
        "Advanced conversational TTS with streaming"
      case .outetts:
        "TTS with speaker profiles"
      case .chatterbox:
        "TTS with reference audio support"
    }
  }

  /// Status message shown in the UI (warnings, tips, etc.)
  public var statusMessage: String {
    switch self {
      case .kokoro:
        ""
      case .orpheus:
        "Supports expressions: <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>"
      case .marvis:
        "Marvis: Advanced conversational TTS with streaming support.\n\nNote: Downloads model weights on first use."
      case .outetts:
        "OuteTTS: Supports custom speaker profiles."
      case .chatterbox:
        "Chatterbox: TTS with reference audio support.\n\nNote: Downloads model weights on first use."
    }
  }

  // MARK: - Audio Properties

  /// Sample rate for this provider's audio output (Hz)
  public var sampleRate: Int {
    switch self {
      case .kokoro: 24000
      case .orpheus: 24000
      case .marvis: 24000
      case .outetts: 24000
      case .chatterbox: 24000
    }
  }

  // MARK: - Feature Flags

  /// Whether this provider supports speed adjustment
  public var supportsSpeed: Bool {
    self == .kokoro
  }

  /// Whether this provider supports emotional expressions
  public var supportsExpressions: Bool {
    self == .orpheus
  }

  /// Whether this provider supports quality level selection
  public var supportsQualityLevels: Bool {
    self == .marvis
  }

  public var supportsReferenceAudio: Bool {
    self == .outetts || self == .chatterbox
  }
}
