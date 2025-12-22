// Copyright © Anthony DePasquale

import Kokoro
import MLXAudio
import SwiftUI

struct VoicePickerView: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    @Bindable var appState = appState
    switch appState.selectedProvider {
      case .kokoro:
        Picker("Voice", selection: $appState.kokoroVoice) {
          ForEach(KokoroEngine.Voice.allCases, id: \.self) { voice in
            Text(voice.displayName).tag(voice)
          }
        }
        .pickerStyle(.menu)
        .buttonStyle(.glass)
      case .orpheus:
        Picker("Voice", selection: $appState.orpheusVoice) {
          ForEach(OrpheusEngine.Voice.allCases, id: \.self) { voice in
            Text(voice.rawValue.capitalized).tag(voice)
          }
        }
        .pickerStyle(.menu)
        .buttonStyle(.glass)
      case .marvis:
        Picker("Voice", selection: $appState.marvisVoice) {
          ForEach(MarvisEngine.Voice.allCases, id: \.self) { voice in
            Text(voice.displayName).tag(voice)
          }
        }
        .pickerStyle(.menu)
        .buttonStyle(.glass)
      case .outetts:
        EmptyView()
      case .chatterbox:
        EmptyView()
      case .chatterboxTurbo:
        EmptyView()
      case .cosyVoice2:
        EmptyView()
      case .cosyVoice3:
        EmptyView()
    }
  }
}

// MARK: - Voice Display Name Extensions

extension KokoroEngine.Voice {
  var displayName: String {
    // Format: afHeart -> Heart
    let name = String(describing: self)
    guard name.count > 2 else { return name.capitalized }
    return String(name.dropFirst(2)).capitalized
  }
}

extension MarvisEngine.Voice {
  var displayName: String {
    switch self {
      case .conversationalA: "Conversational A"
      case .conversationalB: "Conversational B"
    }
  }
}
