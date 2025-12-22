// Copyright © Anthony DePasquale

import MLXAudio
import SwiftUI

struct ProviderPickerView: View {
  let selectedProvider: TTSProvider
  let onSelect: (TTSProvider) -> Void

  var body: some View {
    Picker("Provider", selection: Binding(
      get: { selectedProvider },
      set: { onSelect($0) }
    )) {
      ForEach(TTSProvider.allCases) { provider in
        Text(provider.displayName).tag(provider)
      }
    }
    .pickerStyle(.menu)
    .buttonStyle(.glass)
  }
}
