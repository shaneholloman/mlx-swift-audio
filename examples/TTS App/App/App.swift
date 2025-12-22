// Copyright © Anthony DePasquale

import SwiftUI
#if os(iOS)
import MLXAudio
#endif

@main
struct MLXAudioApp: App {
  init() {
    #if os(iOS)
    AudioSessionManager.configure()
    #endif
  }

  var body: some Scene {
    WindowGroup {
      ContentView()
    }
    #if os(macOS)
    .windowStyle(.automatic)
    .defaultSize(width: 900, height: 600)
    #endif
  }
}
