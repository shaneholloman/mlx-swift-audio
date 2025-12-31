// Copyright © Anthony DePasquale

import SwiftUI

/// Display transcription text with copy functionality
struct TranscriptionTextView: View {
  let text: String
  let isStreaming: Bool

  @State private var showCopied = false

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack {
        Text("Transcription")
          .font(.headline)

        Spacer()

        if !text.isEmpty {
          Button(action: copyText) {
            HStack(spacing: 4) {
              Image(systemName: showCopied ? "checkmark" : "doc.on.doc")
              Text(showCopied ? "Copied" : "Copy")
            }
            .font(.caption)
          }
          .buttonStyle(.glass)
          .controlSize(.small)
        }
      }

      ScrollView {
        HStack {
          Text(text.isEmpty ? "Transcription will appear here..." : text)
            .foregroundStyle(text.isEmpty ? .secondary : .primary)
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)

          if isStreaming {
            // Typing cursor animation
            Rectangle()
              .fill(Color.primary)
              .frame(width: 2, height: 16)
              .opacity(isStreaming ? 1 : 0)
              .animation(.easeInOut(duration: 0.5).repeatForever(autoreverses: true), value: isStreaming)
          }

          Spacer(minLength: 0)
        }
        .padding()
      }
      .frame(minHeight: 100)
      .glassEffect(.regular, in: .rect(cornerRadius: 8))
    }
  }

  private func copyText() {
    #if os(macOS)
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(text, forType: .string)
    #else
    UIPasteboard.general.string = text
    #endif

    withAnimation {
      showCopied = true
    }

    DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
      withAnimation {
        showCopied = false
      }
    }
  }
}
