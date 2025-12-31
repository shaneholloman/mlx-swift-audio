// Copyright © Anthony DePasquale

import SwiftUI

/// Status banner showing loading, processing, streaming, or error state
struct StatusBanner: View {
  let message: String
  let isLoading: Bool
  let isError: Bool
  var isStreaming: Bool = false

  var body: some View {
    if !message.isEmpty {
      HStack(spacing: 8) {
        if isLoading {
          ProgressView()
            .controlSize(.small)
        } else if isError {
          Image(systemName: "exclamationmark.triangle.fill")
            .foregroundStyle(.red)
        } else if isStreaming {
          // Streaming indicator with pulsing animation
          Circle()
            .fill(Color.purple)
            .frame(width: 8, height: 8)
            .overlay(
              Circle()
                .stroke(Color.purple.opacity(0.5), lineWidth: 2)
                .scaleEffect(1.5)
            )
        } else {
          Image(systemName: "checkmark.circle.fill")
            .foregroundStyle(.green)
        }

        Text(message)
          .font(.subheadline)
          .lineLimit(2)

        Spacer()
      }
      .padding(12)
      .glassEffect(.regular.tint(tintColor), in: .rect(cornerRadius: 8))
    }
  }

  private var tintColor: Color {
    if isError {
      Color.red.opacity(0.1)
    } else if isStreaming {
      Color.purple.opacity(0.1)
    } else if isLoading {
      Color.blue.opacity(0.1)
    } else {
      Color.green.opacity(0.1)
    }
  }
}
