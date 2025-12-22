// Copyright © Anthony DePasquale

import MLXAudio
import SwiftUI

struct SpeedSliderView: View {
  @Binding var speed: Float
  var isDisabled: Bool = false

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack {
        Label("Speed", systemImage: "gauge.with.needle")
          .font(.headline)
          .foregroundStyle(.secondary)

        Spacer()

        Text("\(speed.formatted(decimals: 1))x")
          .font(.caption)
          .foregroundStyle(.secondary)
          .monospacedDigit()
      }

      HStack(spacing: 12) {
        Image(systemName: "tortoise")
          .foregroundStyle(.secondary)
          .font(.caption)

        Slider(
          value: $speed,
          in: TTSConstants.Speed.minimum ... TTSConstants.Speed.maximum,
          step: TTSConstants.Speed.step,
        )
        .disabled(isDisabled)

        Image(systemName: "hare")
          .foregroundStyle(.secondary)
          .font(.caption)
      }

      // Quick presets
      HStack(spacing: 8) {
        ForEach([0.5, 1.0, 1.5, 2.0], id: \.self) { preset in
          SpeedPresetButton(
            preset: preset,
            isSelected: speed == Float(preset),
            isDisabled: isDisabled
          ) {
            withAnimation(.easeInOut(duration: 0.2)) {
              speed = Float(preset)
            }
          }
        }
      }
    }
  }
}

private struct SpeedPresetButton: View {
  let preset: Double
  let isSelected: Bool
  let isDisabled: Bool
  let action: () -> Void

  var body: some View {
    Button(action: action) {
      Text("\(preset.formatted(decimals: 1))x")
        .font(.caption)
    }
    .modifier(PresetButtonStyle(isSelected: isSelected))
    .disabled(isDisabled)
  }
}

private struct PresetButtonStyle: ViewModifier {
  let isSelected: Bool

  @ViewBuilder
  func body(content: Content) -> some View {
    if isSelected {
      content.buttonStyle(.glassProminent)
    } else {
      content.buttonStyle(.glass)
    }
  }
}
