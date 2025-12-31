// Copyright © Anthony DePasquale

import MLXAudio
import SwiftUI

/// Display timestamped segments with optional word-level timestamps
struct SegmentListView: View {
  let segments: [TranscriptionSegment]

  @State private var expandedSegments: Set<Int> = []

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack {
        Text("Segments")
          .font(.headline)
        Spacer()
        Text("\(segments.count) segment\(segments.count == 1 ? "" : "s")")
          .font(.caption)
          .foregroundStyle(.secondary)
      }

      if segments.isEmpty {
        Text("No segments available")
          .foregroundStyle(.secondary)
          .frame(maxWidth: .infinity, alignment: .center)
          .padding()
      } else {
        ScrollView {
          LazyVStack(spacing: 8) {
            ForEach(Array(segments.enumerated()), id: \.offset) { index, segment in
              SegmentRow(
                segment: segment,
                index: index,
                isExpanded: expandedSegments.contains(index),
                onToggle: {
                  if expandedSegments.contains(index) {
                    expandedSegments.remove(index)
                  } else {
                    expandedSegments.insert(index)
                  }
                }
              )
            }
          }
          .padding(.vertical, 4)
        }
        .frame(maxHeight: 300)
      }
    }
  }
}

/// Single segment row
struct SegmentRow: View {
  let segment: TranscriptionSegment
  let index: Int
  let isExpanded: Bool
  let onToggle: () -> Void

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      // Main segment content
      HStack(alignment: .top, spacing: 8) {
        // Timestamp badge
        Text(formatTimestamp(segment.start))
          .font(.caption.monospaced())
          .foregroundStyle(.secondary)
          .frame(width: 60, alignment: .leading)

        // Segment text
        Text(segment.text.trimmingCharacters(in: .whitespaces))
          .frame(maxWidth: .infinity, alignment: .leading)

        // Expand button if word timestamps available
        if let words = segment.words, !words.isEmpty {
          Button(action: onToggle) {
            Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
              .foregroundStyle(.secondary)
          }
          .buttonStyle(.plain)
        }
      }

      // Word-level timestamps (expanded)
      if isExpanded, let words = segment.words, !words.isEmpty {
        WordTimestampView(words: words)
          .padding(.leading, 68)
      }
    }
    .padding(12)
    .background(
      RoundedRectangle(cornerRadius: 8)
        .fill(Color.secondary.opacity(0.05))
    )
  }

  private func formatTimestamp(_ time: TimeInterval) -> String {
    let minutes = Int(time) / 60
    let seconds = Int(time) % 60
    let millis = Int((time.truncatingRemainder(dividingBy: 1)) * 100)
    return String(format: "%d:%02d.%02d", minutes, seconds, millis)
  }
}

/// Word-level timestamp display
struct WordTimestampView: View {
  let words: [Word]

  var body: some View {
    FlowLayout(spacing: 4) {
      ForEach(Array(words.enumerated()), id: \.offset) { _, word in
        VStack(spacing: 2) {
          Text(word.word)
            .font(.caption)
          Text(formatTime(word.start))
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 4)
        .background(
          RoundedRectangle(cornerRadius: 4)
            .fill(Color.blue.opacity(0.1))
        )
      }
    }
  }

  private func formatTime(_ time: TimeInterval) -> String {
    String(format: "%.2fs", time)
  }
}

/// Simple flow layout for word chips
struct FlowLayout: Layout {
  var spacing: CGFloat = 4

  func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache _: inout ()) -> CGSize {
    let result = FlowResult(in: proposal.width ?? 0, spacing: spacing, subviews: subviews)
    return result.size
  }

  func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache _: inout ()) {
    let result = FlowResult(in: bounds.width, spacing: spacing, subviews: subviews)
    for (index, position) in result.positions.enumerated() {
      subviews[index].place(at: CGPoint(x: bounds.minX + position.x, y: bounds.minY + position.y), proposal: .unspecified)
    }
  }

  struct FlowResult {
    var size: CGSize = .zero
    var positions: [CGPoint] = []

    init(in maxWidth: CGFloat, spacing: CGFloat, subviews: Subviews) {
      var currentX: CGFloat = 0
      var currentY: CGFloat = 0
      var lineHeight: CGFloat = 0

      for subview in subviews {
        let size = subview.sizeThatFits(.unspecified)

        if currentX + size.width > maxWidth, currentX > 0 {
          currentX = 0
          currentY += lineHeight + spacing
          lineHeight = 0
        }

        positions.append(CGPoint(x: currentX, y: currentY))
        lineHeight = max(lineHeight, size.height)
        currentX += size.width + spacing
        self.size.width = max(self.size.width, currentX - spacing)
      }

      size.height = currentY + lineHeight
    }
  }
}
