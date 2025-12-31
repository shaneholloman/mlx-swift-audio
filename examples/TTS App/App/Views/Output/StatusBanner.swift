// Copyright © Anthony DePasquale

import SwiftUI

struct StatusBanner: View {
  let message: String
  var progress: Double?
  var isError: Bool = false

  var body: some View {
    VStack(alignment: .leading, spacing: 4) {
      if let progress, progress > 0, progress < 1 {
        ProgressView(value: progress)
      }

      Text(message)
        .font(.callout)
        .foregroundStyle(isError ? .red : .secondary)
        .lineLimit(2)
    }
  }
}
