// Copyright ® Canopy Labs (original model implementation)
// Ported to MLX from https://github.com/canopyai/Orpheus-TTS
// Copyright © 2024 Prince Canuma and contributors to Blaizzy/mlx-audio
// Copyright © Anthony DePasquale
// License: licenses/orpheus.txt

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Quantization configuration from Orpheus config.json
struct OrpheusQuantizationConfig: Codable, Sendable {
  var bits: Int
  var groupSize: Int

  enum CodingKeys: String, CodingKey {
    case bits
    case groupSize = "group_size"
  }
}

/// Partial config struct to extract just the quantization section
private struct OrpheusPartialConfig: Codable {
  var quantization: OrpheusQuantizationConfig?
}

class OrpheusWeightLoader {
  private init() {}

  static let defaultRepoId = "mlx-community/orpheus-3b-0.1-ft-4bit"
  static let defaultWeightsFilename = "model.safetensors"

  /// Load weights and quantization config from a local directory
  static func load(
    from directory: URL,
    filename: String = defaultWeightsFilename
  ) throws -> (weights: [String: MLXArray], quantization: OrpheusQuantizationConfig?) {
    // Load weights
    let weightFileURL = directory.appending(path: filename)
    let weights = try MLX.loadArrays(url: weightFileURL)

    // Load quantization config
    let configURL = directory.appending(path: "config.json")
    var quantization: OrpheusQuantizationConfig?
    if let data = try? Data(contentsOf: configURL),
       let config = try? JSONDecoder().decode(OrpheusPartialConfig.self, from: data)
    {
      quantization = config.quantization
    }

    return (weights, quantization)
  }

  /// Download and load weights and quantization config
  static func load(
    id: String = defaultRepoId,
    filename: String = defaultWeightsFilename,
    from downloader: any Downloader,
    progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
  ) async throws -> (weights: [String: MLXArray], quantization: OrpheusQuantizationConfig?) {
    let modelDirectoryURL = try await downloader.download(
      id: id,
      revision: nil,
      matching: [filename, "config.json"],
      useLatest: false,
      progressHandler: progressHandler
    )

    return try load(from: modelDirectoryURL, filename: filename)
  }
}
