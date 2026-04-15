// swift-tools-version:6.2
import PackageDescription

let package = Package(
  name: "mlx-audio",
  platforms: [.macOS("15.4"), .iOS("18.4")],
  products: [
    // Core library without Kokoro (no GPLv3 dependencies)
    .library(
      name: "MLXAudio",
      targets: ["MLXAudio"],
    ),
    // Separate Kokoro package (depends on GPLv3-licensed espeak-ng)
    .library(
      name: "Kokoro",
      targets: ["Kokoro"],
    ),
  ],
  dependencies: [
    // TODO: Switch from this pinned revision to a major-version dependency once mlx-swift-lm publishes a release that includes PR #118.
    .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", revision: "8c9dd6391139242261bcf27d253c326f9cf2d567"),
    .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.3")),
    .package(url: "https://github.com/DePasqualeOrg/swift-tokenizers-mlx", branch: "main", traits: ["Swift"]),
    .package(url: "https://github.com/DePasqualeOrg/swift-hf-api-mlx", branch: "main"),
    .package(url: "https://github.com/DePasqualeOrg/swift-tiktoken", branch: "main"),
    // espeak-ng is GPLv3 licensed - only linked when using Kokoro
    // TODO: Switch back to upstream after https://github.com/espeak-ng/espeak-ng/pull/2327 is merged
    .package(url: "https://github.com/DePasqualeOrg/espeak-ng-spm.git", branch: "fix-path-espeak-data-macro"),
  ],
  targets: [
    .target(
      name: "MLXAudio",
      dependencies: [
        .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
        .product(name: "MLXLLM", package: "mlx-swift-lm"),
        .product(name: "MLXFFT", package: "mlx-swift"),
        .product(name: "MLXLMTokenizers", package: "swift-tokenizers-mlx"),
        .product(name: "MLXLMHFAPI", package: "swift-hf-api-mlx"),
        .product(name: "SwiftTiktoken", package: "swift-tiktoken"),
      ],
      path: "package",
      exclude: ["TTS/Kokoro", "Tests"],
      resources: [
        .process("TTS/OuteTTS/default_speaker.json"), // Default speaker profile for OuteTTS
      ],
    ),
    .target(
      name: "Kokoro",
      dependencies: [
        "MLXAudio",
        .product(name: "libespeak-ng", package: "espeak-ng-spm"),
        .product(name: "espeak-ng-data", package: "espeak-ng-spm"),
      ],
      path: "package/TTS/Kokoro",
    ),
    .testTarget(
      name: "MLXAudioTests",
      dependencies: ["MLXAudio"],
      path: "package/Tests",
    ),
  ],
)
