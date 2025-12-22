// Copyright © 2025 Resemble AI (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/resemble-ai/chatterbox
// License: licenses/chatterbox.txt

// HiFi-GAN vocoder for Chatterbox Turbo S3Gen
// Uses HiFTGenerator from shared Codec with Turbo-specific parameters

import Foundation
import MLX
import MLXNN

// MARK: - F0 Predictor for Turbo

/// F0 predictor from mel-spectrogram (Turbo variant with ELU activation)
class F0PredictorTurbo: Module {
  @ModuleInfo(key: "condnet") var condnet: [Conv1d]
  @ModuleInfo(key: "classifier") var classifier: Linear

  init(inChannels: Int = 80, hiddenChannels: Int = 512, numLayers: Int = 5) {
    var layers: [Conv1d] = []
    for i in 0 ..< numLayers {
      let inCh = i == 0 ? inChannels : hiddenChannels
      layers.append(Conv1d(
        inputChannels: inCh,
        outputChannels: hiddenChannels,
        kernelSize: 3,
        padding: 1
      ))
    }
    _condnet.wrappedValue = layers
    _classifier.wrappedValue = Linear(hiddenChannels, 1)
  }

  func callAsFunction(_ mel: MLXArray) -> MLXArray {
    // Input: (B, 80, T) in PyTorch format
    // Conv1d needs (B, T, C)
    var x = mel.transposed(0, 2, 1) // (B, T, C)

    for conv in condnet {
      x = conv(x)
      x = elu(x)
    }

    // classifier expects (B, T, C)
    var f0 = classifier(x) // (B, T, 1)
    f0 = f0.squeezed(axis: -1) // (B, T)

    // Apply abs() to ensure non-negative
    return MLX.abs(f0)
  }
}

// MARK: - ELU Activation

/// ELU activation: x if x > 0, alpha * (exp(x) - 1) otherwise
func elu(_ x: MLXArray, alpha: Float = 1.0) -> MLXArray {
  MLX.where(x .> 0, x, alpha * (MLX.exp(x) - 1))
}

// MARK: - HiFTGenerator for Turbo

/// HiFi-GAN vocoder configured for Chatterbox Turbo
/// Sample rate: 24000 Hz, upsample_rates: [8, 5, 3]
class HiFTGeneratorTurbo: Module {
  let samplingRate: Int
  let istftParams: [String: Int]
  let audioLimit: Float
  let numKernels: Int
  let numUpsamples: Int
  let f0UpsampleScale: Int

  @ModuleInfo(key: "m_source") var mSource: SourceModuleHnNSF
  @ModuleInfo(key: "conv_pre") var convPre: Conv1d
  @ModuleInfo(key: "ups") var ups: [ConvTransposed1d]
  @ModuleInfo(key: "source_downs") var sourceDowns: [Conv1d]
  @ModuleInfo(key: "source_resblocks") var sourceResblocks: [HiFiGANResBlock]
  @ModuleInfo(key: "resblocks") var resblocks: [HiFiGANResBlock]
  @ModuleInfo(key: "conv_post") var convPost: Conv1d
  @ModuleInfo(key: "f0_predictor") var f0Predictor: F0PredictorTurbo

  var stftWindow: MLXArray

  init(
    inChannels: Int = 80,
    baseChannels: Int = 512,
    nbHarmonics: Int = 8,
    samplingRate: Int = 24000,
    nsfAlpha: Float = 0.1,
    nsfSigma: Float = 0.003,
    nsfVoicedThreshold: Float = 10,
    upsampleRates: [Int] = [8, 5, 3],
    upsampleKernelSizes: [Int] = [16, 11, 7],
    resblockKernelSizes: [Int] = [3, 7, 11],
    resblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    sourceResblockKernelSizes: [Int] = [7, 7, 11],
    sourceResblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    istftParams: [String: Int] = ["n_fft": 16, "hop_len": 4]
  ) {
    self.samplingRate = samplingRate
    self.istftParams = istftParams
    audioLimit = 0.99
    numKernels = resblockKernelSizes.count
    numUpsamples = upsampleRates.count

    let upsampleScale = upsampleRates.reduce(1, *) * istftParams["hop_len"]!
    f0UpsampleScale = upsampleScale

    // F0 predictor
    _f0Predictor.wrappedValue = F0PredictorTurbo()

    // Neural Source Filter
    _mSource.wrappedValue = SourceModuleHnNSF(
      samplingRate: samplingRate,
      upsampleScale: upsampleScale,
      harmonicNum: nbHarmonics,
      sineAmp: nsfAlpha,
      addNoiseStd: nsfSigma,
      voicedThreshold: nsfVoicedThreshold
    )

    // Pre-convolution
    _convPre.wrappedValue = Conv1d(
      inputChannels: inChannels,
      outputChannels: baseChannels,
      kernelSize: 7,
      padding: 3
    )

    // Upsampling layers
    var upsArray: [ConvTransposed1d] = []
    for (i, (u, k)) in zip(upsampleRates, upsampleKernelSizes).enumerated() {
      upsArray.append(ConvTransposed1d(
        inputChannels: baseChannels >> i,
        outputChannels: baseChannels >> (i + 1),
        kernelSize: k,
        stride: u,
        padding: (k - u) / 2
      ))
    }
    _ups.wrappedValue = upsArray

    // Source downsampling and resblocks
    var sourceDownsArray: [Conv1d] = []
    var sourceResArray: [HiFiGANResBlock] = []
    let downsampleRates = [1] + Array(upsampleRates.reversed().dropLast())
    var downsampleCum: [Int] = []
    var cumProd = 1
    for rate in downsampleRates {
      cumProd *= rate
      downsampleCum.append(cumProd)
    }

    let nFft = istftParams["n_fft"]!
    for (i, (u, (k, d))) in zip(
      downsampleCum.reversed(),
      zip(sourceResblockKernelSizes, sourceResblockDilationSizes)
    ).enumerated() {
      let outCh = baseChannels >> (i + 1)
      if u == 1 {
        sourceDownsArray.append(Conv1d(
          inputChannels: nFft + 2,
          outputChannels: outCh,
          kernelSize: 1
        ))
      } else {
        sourceDownsArray.append(Conv1d(
          inputChannels: nFft + 2,
          outputChannels: outCh,
          kernelSize: u * 2,
          stride: u,
          padding: u / 2
        ))
      }
      sourceResArray.append(HiFiGANResBlock(
        channels: outCh,
        kernelSize: k,
        dilations: d
      ))
    }
    _sourceDowns.wrappedValue = sourceDownsArray
    _sourceResblocks.wrappedValue = sourceResArray

    // Main resblocks
    var resArray: [HiFiGANResBlock] = []
    for i in 0 ..< upsampleRates.count {
      let resCh = baseChannels >> (i + 1)
      for (k, d) in zip(resblockKernelSizes, resblockDilationSizes) {
        resArray.append(HiFiGANResBlock(channels: resCh, kernelSize: k, dilations: d))
      }
    }
    _resblocks.wrappedValue = resArray

    // Final conv
    let finalCh = baseChannels >> upsampleRates.count
    _convPost.wrappedValue = Conv1d(
      inputChannels: finalCh,
      outputChannels: nFft + 2,
      kernelSize: 7,
      padding: 3
    )

    // STFT window
    stftWindow = hannWindowPeriodic(size: nFft)
  }

  /// Upsample F0 using repeat
  private func f0Upsample(_ f0: MLXArray) -> MLXArray {
    MLX.repeated(f0, count: f0UpsampleScale, axis: 2)
  }

  /// Decode mel-spectrogram to waveform
  func decode(x: MLXArray, s: MLXArray) -> MLXArray {
    // STFT of source signal
    let (sStftReal, sStftImag) = stftHiFiGAN(
      x: s.squeezed(axis: 1),
      nFft: istftParams["n_fft"]!,
      hopLength: istftParams["hop_len"]!,
      window: stftWindow
    )
    let sStft = MLX.concatenated([sStftReal, sStftImag], axis: 1)

    // Pre-convolution
    var h = x.swappedAxes(1, 2)
    h = convPre(h)
    h = h.swappedAxes(1, 2)

    for i in 0 ..< numUpsamples {
      h = leakyRelu(h, negativeSlope: 0.1)
      // ConvTranspose1d
      h = h.swappedAxes(1, 2)
      h = ups[i](h)
      h = h.swappedAxes(1, 2)

      if i == numUpsamples - 1 {
        // Reflection pad: pad 1 sample on left
        h = MLX.concatenated([h[0..., 0..., 1 ..< 2], h], axis: 2)
      }

      // Source fusion
      var si = sStft.swappedAxes(1, 2)
      si = sourceDowns[i](si)
      si = si.swappedAxes(1, 2)
      si = sourceResblocks[i](si)

      // Match lengths
      let minLen = min(h.shape[2], si.shape[2])
      h = h[0..., 0..., 0 ..< minLen] + si[0..., 0..., 0 ..< minLen]

      // Apply residual blocks
      let baseIdx = i * numKernels
      let resOutputs = (0 ..< numKernels).map { resblocks[baseIdx + $0](h) }
      h = resOutputs.dropFirst().reduce(resOutputs[0], +) / Float(numKernels)
    }

    h = leakyRelu(h, negativeSlope: 0.1)
    h = h.swappedAxes(1, 2)
    h = convPost(h)
    h = h.swappedAxes(1, 2)

    // Split into magnitude and phase
    let nFftHalf = istftParams["n_fft"]! / 2 + 1
    let magnitude = MLX.exp(h[0..., 0 ..< nFftHalf, 0...])
    let phase = MLX.sin(h[0..., nFftHalf..., 0...])

    // Inverse STFT
    var output = istftHiFiGAN(
      magnitude: magnitude,
      phase: phase,
      nFft: istftParams["n_fft"]!,
      hopLength: istftParams["hop_len"]!,
      window: stftWindow
    )
    output = MLX.clip(output, min: -audioLimit, max: audioLimit)

    return output
  }

  /// Generate waveform from mel-spectrogram
  func callAsFunction(_ mel: MLXArray) -> (MLXArray, MLXArray) {
    // mel: (B, 80, T)

    // Predict F0
    let f0 = f0Predictor(mel)

    // Upsample F0
    var s = f0Upsample(f0.expandedDimensions(axis: 1))
    s = s.swappedAxes(1, 2) // (B, T, 1)

    // Generate source from F0
    let (sineMerge, _, _) = mSource(s)
    s = sineMerge.swappedAxes(1, 2) // (B, 1, T)

    // Decode
    let audio = decode(x: mel, s: s)

    return (audio, f0)
  }

  /// Inference function matching original API
  func inference(_ speechFeat: MLXArray, cacheSource: MLXArray? = nil) -> (MLXArray, MLXArray) {
    // speechFeat: (B, T, 80) - transposed from forward
    let mel = speechFeat.transposed(0, 2, 1)

    // Predict F0
    let f0 = f0Predictor(mel)

    // Upsample F0
    var s = f0Upsample(f0.expandedDimensions(axis: 1))
    s = s.swappedAxes(1, 2)

    // Generate source from F0
    let (sineMerge, _, _) = mSource(s)
    s = sineMerge.swappedAxes(1, 2)

    // Use cache if provided
    if let cache = cacheSource, cache.shape[2] > 0 {
      let cacheLen = cache.shape[2]
      s = MLX.concatenated([cache, s[0..., 0..., cacheLen...]], axis: 2)
    }

    // Decode
    let audio = decode(x: mel, s: s)

    return (audio, s)
  }
}
