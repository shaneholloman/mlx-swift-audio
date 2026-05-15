// Copyright © 2025 FunASR (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/modelscope/FunASR
// License: licenses/funasr.txt

import Foundation
import MLX

// MARK: - Fun-ASR Audio Constants

/// Fun-ASR audio hyperparameters
public enum FunASRAudio {
  public static let sampleRate = 16000
  public static let nFft = 400 // 25ms window at 16kHz
  public static let hopLength = 160 // 10ms hop
  public static let nMels = 80

  // LFR (Low Frame Rate) parameters
  public static let lfrM = 7 // Stack every 7 frames
  public static let lfrN = 6 // Subsample by factor of 6

  // Derived values
  public static let inputDim = nMels * lfrM // 560
  public static let framesPerSecond = sampleRate / hopLength // 100
}

// MARK: - Window Functions

/// Create a Hamming window (Fun-ASR uses Hamming, not Hann)
///
/// Formula: w[n] = 0.54 - 0.46 * cos(2 * pi * n / (N - 1))
///
/// - Parameter length: Window length
/// - Returns: Hamming window array
func hammingWindow(length: Int) -> MLXArray {
  if length == 1 {
    return MLXArray([1.0])
  }

  // Use MLX array creation directly (avoid Swift array intermediate)
  let n = MLXArray(0 ..< length).asType(.float32)
  let factor = 2.0 * Float.pi / Float(length - 1)

  return 0.54 - 0.46 * MLX.cos(n * factor)
}

// MARK: - Mel Spectrogram

/// Compute kaldi-compatible log-mel filterbank features for Fun-ASR.
///
/// Approximates `torchaudio.compliance.kaldi.fbank` with the WavFrontend
/// defaults from upstream Fun-ASR (`upsacle_samples=True`, `dither=1.0`,
/// `pre-emphasis=0.97`, `window=hamming`, `snip_edges=True`, HTK mel,
/// no Slaney normalization, `energy_floor=0`) plus the kaldi defaults
/// that WavFrontend doesn't override (`remove_dc_offset=True`,
/// `round_to_power_of_two=True`, `low_freq=20`):
///
/// 1. Upscale float audio by 32768 to int16 magnitude
/// 2. Frame the signal with `snip_edges=True` (no padding; the trailing
///    samples that don't fit a full window are discarded)
/// 3. Optional Gaussian dither at int16 scale
/// 4. Per-frame DC offset removal: subtract per-frame mean
/// 5. Per-frame pre-emphasis: `y[i] = x[i] - 0.97 * x[i-1]`,
///    `y[0] = (1 - 0.97) * x[0]`
/// 6. Apply Hamming window
/// 7. Zero-pad each windowed frame to the next power of two
/// 8. Real FFT, drop the Nyquist bin
/// 9. Power spectrum (`|X|^2`)
/// 10. HTK mel filterbank with `low_freq=20` and no Slaney area normalization;
///    frequency points are the actual FFT bin centers
/// 11. Natural log with a small floor to avoid `log(0)`
///
/// - Parameters:
///   - audio: Audio waveform (T,) at 16 kHz, float in [-1, 1]
///   - nMels: Number of mel filterbank bins (default: 80)
///   - nFft: Window length in samples (default: 400, i.e., 25 ms at 16 kHz).
///     The actual FFT length is rounded up to the next power of two.
///   - hopLength: Hop length (default: 160, i.e., 10 ms at 16 kHz)
/// - Returns: Log-mel features `(numFrames, nMels)`
func funASRLogMelSpectrogram(
  audio: MLXArray,
  nMels: Int = FunASRAudio.nMels,
  nFft: Int = FunASRAudio.nFft,
  hopLength: Int = FunASRAudio.hopLength,
  dither: Float = 0.0,
) -> MLXArray {
  // Upscale to int16 magnitude (matches `upsacle_samples=True`).
  let upscaled = audio * Float(1 << 15)

  // snip_edges=True framing: no padding, drop trailing partial frame.
  precondition(upscaled.shape[0] >= nFft, "Audio is too short for one frame (\(upscaled.shape[0]) < \(nFft))")
  let numFrames = (upscaled.shape[0] - nFft) / hopLength + 1
  var frames = MLX.asStrided(upscaled, [numFrames, nFft], strides: [hopLength, 1])

  // Optional Gaussian dither (kaldi default `dither=1.0`).
  if dither != 0 {
    frames = frames + MLXRandom.normal([numFrames, nFft]) * dither
  }

  // Per-frame DC offset removal (kaldi `remove_dc_offset=True`).
  let frameMean = frames.mean(axis: -1, keepDims: true)
  let dcRemoved = frames - frameMean

  // Per-frame pre-emphasis: y[i] = x[i] - 0.97 * x[i-1], y[0] = 0.03 * x[0].
  let firstColumn = dcRemoved[0..., 0 ..< 1]
  let allButLast = dcRemoved[0..., 0 ..< (nFft - 1)]
  let shifted = MLX.concatenated([firstColumn, allButLast], axis: -1)
  let preemphasized = dcRemoved - 0.97 * shifted

  // Hamming window.
  let window = hammingWindow(length: nFft)
  let windowed = preemphasized * window

  // Zero-pad to next power of two for the FFT (kaldi `round_to_power_of_two`).
  let fftLength = nextPowerOfTwo(nFft)
  let padLength = fftLength - nFft
  let padded: MLXArray = if padLength > 0 {
    MLX.concatenated([windowed, MLXArray.zeros([numFrames, padLength])], axis: -1)
  } else {
    windowed
  }
  let spec = MLX.rfft(padded)

  // Power spectrum, drop the Nyquist bin to keep nFreqs = fftLength/2.
  let nFreqs = fftLength / 2
  let freqs = spec[0..., 0 ..< nFreqs]
  let magnitudes = MLX.pow(MLX.abs(freqs), 2)

  // Kaldi-compatible mel filterbank: HTK scale triangles in mel space,
  // no Slaney normalization, low_freq=20 Hz, FFT-bin frequencies.
  let filters = funASRMelFilters(
    sampleRate: FunASRAudio.sampleRate,
    fftLength: fftLength,
    nMels: nMels,
    fMin: 20.0,
  )

  // (T, F) @ (F, M) -> (T, M)
  let melSpec = MLX.matmul(magnitudes, filters.transposed())

  // Natural log with kaldi's FLT_EPSILON floor (≈ 1.19e-7).
  return MLX.log(MLX.maximum(melSpec, MLXArray(Float.ulpOfOne)))
}

/// Smallest power of two that is >= `n`. Returns 1 for `n <= 0`.
private func nextPowerOfTwo(_ n: Int) -> Int {
  guard n > 1 else { return max(1, n) }
  var p = 1
  while p < n {
    p <<= 1
  }
  return p
}

// MARK: - LFR Processing

/// Apply Low Frame Rate (LFR) processing to features
///
/// This stacks consecutive frames and subsamples to reduce the frame rate.
/// Uses vectorized gather operations for efficiency.
///
/// - Parameters:
///   - features: Input mel spectrogram (n_frames, n_mels)
///   - lfrM: Number of frames to stack (default: 7)
///   - lfrN: Subsampling factor (default: 6)
/// - Returns: LFR-processed features (ceil(n_frames / lfrN), n_mels * lfrM)
func applyLFR(
  _ features: MLXArray,
  lfrM: Int = FunASRAudio.lfrM,
  lfrN: Int = FunASRAudio.lfrN,
) -> MLXArray {
  let T = features.shape[0]
  let nMels = features.shape[1]

  // Output length uses ceiling division
  let tLFR = Int(ceil(Double(T) / Double(lfrN)))

  // Left padding with first frame repeated
  let leftPad = (lfrM - 1) / 2
  var paddedFeatures = features

  if leftPad > 0 {
    // Broadcast first frame to create left padding
    let firstFrame = features[0].expandedDimensions(axis: 0)
    let leftPadding = MLX.broadcast(firstFrame, to: [leftPad, nMels])
    paddedFeatures = MLX.concatenated([leftPadding, paddedFeatures], axis: 0)
  }

  // Right padding to ensure we have enough frames
  let tPadded = paddedFeatures.shape[0]
  let totalNeeded = (tLFR - 1) * lfrN + lfrM
  if totalNeeded > tPadded {
    let rightPad = totalNeeded - tPadded
    let lastFrame = paddedFeatures[tPadded - 1].expandedDimensions(axis: 0)
    let rightPadding = MLX.broadcast(lastFrame, to: [rightPad, nMels])
    paddedFeatures = MLX.concatenated([paddedFeatures, rightPadding], axis: 0)
  }

  // Create indices for all output frames using vectorized gather
  // startIndices: [0, lfrN, 2*lfrN, ..., (tLFR-1)*lfrN]
  let startIndices = MLXArray(0 ..< tLFR) * lfrN
  // offsets: [0, 1, 2, ..., lfrM-1]
  let offsets = MLXArray(0 ..< lfrM)

  // Broadcasting: (tLFR, 1) + (lfrM,) -> (tLFR, lfrM)
  let indices = startIndices.expandedDimensions(axis: 1) + offsets.expandedDimensions(axis: 0)

  // Gather frames: paddedFeatures[indices] -> (tLFR, lfrM, nMels)
  let gathered = paddedFeatures[indices]

  // Reshape to (tLFR, lfrM * nMels)
  return gathered.reshaped([tLFR, lfrM * nMels])
}

// MARK: - CMVN Normalization

/// Apply Cepstral Mean and Variance Normalization (CMVN)
///
/// - Parameters:
///   - features: Input features (T, D)
///   - cmvnMean: Precomputed mean shift (optional)
///   - cmvnIstd: Precomputed inverse std (optional)
/// - Returns: Normalized features
func applyCMVN(
  _ features: MLXArray,
  cmvnMean: MLXArray? = nil,
  cmvnIstd: MLXArray? = nil,
) -> MLXArray {
  if let mean = cmvnMean, let istd = cmvnIstd {
    // Apply precomputed CMVN: (x + mean) * istd
    // Note: cmvnMean is actually the negative mean (shift)
    return (features + mean) * istd
  }

  // Per-utterance normalization
  let mean = features.mean(axis: 0, keepDims: true)
  let std = MLX.variance(features, axis: 0, keepDims: true).sqrt() + 1e-6
  return (features - mean) / std
}

// MARK: - Full Preprocessing Pipeline

/// Full audio preprocessing pipeline for Fun-ASR
///
/// 1. Compute kaldi-compatible log-mel filterbank features
/// 2. Apply LFR (frame stacking and subsampling)
/// 3. Optionally apply CMVN normalization (off by default; the upstream
///    Fun-ASR-Nano-2512 config has `cmvn_file: null`)
///
/// - Parameters:
///   - audio: Input audio waveform (T,) at 16kHz
///   - nMels: Number of mel bins (default: 80)
///   - lfrM: LFR frame stacking count (default: 7)
///   - lfrN: LFR subsampling factor (default: 6)
///   - applyNormalization: Whether to apply CMVN (default: false)
/// - Returns: Preprocessed features (ceil(T / (hopLength * lfrN)), nMels * lfrM)
func preprocessAudio(
  _ audio: MLXArray,
  nMels: Int = FunASRAudio.nMels,
  lfrM: Int = FunASRAudio.lfrM,
  lfrN: Int = FunASRAudio.lfrN,
  applyNormalization: Bool = false,
  dither: Float = 0.0,
) -> MLXArray {
  // Compute log mel spectrogram
  var features = funASRLogMelSpectrogram(audio: audio, nMels: nMels, dither: dither)

  // Apply LFR processing
  features = applyLFR(features, lfrM: lfrM, lfrN: lfrN)

  // Apply normalization
  if applyNormalization {
    features = applyCMVN(features)
  }

  return features
}

/// Compute output feature lengths after preprocessing
///
/// - Parameters:
///   - audioLength: Length of input audio in samples
///   - hopLength: Hop length for STFT (default: 160)
///   - lfrN: LFR subsampling factor (default: 6)
/// - Returns: Output feature length
func computeFeatureLength(
  audioLength: Int,
  hopLength: Int = FunASRAudio.hopLength,
  lfrN: Int = FunASRAudio.lfrN,
) -> Int {
  // Frames after STFT (approximately)
  let nFrames = audioLength / hopLength

  // Frames after LFR (ceiling division)
  return (nFrames + lfrN - 1) / lfrN
}

// MARK: - Private Helper Functions

/// Build the kaldi-compliant HTK mel filterbank.
///
/// Triangles are constructed in **mel space** (not Hz space): filter centers
/// are evenly spaced in mel from `fMin` to Nyquist, and the rising/falling
/// edges are normalized by the mel widths of each triangle. FFT-bin
/// frequencies are converted to mel via the HTK formula
/// `2595 · log10(1 + hz/700)` and the up/down slopes are computed against the
/// mel center positions. This matches `torchaudio.compliance.kaldi.get_mel_banks`.
///
/// - Returns: filterbank of shape `(nMels, nFreqs)` where `nFreqs = fftLength / 2`.
private func funASRMelFilters(
  sampleRate: Int,
  fftLength: Int,
  nMels: Int,
  fMin: Float = 0.0,
  fMax: Float? = nil,
) -> MLXArray {
  let actualFMax = fMax ?? Float(sampleRate) / 2.0
  let nFreqs = fftLength / 2

  // FFT bin center frequencies in Hz: k * sample_rate / fft_length.
  let fftBinWidth = Float(sampleRate) / Float(fftLength)
  let allFreqsHz = MLXArray(0 ..< nFreqs).asType(.float32) * fftBinWidth

  // Convert FFT bin frequencies to mel via HTK formula.
  let allFreqsMel = 2595.0 * MLX.log10(1.0 + allFreqsHz / 700.0)

  // Mel center positions for the (nMels + 2) bin edges, evenly spaced in mel.
  let melLowFreq = 2595.0 * log10f(1.0 + fMin / 700.0)
  let melHighFreq = 2595.0 * log10f(1.0 + actualFMax / 700.0)
  let melFreqDelta = (melHighFreq - melLowFreq) / Float(nMels + 1)

  // Per-bin mel triangle vertices: left = lowFreq + i·delta, center = lowFreq + (i+1)·delta,
  // right = lowFreq + (i+2)·delta, for i in 0..<nMels.
  let bin = MLXArray(0 ..< nMels).asType(.float32)
  let leftMel = melLowFreq + bin * melFreqDelta // (nMels,)
  let centerMel = melLowFreq + (bin + 1.0) * melFreqDelta
  let rightMel = melLowFreq + (bin + 2.0) * melFreqDelta

  // Broadcast to (nFreqs, nMels): up/down slopes in mel space.
  let melMatrix = allFreqsMel.expandedDimensions(axis: 1) // (nFreqs, 1)
  let upSlope = (melMatrix - leftMel.expandedDimensions(axis: 0)) /
    (centerMel - leftMel).expandedDimensions(axis: 0)
  let downSlope = (rightMel.expandedDimensions(axis: 0) - melMatrix) /
    (rightMel - centerMel).expandedDimensions(axis: 0)

  // Triangle: max(0, min(up, down)). Shape: (nFreqs, nMels).
  let filterbank = MLX.maximum(
    MLXArray.zeros(like: upSlope),
    MLX.minimum(upSlope, downSlope),
  )

  // Transpose to (nMels, nFreqs) to match callsite.
  return filterbank.transposed()
}
