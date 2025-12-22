// Copyright © 2025 Resemble AI (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/resemble-ai/chatterbox
// License: licenses/chatterbox.txt

// Causal Conditional Flow Matching for Chatterbox Turbo S3Gen

import Foundation
import MLX
import MLXNN
import MLXRandom

// MARK: - Conditional CFM

/// Conditional Flow Matching for speech generation
/// Uses Euler solver for ODE integration
class CBTConditionalCFM: Module {
  let inChannels: Int
  let sigmaMin: Float
  let tScheduler: String
  let inferenceCfgRate: Float
  var estimator: CBTConditionalDecoder?

  init(
    inChannels: Int = 240,
    nSpks _: Int = 1,
    spkEmbDim _: Int = 80,
    sigmaMin: Float = 1e-6,
    tScheduler: String = "cosine",
    inferenceCfgRate: Float = 0.7,
    estimator: CBTConditionalDecoder? = nil
  ) {
    self.inChannels = inChannels
    self.sigmaMin = sigmaMin
    self.tScheduler = tScheduler
    self.inferenceCfgRate = inferenceCfgRate
    self.estimator = estimator

    super.init()
  }

  /// Forward diffusion/flow matching
  func callAsFunction(
    mu: MLXArray,
    mask: MLXArray,
    nTimesteps: Int,
    temperature: Float = 1.0,
    spks: MLXArray? = nil,
    cond: MLXArray? = nil,
    noisedMels: MLXArray? = nil,
    meanflow: Bool = false
  ) -> MLXArray {
    // Initialize with random noise
    var z = MLXRandom.normal(mu.shape) * temperature

    if let noisedMels {
      let promptLen = mu.shape[2] - noisedMels.shape[2]
      z = MLX.concatenated([z[0..., 0..., 0 ..< promptLen], noisedMels], axis: 2)
    }

    // Time steps for reverse diffusion
    var tSpan = MLX.linspace(Float32(0), Float32(1), count: nTimesteps + 1)
    if !meanflow, tScheduler == "cosine" {
      tSpan = 1 - MLX.cos(tSpan * 0.5 * Float.pi)
    }

    if meanflow {
      return basicEuler(x: z, tSpan: tSpan, mu: mu, mask: mask, spks: spks, cond: cond)
    }

    return solveEulerCfg(x: z, tSpan: tSpan, mu: mu, mask: mask, spks: spks, cond: cond, meanflow: meanflow)
  }

  /// Basic Euler solver without CFG (for meanflow distilled models)
  func basicEuler(
    x: MLXArray,
    tSpan: MLXArray,
    mu: MLXArray,
    mask: MLXArray,
    spks: MLXArray?,
    cond: MLXArray?
  ) -> MLXArray {
    guard let estimator else {
      fatalError("Estimator not set")
    }

    var xCurrent = x
    let numSteps = tSpan.shape[0] - 1

    for i in 0 ..< numSteps {
      let t = tSpan[i ..< (i + 1)]
      let r = tSpan[(i + 1) ..< (i + 2)]

      // Predict velocity
      let dxdt = estimator(
        xCurrent,
        mask: mask,
        mu: mu,
        t: t,
        spks: spks,
        cond: cond,
        r: r
      )

      // Euler step
      let dt = r - t
      xCurrent = xCurrent + dt * dxdt

      // Force evaluation to prevent computation graph accumulation
      eval(xCurrent)
    }

    return xCurrent
  }

  /// Euler solver with classifier-free guidance
  func solveEulerCfg(
    x: MLXArray,
    tSpan: MLXArray,
    mu: MLXArray,
    mask: MLXArray,
    spks: MLXArray?,
    cond: MLXArray?,
    meanflow: Bool
  ) -> MLXArray {
    guard let estimator else {
      fatalError("Estimator not set")
    }

    let B = mu.shape[0]
    var xCurrent = x

    // Pre-compute static zero arrays for unconditional generation
    let muZeros = MLXArray.zeros(like: mu)
    let spksZeros = spks != nil ? MLXArray.zeros(like: spks!) : nil
    let condZeros = cond != nil ? MLXArray.zeros(like: cond!) : nil

    // Pre-compute mask_in since mask doesn't change
    let maskIn = MLX.concatenated([mask, mask], axis: 0)

    let numSteps = tSpan.shape[0] - 1

    for i in 0 ..< numSteps {
      let t = tSpan[i ..< (i + 1)]
      let r = tSpan[(i + 1) ..< (i + 2)]

      // Duplicate for CFG: [cond, uncond]
      let xIn = MLX.concatenated([xCurrent, xCurrent], axis: 0)
      let muIn = MLX.concatenated([mu, muZeros], axis: 0)
      let tIn = MLX.broadcast(t, to: [2 * B])
      let rIn = meanflow ? MLX.broadcast(r, to: [2 * B]) : nil

      let spksIn: MLXArray? = if let spks, let spksZeros {
        MLX.concatenated([spks, spksZeros], axis: 0)
      } else {
        nil
      }

      let condIn: MLXArray? = if let cond, let condZeros {
        MLX.concatenated([cond, condZeros], axis: 0)
      } else {
        nil
      }

      // Predict velocity
      let dxdt = estimator(
        xIn,
        mask: maskIn,
        mu: muIn,
        t: tIn,
        spks: spksIn,
        cond: condIn,
        r: rIn
      )

      // CFG combination
      let dxdtSplit = dxdt.split(parts: 2, axis: 0)
      let dxdtCond = dxdtSplit[0]
      let dxdtUncond = dxdtSplit[1]
      let dxdtCombined = (1.0 + inferenceCfgRate) * dxdtCond - inferenceCfgRate * dxdtUncond

      // Euler step
      let dt = r - t
      xCurrent = xCurrent + dt * dxdtCombined

      // Force evaluation to prevent computation graph accumulation
      eval(xCurrent)
    }

    return xCurrent
  }
}

// MARK: - Causal Conditional CFM

/// Causal version of Conditional CFM for streaming
class CBTCausalConditionalCFM: CBTConditionalCFM {
  override init(
    inChannels: Int = 240,
    nSpks: Int = 1,
    spkEmbDim: Int = 80,
    sigmaMin: Float = 1e-6,
    tScheduler: String = "cosine",
    inferenceCfgRate: Float = 0.7,
    estimator: CBTConditionalDecoder? = nil
  ) {
    super.init(
      inChannels: inChannels,
      nSpks: nSpks,
      spkEmbDim: spkEmbDim,
      sigmaMin: sigmaMin,
      tScheduler: tScheduler,
      inferenceCfgRate: inferenceCfgRate,
      estimator: estimator
    )
  }

  /// Forward with meanflow mode for distilled models
  override func callAsFunction(
    mu: MLXArray,
    mask: MLXArray,
    nTimesteps: Int,
    temperature _: Float = 1.0,
    spks: MLXArray? = nil,
    cond: MLXArray? = nil,
    noisedMels: MLXArray? = nil,
    meanflow: Bool = false
  ) -> MLXArray {
    // Initialize with random noise
    var z = MLXRandom.normal(mu.shape)

    if let noisedMels {
      let promptLen = mu.shape[2] - noisedMels.shape[2]
      z = MLX.concatenated([z[0..., 0..., 0 ..< promptLen], noisedMels], axis: 2)
    }

    // Time steps
    var tSpan = MLX.linspace(Float32(0), Float32(1), count: nTimesteps + 1)
    if !meanflow, tScheduler == "cosine" {
      tSpan = 1 - MLX.cos(tSpan * 0.5 * Float.pi)
    }

    // Meanflow distilled models don't need CFG
    if meanflow {
      return basicEuler(x: z, tSpan: tSpan, mu: mu, mask: mask, spks: spks, cond: cond)
    }

    return solveEulerCfg(x: z, tSpan: tSpan, mu: mu, mask: mask, spks: spks, cond: cond, meanflow: meanflow)
  }
}
