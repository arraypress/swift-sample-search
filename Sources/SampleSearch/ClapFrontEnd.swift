//
//  ClapFrontEnd.swift
//  SampleSearch
//
//  Created by David Sherlock on 2026.
//
//  CLAP's log-mel front end as `ClapFeatureExtractor` computes it for the
//  music model: 48 kHz mono, a periodic Hann window of 1024, hop 480, centre
//  reflect padding, power spectrum, librosa's Slaney filterbank (64 bands,
//  50–14000 Hz, area-normalised) and 10·log10(max(1e-10, x)). A clip shorter
//  than ten seconds is repeated whole as many times as fits, then zero-padded
//  (`padding="repeatpad"`); ten seconds is 480,000 samples and 1,001 frames.
//

import Accelerate
import Foundation

/// The 1,001 × 64 log-mel a ten-second window becomes.
public struct ClapFrontEnd: Sendable {

    public static let sampleRate = 48_000
    public static let mels = 64
    public static let hop = 480
    public static let nFFT = 1024
    /// Ten seconds, the model's window.
    public static let windowSamples = 480_000
    public static let frames = 1 + windowSamples / hop   // 1001
    static let bins = nFFT / 2 + 1
    static let fMin = 50.0, fMax = 14_000.0
    static let floor: Float = 1e-10

    private let filterbank: [Float]   // [bins][mels]
    private let window: [Float]

    public init() {
        filterbank = Self.slaneyFilterbank()
        window = (0..<Self.nFFT).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(Self.nFFT))) }
    }

    /// Upstream's `repeatpad`: the clip tiled `⌊window / length⌋` times, then zeros to the window.
    public static func repeatPadded(_ samples: [Float]) -> [Float] {
        precondition(samples.count <= windowSamples && !samples.isEmpty)
        if samples.count == windowSamples { return samples }
        let repeats = windowSamples / samples.count
        var out = [Float]()
        out.reserveCapacity(windowSamples)
        for _ in 0..<repeats { out.append(contentsOf: samples) }
        out.append(contentsOf: [Float](repeating: 0, count: windowSamples - out.count))
        return out
    }

    /// `[frames][mels]` for exactly one window of samples.
    public func logMel(window samples: [Float]) -> [Float] {
        precondition(samples.count == Self.windowSamples)
        let half = Self.nFFT / 2
        var padded = [Float]()
        padded.reserveCapacity(samples.count + Self.nFFT)
        padded.append(contentsOf: (1...half).reversed().map { samples[$0] })
        padded.append(contentsOf: samples)
        padded.append(contentsOf: (2...(half + 1)).map { samples[samples.count - $0] })
        guard let setup = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(Self.nFFT), .FORWARD) else {
            preconditionFailure("could not create a \(Self.nFFT)-point DFT")
        }
        defer { vDSP_DFT_DestroySetup(setup) }
        var inRe = [Float](repeating: 0, count: Self.nFFT)
        let inIm = [Float](repeating: 0, count: Self.nFFT)
        var outRe = [Float](repeating: 0, count: Self.nFFT)
        var outIm = [Float](repeating: 0, count: Self.nFFT)
        var power = [Float](repeating: 0, count: Self.frames * Self.bins)
        for f in 0..<Self.frames {
            let start = f * Self.hop
            vDSP_vmul(Array(padded[start..<(start + Self.nFFT)]), 1, window, 1, &inRe, 1, vDSP_Length(Self.nFFT))
            vDSP_DFT_Execute(setup, inRe, inIm, &outRe, &outIm)
            for b in 0..<Self.bins { power[f * Self.bins + b] = outRe[b] * outRe[b] + outIm[b] * outIm[b] }
        }
        var mel = [Float](repeating: 0, count: Self.frames * Self.mels)
        vDSP_mmul(power, 1, filterbank, 1, &mel, 1, vDSP_Length(Self.frames), vDSP_Length(Self.mels), vDSP_Length(Self.bins))
        for i in mel.indices { mel[i] = 10 * log10(max(Self.floor, mel[i])) }
        return mel
    }

    /// `librosa.filters.mel(sr=48000, n_fft=1024, n_mels=64, fmin=50, fmax=14000)`: Slaney
    /// scale, `norm="slaney"`, as `[bins][mels]`.
    static func slaneyFilterbank() -> [Float] {
        let nFreqs = bins, nMels = mels
        let allFreqs = (0..<nFreqs).map { Double($0) * Double(sampleRate) / 2 / Double(nFreqs - 1) }
        let fSp = 200.0 / 3, minLogHz = 1000.0, minLogMel = minLogHz / fSp, logstep = log(6.4) / 27
        func hzToMel(_ f: Double) -> Double { f >= minLogHz ? minLogMel + log(f / minLogHz) / logstep : f / fSp }
        func melToHz(_ m: Double) -> Double { m >= minLogMel ? minLogHz * exp(logstep * (m - minLogMel)) : m * fSp }
        let mMin = hzToMel(fMin), mMax = hzToMel(fMax)
        let fPts = (0..<(nMels + 2)).map { melToHz(mMin + (mMax - mMin) * Double($0) / Double(nMels + 1)) }
        let fDiff = (0..<(nMels + 1)).map { fPts[$0 + 1] - fPts[$0] }
        var fb = [Float](repeating: 0, count: nFreqs * nMels)
        for m in 0..<nMels {
            let enorm = 2 / (fPts[m + 2] - fPts[m])
            for i in 0..<nFreqs {
                let lower = (allFreqs[i] - fPts[m]) / fDiff[m]
                let upper = (fPts[m + 2] - allFreqs[i]) / fDiff[m + 1]
                fb[i * nMels + m] = Float(max(0, min(lower, upper)) * enorm)
            }
        }
        return fb
    }
}
