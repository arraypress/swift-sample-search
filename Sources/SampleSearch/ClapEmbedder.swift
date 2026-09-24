//
//  ClapEmbedder.swift
//  SampleSearch
//
//  Created by David Sherlock on 2026.
//
//  CLAP (LAION, Apache 2.0) on Core AI: a sound and a sentence each become a
//  512-number vector in one space, so a library can be searched by words or
//  by example and tagged from a list of phrases. The audio encoder (HTSAT)
//  and the text encoder (RoBERTa) sit in the .aimodel; the log-mel, the
//  tokenizer, the L2 normalisation and the window policy for long audio are
//  here. Upstream random-crops audio longer than ten seconds; this averages
//  the embeddings of consecutive ten-second windows, which is deterministic.
//

import AVFoundation
import CoreAI
import Foundation

/// Audio and text embeddings from `crate-clap-music-float32.aimodel`.
public final class ClapEmbedder: @unchecked Sendable {

    public static let assetName = "crate-clap-music-float32.aimodel"
    /// `vocab.json`, `merges.txt` and `scales.json` beside the asset.
    public static let supportFolder = "clap-support"
    /// The text encoder's fixed length.
    public static let tokens = 77
    public static let dimension = 512

    public let url: URL
    public let tokenizer: BytePairTokenizer
    /// exp(`logit_scale_a`) from the checkpoint: what upstream multiplies cosines by before the zero-shot softmax.
    public let logitScale: Float
    private let audio: InferenceFunction
    private let text: InferenceFunction
    private let frontEnd = ClapFrontEnd()

    public enum Error: Swift.Error, CustomStringConvertible {
        case modelNotFound(String), modelUnavailable(String), audioUnreadable(String), emptyAudio
        public var description: String {
            switch self {
            case .modelNotFound(let p): return "no model at \(p)"
            case .modelUnavailable(let m): return m
            case .audioUnreadable(let m): return m
            case .emptyAudio: return "the audio is empty"
            }
        }
    }

    /// `asset` is the `.aimodel`; `support` the folder with `vocab.json`, `merges.txt` and `scales.json`.
    public init(contentsOf asset: URL, support: URL) async throws {
        guard FileManager.default.fileExists(atPath: asset.path) else { throw Error.modelNotFound(asset.path) }
        self.url = asset
        do { self.tokenizer = try BytePairTokenizer(contentsOf: support) } catch {
            throw Error.modelUnavailable("\(support.lastPathComponent)/: \(error)")
        }
        self.logitScale = try Self.readLogitScale(support.appendingPathComponent("scales.json"))
        var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        options.expectFrequentReshapes = true
        let model: AIModel
        do { model = try await AIModel(contentsOf: asset, options: options) } catch {
            throw Error.modelUnavailable("could not load \(asset.lastPathComponent): \(error)")
        }
        guard let audio = try model.loadFunction(named: "audio"), let text = try model.loadFunction(named: "text") else {
            throw Error.modelUnavailable("\(asset.lastPathComponent) needs audio and text entry points; found \(model.functionNames)")
        }
        self.audio = audio
        self.text = text
    }

    /// The asset and support folder next to each other, as `--install` and the CLI lay them out.
    public convenience init(inFolder folder: URL) async throws {
        try await self.init(contentsOf: folder.appendingPathComponent(Self.assetName), support: folder.appendingPathComponent(Self.supportFolder))
    }

    static func readLogitScale(_ url: URL) throws -> Float {
        guard let data = try? Data(contentsOf: url) else { throw Error.modelNotFound(url.path) }
        guard let table = try? JSONSerialization.jsonObject(with: data) as? [String: Any], let scale = table["logit_scale_audio"] as? Double else {
            throw Error.modelUnavailable("\(url.lastPathComponent) has no logit_scale_audio")
        }
        return Float(scale)
    }

    // MARK: - Text

    /// The unit vector for a phrase.
    public func embed(text phrase: String) async throws -> [Float] {
        let (ids, mask) = tokenizer.encode(phrase, length: Self.tokens)
        var out = NDArray(shape: [1, Self.dimension], scalarType: .float32)
        var views = InferenceFunction.MutableViews()
        views.insert(out.mutableRawView(), for: "embedding")
        do {
            _ = try await text.run(inputs: ["ids": Self.ints(ids, shape: [1, Self.tokens]), "mask": Self.ints(mask, shape: [1, Self.tokens])],
                                   states: InferenceFunction.MutableViews(), outputViews: consume views)
        } catch { throw Error.modelUnavailable("text inference failed: \(error)") }
        return Self.normalised(Self.floats(out))
    }

    // MARK: - Audio

    /// The unit vector for one ten-second window of log-mel (`[frames][mels]`).
    public func embed(mel: [Float]) async throws -> [Float] {
        precondition(mel.count == ClapFrontEnd.frames * ClapFrontEnd.mels)
        var out = NDArray(shape: [1, Self.dimension], scalarType: .float32)
        var views = InferenceFunction.MutableViews()
        views.insert(out.mutableRawView(), for: "embedding")
        do {
            _ = try await audio.run(inputs: ["mel": Self.array(mel, shape: [1, 1, ClapFrontEnd.frames, ClapFrontEnd.mels])],
                                    states: InferenceFunction.MutableViews(), outputViews: consume views)
        } catch { throw Error.modelUnavailable("audio inference failed: \(error)") }
        return Self.normalised(Self.floats(out))
    }

    /// The unit vector for 48 kHz mono samples: one repeat-padded window up to ten
    /// seconds, else the mean of consecutive ten-second windows (a final window
    /// shorter than one second is dropped), re-normalised.
    public func embed(samples48k: [Float]) async throws -> [Float] {
        guard !samples48k.isEmpty else { throw Error.emptyAudio }
        let W = ClapFrontEnd.windowSamples
        if samples48k.count <= W {
            return try await embed(mel: frontEnd.logMel(window: ClapFrontEnd.repeatPadded(samples48k)))
        }
        var sum = [Float](repeating: 0, count: Self.dimension)
        var windows = 0
        var start = 0
        while start < samples48k.count {
            let end = min(start + W, samples48k.count)
            let slice = Array(samples48k[start..<end])
            if slice.count < ClapFrontEnd.sampleRate && windows > 0 { break }
            let e = try await embed(mel: frontEnd.logMel(window: ClapFrontEnd.repeatPadded(slice)))
            for i in 0..<Self.dimension { sum[i] += e[i] }
            windows += 1
            start += W
        }
        return Self.normalised(sum.map { $0 / Float(windows) })
    }

    /// Any audio file AVFoundation decodes: mixed to mono, resampled to 48 kHz.
    public func embed(contentsOf url: URL) async throws -> (embedding: [Float], seconds: Double) {
        let samples = try Self.load(url)
        return (try await embed(samples48k: samples), Double(samples.count) / Double(ClapFrontEnd.sampleRate))
    }

    /// Decode to mono (the mean of the channels, as upstream's `y.mean(1)`) and, when the file
    /// is not already 48 kHz, resample with AVAudioConverter at its best quality. A 48 kHz file is
    /// read sample for sample — no converter touches it, so fixtures match upstream exactly.
    public static func load(_ url: URL) throws -> [Float] {
        let file: AVAudioFile
        do { file = try AVAudioFile(forReading: url) } catch { throw Error.audioUnreadable("\(url.lastPathComponent): \(error.localizedDescription)") }
        let format = file.processingFormat
        let total = Int(file.length)
        let channels = Int(format.channelCount)
        guard total > 0, let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 1 << 16) else { throw Error.emptyAudio }
        // One read(into:) can return short of the file on some WAVs; read until the position reaches the length.
        var mono = [Float]()
        mono.reserveCapacity(total)
        let scale = 1 / Float(channels)
        while file.framePosition < file.length {
            try file.read(into: buffer, frameCount: buffer.frameCapacity)
            let frames = Int(buffer.frameLength)
            guard frames > 0, let data = buffer.floatChannelData else { break }
            for i in 0..<frames {
                var v = data[0][i]
                if channels > 1 { for c in 1..<channels { v += data[c][i] }; v *= scale }
                mono.append(v)
            }
        }
        guard !mono.isEmpty else { throw Error.emptyAudio }
        return format.sampleRate == Double(ClapFrontEnd.sampleRate) ? mono : resample(mono, from: format.sampleRate, to: Double(ClapFrontEnd.sampleRate))
    }

    static func resample(_ input: [Float], from source: Double, to destination: Double) -> [Float] {
        guard source != destination, !input.isEmpty,
              let inFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: source, channels: 1, interleaved: false),
              let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: destination, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: inFormat, to: outFormat),
              let inBuffer = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: AVAudioFrameCount(input.count))
        else { return input }
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        inBuffer.frameLength = AVAudioFrameCount(input.count)
        input.withUnsafeBufferPointer { inBuffer.floatChannelData![0].update(from: $0.baseAddress!, count: input.count) }
        let capacity = AVAudioFrameCount(Double(input.count) * destination / source) + 16
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity) else { return input }
        nonisolated(unsafe) var delivered = false
        let status = converter.convert(to: outBuffer, error: nil) { _, outStatus in
            if delivered { outStatus.pointee = .endOfStream; return nil }
            delivered = true; outStatus.pointee = .haveData; return inBuffer
        }
        guard status != .error, let data = outBuffer.floatChannelData else { return input }
        return Array(UnsafeBufferPointer(start: data[0], count: Int(outBuffer.frameLength)))
    }

    // MARK: - Vectors

    public static func normalised(_ v: [Float]) -> [Float] {
        let norm = v.reduce(0) { $0 + $1 * $1 }.squareRoot()
        return norm > 0 ? v.map { $0 / norm } : v
    }

    /// Cosine similarity of two unit vectors.
    public static func similarity(_ a: [Float], _ b: [Float]) -> Float {
        var s: Float = 0
        for i in 0..<min(a.count, b.count) { s += a[i] * b[i] }
        return s
    }

    static func array(_ values: [Float], shape: [Int]) -> NDArray {
        var a = NDArray(shape: shape, scalarType: .float32)
        let view = a.mutableView(as: Float.self)
        view.withUnsafeMutablePointer { p, _, _ in values.withUnsafeBufferPointer { p.update(from: $0.baseAddress!, count: values.count) } }
        return a
    }

    static func ints(_ values: [Int32], shape: [Int]) -> NDArray {
        var a = NDArray(shape: shape, scalarType: .int32)
        let view = a.mutableView(as: Int32.self)
        view.withUnsafeMutablePointer { p, _, _ in values.withUnsafeBufferPointer { p.update(from: $0.baseAddress!, count: values.count) } }
        return a
    }

    static func floats(_ array: NDArray) -> [Float] {
        let count = array.shape.reduce(1, *)
        var out = [Float](repeating: 0, count: count)
        array.view(as: Float.self).withUnsafePointer { p, _, _ in out.withUnsafeMutableBufferPointer { $0.baseAddress!.update(from: p, count: count) } }
        return out
    }
}
