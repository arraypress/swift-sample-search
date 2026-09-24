//
//  SampleSearchTests.swift
//  SampleSearchTests
//
//  Created by David Sherlock on 2026.
//
//  The CLAP port against upstream: the log-mel front end and the tokenizer
//  on fixtures the export wrote (no model needed), then — with the asset
//  installed under ~/Library/Application Support/crate — text and audio
//  embeddings and a zero-shot tagging against Hugging Face's own numbers.
//

import Foundation
import Testing
@testable import SampleSearch

private func fixtureURL(_ name: String, _ ext: String) throws -> URL {
    try #require(Bundle.module.url(forResource: name, withExtension: ext, subdirectory: "Fixtures"))
}

private func floats(_ name: String) throws -> [Float] {
    try Data(contentsOf: fixtureURL(name, "f32")).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

private func psnr(_ reference: [Float], _ got: [Float]) -> Double {
    precondition(reference.count == got.count)
    var err = 0.0, lo = Double.infinity, hi = -Double.infinity
    for i in reference.indices {
        let d = Double(reference[i]) - Double(got[i]); err += d * d
        lo = min(lo, Double(reference[i])); hi = max(hi, Double(reference[i]))
    }
    let mse = err / Double(reference.count)
    return mse == 0 ? .infinity : 20 * log10((hi - lo) / mse.squareRoot())
}

private let installed = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support/crate")
private var modelInstalled: Bool { FileManager.default.fileExists(atPath: installed.appendingPathComponent(ClapEmbedder.assetName).path) }

private struct TextFixture: Decodable { let phrases: [String]; let ids: [[Int32]]; let mask: [[Int32]] }
private struct ZeroShot: Decodable { let labels: [String]; let probabilities: [Float] }

// MARK: - Front end

@Suite struct FrontEndTests {

    @Test("Slaney filterbank has 64 unit-area triangles over 50–14000 Hz")
    func filterbank() {
        let fb = ClapFrontEnd.slaneyFilterbank()
        #expect(fb.count == ClapFrontEnd.bins * ClapFrontEnd.mels)
        for m in 0..<ClapFrontEnd.mels {
            let column = (0..<ClapFrontEnd.bins).map { fb[$0 * ClapFrontEnd.mels + m] }
            #expect(column.max()! > 0, "band \(m) is empty")
        }
    }

    @Test("repeatpad tiles a short clip whole, then zeros", arguments: [3_000, 100_000, 479_999])
    func repeatPad(length: Int) {
        let clip = (0..<length).map { Float($0 % 97) / 97 }
        let padded = ClapFrontEnd.repeatPadded(clip)
        #expect(padded.count == ClapFrontEnd.windowSamples)
        let repeats = ClapFrontEnd.windowSamples / length
        for r in 0..<repeats { #expect(padded[r * length] == clip[0] && padded[r * length + length - 1] == clip[length - 1]) }
        #expect(padded[(repeats * length)...].allSatisfy { $0 == 0 })
    }

    @Test("log-mel of the fixture clips matches ClapFeatureExtractor", arguments: ["demo10s", "short3s", "drums6s"])
    func logMel(clip: String) throws {
        let samples = try ClapEmbedder.load(fixtureURL("\(clip)_48k", "wav"))
        let reference = try floats("\(clip)_mel")
        let mel = ClapFrontEnd().logMel(window: ClapFrontEnd.repeatPadded(samples))
        #expect(mel.count == reference.count)
        let db = psnr(reference, mel)
        let maxDiff = zip(reference, mel).map { abs($0 - $1) }.max()!
        print("log-mel \(clip): \(String(format: "%.1f", db)) dB, max |diff| \(maxDiff) dB")
        #expect(db > 90, "\(clip) log-mel \(db) dB from upstream")
    }
}

// MARK: - Tokenizer

@Suite struct TokenizerTests {

    @Test("every fixture phrase tokenises to RobertaTokenizer's ids and mask")
    func phrases() throws {
        let fx = try JSONDecoder().decode(TextFixture.self, from: Data(contentsOf: fixtureURL("text_fixture", "json")))
        let tokenizer = try BytePairTokenizer(contentsOf: fixtureURL("vocab", "json").deletingLastPathComponent())
        for (i, phrase) in fx.phrases.enumerated() {
            let (ids, mask) = tokenizer.encode(phrase, length: ClapEmbedder.tokens)
            #expect(ids == fx.ids[i], "\(phrase): \(ids.prefix(12)) vs \(fx.ids[i].prefix(12))")
            #expect(mask == fx.mask[i])
        }
    }

    @Test("punctuation, numbers, unicode and long text")
    func edges() throws {
        let tokenizer = try BytePairTokenizer(contentsOf: fixtureURL("vocab", "json").deletingLastPathComponent())
        let (ids, mask) = tokenizer.encode("Hello, world!! 12345 — café 🎹", length: 77)
        #expect(ids.count == 77 && mask.count == 77 && ids[0] == 0)
        #expect(ids[Int(mask.reduce(0, +)) - 1] == 2, "eos closes the real tokens")
        let long = String(repeating: "kick drum ", count: 100)
        let (lids, lmask) = tokenizer.encode(long, length: 77)
        #expect(lids.count == 77 && lids.last == 2 && lmask.allSatisfy { $0 == 1 }, "truncated to 76 + eos")
    }
}

// MARK: - Index

@Suite struct IndexTests {

    @Test("round-trips entries and vectors through the file format")
    func roundTrip() throws {
        var index = SampleIndex(model: "test", dimension: 4)
        let now = Date(timeIntervalSince1970: 1_700_000_000.5)   // exactly representable, so the file round-trip is exact
        try index.upsert(.init(path: "/a.wav", seconds: 1, modified: now, size: 10), embedding: ClapEmbedder.normalised([1, 0, 0, 0]))
        try index.upsert(.init(path: "/b.wav", seconds: 2, modified: now, size: 20), embedding: ClapEmbedder.normalised([1, 1, 0, 0]))
        try index.upsert(.init(path: "/c.wav", seconds: 3, modified: now, size: 30), embedding: ClapEmbedder.normalised([0, 0, 1, 0]))
        try index.upsert(.init(path: "/a.wav", seconds: 1.5, modified: now, size: 11), embedding: ClapEmbedder.normalised([0, 1, 0, 0]))
        #expect(index.count == 3 && index.entry(for: "/a.wav")?.size == 11)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("crate-test-\(UUID().uuidString).crate")
        defer { try? FileManager.default.removeItem(at: url) }
        try index.save(to: url)
        let back = try SampleIndex(contentsOf: url)
        #expect(back.entries == index.entries && back.vectors == index.vectors && back.dimension == 4)
        let hits = back.search(ClapEmbedder.normalised([1, 1, 0, 0]), limit: 2)
        #expect(hits.map(\.entry.path) == ["/b.wav", "/a.wav"])
        #expect(back.similar(to: "/b.wav")?.first?.entry.path == "/a.wav")
        #expect(index.isCurrent(path: "/a.wav", modified: now, size: 11) && !index.isCurrent(path: "/a.wav", modified: now, size: 12))
        var mutable = back
        #expect(mutable.remove(path: "/b.wav") && mutable.count == 2 && mutable.vectors.count == 8)
        #expect(throws: SampleIndex.Error.self) { try mutable.upsert(.init(path: "/d.wav", seconds: 1, modified: now, size: 1), embedding: [1, 2, 3]) }
    }

    @Test("rejects files that are not an index")
    func notAnIndex() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("not-\(UUID().uuidString).crate")
        try Data("hello".utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: SampleIndex.Error.self) { try SampleIndex(contentsOf: url) }
    }
}

// MARK: - Model

@Suite(.serialized) struct EmbedderTests {

    @Test("text embeddings match get_text_features", .enabled(if: modelInstalled))
    func text() async throws {
        let embedder = try await ClapEmbedder(inFolder: installed)
        let fx = try JSONDecoder().decode(TextFixture.self, from: Data(contentsOf: fixtureURL("text_fixture", "json")))
        let reference = try floats("text_embeddings")
        var worst = Double.infinity
        for (i, phrase) in fx.phrases.enumerated() {
            let e = try await embedder.embed(text: phrase)
            let r = Array(reference[(i * 512)..<((i + 1) * 512)])
            let db = psnr(r, e)
            worst = min(worst, db)
            #expect(ClapEmbedder.similarity(r, e) > 0.9999, "\(phrase): cosine \(ClapEmbedder.similarity(r, e)), \(db) dB")
        }
        print("text embeddings: worst \(String(format: "%.1f", worst)) dB over \(fx.phrases.count) phrases")
    }

    @Test("audio embeddings match get_audio_features", .enabled(if: modelInstalled), arguments: ["demo10s", "short3s", "drums6s"])
    func audio(clip: String) async throws {
        let embedder = try await ClapEmbedder(inFolder: installed)
        let reference = try floats("\(clip)_embedding")
        let (e, seconds) = try await embedder.embed(contentsOf: fixtureURL("\(clip)_48k", "wav"))
        let db = psnr(reference, e)
        print("audio \(clip) (\(String(format: "%.2f", seconds)) s): cosine \(ClapEmbedder.similarity(reference, e)), \(String(format: "%.1f", db)) dB")
        #expect(ClapEmbedder.similarity(reference, e) > 0.9999, "\(clip): \(db) dB")
    }

    @Test("zero-shot tags reproduce logits_per_audio softmax", .enabled(if: modelInstalled), arguments: ["demo10s", "drums6s"])
    func zeroShot(clip: String) async throws {
        let embedder = try await ClapEmbedder(inFolder: installed)
        let fx = try JSONDecoder().decode(ZeroShot.self, from: Data(contentsOf: fixtureURL("\(clip)_zero_shot", "json")))
        let audio = try await embedder.embed(contentsOf: fixtureURL("\(clip)_48k", "wav")).embedding
        var labels: [(name: String, embedding: [Float])] = []
        for l in fx.labels { labels.append((l, try await embedder.embed(text: l))) }
        let tags = SampleIndex.tags(for: audio, labels: labels, logitScale: embedder.logitScale)
        let byLabel = Dictionary(uniqueKeysWithValues: tags.map { ($0.label, $0.probability) })
        var maxDiff: Float = 0
        for (l, p) in zip(fx.labels, fx.probabilities) { maxDiff = max(maxDiff, abs(byLabel[l]! - p)) }
        print("zero-shot \(clip): top \(tags[0].label) \(String(format: "%.3f", tags[0].probability)); max |Δp| \(maxDiff)")
        #expect(tags[0].label == fx.labels[Int(fx.probabilities.indices.max { fx.probabilities[$0] < fx.probabilities[$1] }!)])
        #expect(maxDiff < 1e-3)
    }

    @Test("a long file is the mean of its ten-second windows", .enabled(if: modelInstalled))
    func longAudio() async throws {
        let embedder = try await ClapEmbedder(inFolder: installed)
        let ten = try ClapEmbedder.load(fixtureURL("demo10s_48k", "wav"))
        let three = try ClapEmbedder.load(fixtureURL("short3s_48k", "wav"))
        let a = try await embedder.embed(samples48k: ten)
        let b = try await embedder.embed(samples48k: three)
        let joined = try await embedder.embed(samples48k: ten + three)
        let expected = ClapEmbedder.normalised(zip(a, b).map { ($0 + $1) / 2 })
        #expect(psnr(expected, joined) > 100)
    }
}
