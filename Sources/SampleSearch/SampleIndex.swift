//
//  SampleIndex.swift
//  SampleSearch
//
//  Created by David Sherlock on 2026.
//
//  A library's embeddings on disk and the three questions asked of them:
//  which sounds match these words, which sound like this one, and which of
//  these labels fits each sound. One file: a small header, a JSON manifest
//  of entries (path, length, modification date, size) and the unit vectors
//  as raw float32, so a hundred thousand samples is 200 MB and loads in
//  a moment. Scores are cosine similarities of unit vectors; zero-shot
//  tagging is upstream's `logits_per_audio` softmax, with the checkpoint's
//  own logit scale.
//

import Foundation

/// Paths and their CLAP vectors, searchable by text or by example.
public struct SampleIndex: Sendable {

    public struct Entry: Codable, Sendable, Equatable {
        public var path: String
        public var seconds: Double
        public var modified: Date
        public var size: Int
        public init(path: String, seconds: Double, modified: Date, size: Int) {
            self.path = path; self.seconds = seconds; self.modified = modified; self.size = size
        }
    }

    public struct Hit: Sendable {
        public let entry: Entry
        public let score: Float
    }

    public struct Tag: Sendable {
        public let label: String
        /// Softmax over the offered labels of cosine × logit scale, as upstream's zero-shot pipeline reports.
        public let probability: Float
        public let score: Float
    }

    public struct Manifest: Codable, Sendable {
        public var model: String
        public var dimension: Int
        public var created: Date
        public var updated: Date
        public var entries: [Entry]
    }

    public static let magic = "CRATEIDX"
    public static let version: UInt32 = 1

    public private(set) var manifest: Manifest
    /// `entries.count × dimension`, row-major unit vectors.
    public private(set) var vectors: [Float]

    public var entries: [Entry] { manifest.entries }
    public var count: Int { manifest.entries.count }
    public var dimension: Int { manifest.dimension }

    public init(model: String, dimension: Int = ClapEmbedder.dimension) {
        manifest = Manifest(model: model, dimension: dimension, created: Date(), updated: Date(), entries: [])
        vectors = []
    }

    public enum Error: Swift.Error, CustomStringConvertible {
        case notAnIndex(String), corrupt(String), dimensionMismatch(expected: Int, got: Int)
        public var description: String {
            switch self {
            case .notAnIndex(let p): return "\(p) is not a crate index"
            case .corrupt(let m): return "index is corrupt: \(m)"
            case .dimensionMismatch(let e, let g): return "vector has \(g) values; the index holds \(e)"
            }
        }
    }

    // MARK: - Editing

    /// Adds or replaces the entry for `entry.path`.
    public mutating func upsert(_ entry: Entry, embedding: [Float]) throws {
        guard embedding.count == manifest.dimension else { throw Error.dimensionMismatch(expected: manifest.dimension, got: embedding.count) }
        if let i = manifest.entries.firstIndex(where: { $0.path == entry.path }) {
            manifest.entries[i] = entry
            vectors.replaceSubrange((i * dimension)..<((i + 1) * dimension), with: embedding)
        } else {
            manifest.entries.append(entry)
            vectors.append(contentsOf: embedding)
        }
        manifest.updated = Date()
    }

    @discardableResult
    public mutating func remove(path: String) -> Bool {
        guard let i = manifest.entries.firstIndex(where: { $0.path == path }) else { return false }
        manifest.entries.remove(at: i)
        vectors.removeSubrange((i * dimension)..<((i + 1) * dimension))
        manifest.updated = Date()
        return true
    }

    public func entry(for path: String) -> Entry? { manifest.entries.first { $0.path == path } }

    /// True when the file at `path` is already indexed with this modification date and size.
    public func isCurrent(path: String, modified: Date, size: Int) -> Bool {
        guard let e = entry(for: path) else { return false }
        return e.size == size && abs(e.modified.timeIntervalSince(modified)) < 1
    }

    public func embedding(for path: String) -> [Float]? {
        guard let i = manifest.entries.firstIndex(where: { $0.path == path }) else { return nil }
        return Array(vectors[(i * dimension)..<((i + 1) * dimension)])
    }

    // MARK: - Questions

    /// Entries ranked by cosine similarity to `query` (a unit vector).
    public func search(_ query: [Float], limit: Int = 20, excluding path: String? = nil) -> [Hit] {
        let d = dimension
        var hits: [Hit] = []
        hits.reserveCapacity(count)
        for (i, e) in manifest.entries.enumerated() where e.path != path {
            var s: Float = 0
            let base = i * d
            for k in 0..<d { s += vectors[base + k] * query[k] }
            hits.append(Hit(entry: e, score: s))
        }
        hits.sort { $0.score > $1.score }
        return Array(hits.prefix(limit))
    }

    /// Entries most like the one at `path`, itself excluded.
    public func similar(to path: String, limit: Int = 20) -> [Hit]? {
        guard let q = embedding(for: path) else { return nil }
        return search(q, limit: limit, excluding: path)
    }

    /// Zero-shot: each offered label's probability for one embedding, best first.
    public static func tags(for embedding: [Float], labels: [(name: String, embedding: [Float])], logitScale: Float) -> [Tag] {
        let scores = labels.map { ClapEmbedder.similarity(embedding, $0.embedding) }
        let logits = scores.map { $0 * logitScale }
        let m = logits.max() ?? 0
        let exps = logits.map { exp($0 - m) }
        let z = exps.reduce(0, +)
        return zip(labels, zip(scores, exps)).map { Tag(label: $0.name, probability: $1.1 / z, score: $1.0) }
            .sorted { $0.probability > $1.probability }
    }

    // MARK: - Files

    public func save(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .secondsSince1970
        let json = try encoder.encode(manifest)
        var data = Data(Self.magic.utf8)
        var version = Self.version.littleEndian
        var length = UInt32(json.count).littleEndian
        withUnsafeBytes(of: &version) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &length) { data.append(contentsOf: $0) }
        data.append(json)
        vectors.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
    }

    public init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url)
        let head = Self.magic.utf8.count
        guard data.count >= head + 8, String(decoding: data.prefix(head), as: UTF8.self) == Self.magic else { throw Error.notAnIndex(url.path) }
        let version = data[head..<(head + 4)].withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(as: UInt32.self)) }
        guard version == Self.version else { throw Error.corrupt("version \(version); this build reads \(Self.version)") }
        let length = Int(data[(head + 4)..<(head + 8)].withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(as: UInt32.self)) })
        let jsonStart = head + 8
        guard data.count >= jsonStart + length else { throw Error.corrupt("truncated manifest") }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .secondsSince1970
        let manifest: Manifest
        do { manifest = try decoder.decode(Manifest.self, from: data[jsonStart..<(jsonStart + length)]) } catch { throw Error.corrupt("manifest: \(error)") }
        let blob = data[(jsonStart + length)...]
        let expected = manifest.entries.count * manifest.dimension
        guard blob.count == expected * MemoryLayout<Float>.size else { throw Error.corrupt("\(blob.count) bytes of vectors for \(manifest.entries.count) entries") }
        var vectors = [Float](repeating: 0, count: expected)
        _ = vectors.withUnsafeMutableBytes { blob.copyBytes(to: $0) }
        self.manifest = manifest
        self.vectors = vectors
    }
}
