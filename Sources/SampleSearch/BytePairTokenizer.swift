//
//  BytePairTokenizer.swift
//  SampleSearch
//
//  Created by David Sherlock on 2026.
//
//  RoBERTa's byte-level byte-pair encoding, as Hugging Face's
//  `RobertaTokenizer` does it: GPT-2's pre-tokenising pattern, bytes mapped
//  to printable Unicode, merges applied by rank, `<s>` and `</s>` around the
//  ids, `<pad>` to the fixed length the text encoder was exported with. Read
//  from the checkpoint's own `vocab.json` and `merges.txt`, so a phrase
//  tokenises exactly as it did for the model.
//

import Foundation

/// The text side of CLAP's front end.
public struct BytePairTokenizer: Sendable {

    public let bos: Int32 = 0    // <s>
    public let pad: Int32 = 1    // <pad>
    public let eos: Int32 = 2    // </s>
    public let unknown: Int32 = 3

    private let vocab: [String: Int32]
    private let ranks: [String: Int]           // "a b" → rank
    private let byteToUnicode: [Character]
    private let pattern: NSRegularExpression

    /// `folder` holds `vocab.json` and `merges.txt`.
    public init(contentsOf folder: URL) throws {
        let vocabData = try Data(contentsOf: folder.appendingPathComponent("vocab.json"))
        guard let table = try JSONSerialization.jsonObject(with: vocabData) as? [String: Int] else {
            throw TokenizerError.malformed("vocab.json is not a string → int table")
        }
        vocab = table.mapValues { Int32($0) }
        let merges = try String(contentsOf: folder.appendingPathComponent("merges.txt"), encoding: .utf8)
        var ranks: [String: Int] = [:]
        var rank = 0
        for line in merges.split(separator: "\n", omittingEmptySubsequences: true) {
            if line.hasPrefix("#") { continue }
            ranks[String(line)] = rank; rank += 1
        }
        self.ranks = ranks
        byteToUnicode = Self.bytesToUnicode()
        // GPT-2's pattern: contractions, words with an optional leading space, numbers, punctuation runs, whitespace.
        pattern = try NSRegularExpression(pattern: #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#)
    }

    public enum TokenizerError: Error { case malformed(String) }

    /// Ids and attention mask for one phrase, padded or truncated to `length`.
    public func encode(_ text: String, length: Int) -> (ids: [Int32], mask: [Int32]) {
        var ids: [Int32] = [bos]
        for piece in pretokenize(text) {
            for token in bpe(piece) { ids.append(vocab[token] ?? unknown) }
        }
        if ids.count > length - 1 { ids = Array(ids.prefix(length - 1)) }
        ids.append(eos)
        let mask = [Int32](repeating: 1, count: ids.count) + [Int32](repeating: 0, count: length - ids.count)
        ids.append(contentsOf: [Int32](repeating: pad, count: length - ids.count))
        return (ids, mask)
    }

    /// The ids without padding, for tests and for anyone counting tokens.
    public func tokenIDs(_ text: String) -> [Int32] {
        var ids: [Int32] = [bos]
        for piece in pretokenize(text) { for token in bpe(piece) { ids.append(vocab[token] ?? unknown) } }
        ids.append(eos)
        return ids
    }

    // MARK: - Pieces

    private func pretokenize(_ text: String) -> [String] {
        let ns = text as NSString
        return pattern.matches(in: text, range: NSRange(location: 0, length: ns.length)).map { ns.substring(with: $0.range) }
    }

    /// Byte-level BPE on one pre-token: bytes → unicode symbols, then merge the lowest-ranked pair until none is left.
    private func bpe(_ piece: String) -> [String] {
        var word: [String] = piece.utf8.map { String(byteToUnicode[Int($0)]) }
        guard word.count > 1 else { return word }
        while true {
            var best: (rank: Int, index: Int)? = nil
            for i in 0..<(word.count - 1) {
                if let r = ranks[word[i] + " " + word[i + 1]], best == nil || r < best!.rank { best = (r, i) }
            }
            guard let (_, i) = best else { break }
            let merged = word[i] + word[i + 1]
            var next: [String] = []
            var j = 0
            while j < word.count {
                if j < word.count - 1 && word[j] == word[i] && word[j + 1] == word[i + 1] {
                    next.append(merged); j += 2
                } else {
                    next.append(word[j]); j += 1
                }
            }
            word = next
            if word.count == 1 { break }
        }
        return word
    }

    /// GPT-2's `bytes_to_unicode`: printable bytes map to themselves, the rest to U+0100 onwards.
    static func bytesToUnicode() -> [Character] {
        var bytes: [Int] = Array(33...126) + Array(161...172) + Array(174...255)
        var chars = bytes
        var n = 0
        for b in 0..<256 where !bytes.contains(b) {
            bytes.append(b); chars.append(256 + n); n += 1
        }
        var table = [Character](repeating: " ", count: 256)
        for (b, c) in zip(bytes, chars) { table[b] = Character(UnicodeScalar(c)!) }
        return table
    }
}
