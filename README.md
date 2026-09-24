# SampleSearch

On-device **CLAP** for Swift — a sound and a sentence each become a 512-number vector in one
space, so a sample library can be searched by words, searched by example, and labelled from any
list of phrases without training anything. LAION's `larger_clap_general` (Apache 2.0) runs on
Core AI; the log-mel front end, the RoBERTa tokenizer, the L2 normalisation and the on-disk
index are Swift. Nothing leaves the machine.

```swift
import SampleSearch

let clap = try await ClapEmbedder(inFolder: modelFolder)          // the .aimodel beside clap-support/
let kick = try await clap.embed(contentsOf: kickURL).embedding    // [Float] × 512, unit length
let words = try await clap.embed(text: "a punchy 808 kick drum")
ClapEmbedder.similarity(kick, words)                              // cosine, about 0.3+ for a real match

var index = SampleIndex(model: ClapEmbedder.assetName)
try index.upsert(.init(path: kickURL.path, seconds: 1.2, modified: date, size: bytes), embedding: kick)
try index.save(to: indexURL)
index.search(words, limit: 20)                                    // [Hit] best first
index.similar(to: kickURL.path)                                   // the rest of the library, nearest first
SampleIndex.tags(for: kick, labels: labelVectors, logitScale: clap.logitScale)   // zero-shot, best first
```

## What it is exactly

The model is Hugging Face's `ClapModel` for `laion/larger_clap_general`: the HTSAT audio encoder
(68 M parameters) and the RoBERTa text encoder (125 M) with their projection layers, exported to
one `.aimodel` with two entry points. `audio` takes a log-mel `[1, 1, 1001, 64]` — ten seconds
at 48 kHz — and returns the projected embedding; `text` takes 77 token ids and an attention mask.
The host normalises to unit length, as `get_audio_features` and `get_text_features` do.

The front end is `ClapFeatureExtractor` for the checkpoint, reproduced in `ClapFrontEnd`:
48 kHz mono, a periodic Hann of 1024, hop 480, centred reflect padding, power spectrum, librosa's
Slaney filterbank (64 bands, 50–14000 Hz, area-normalised) and `10·log10(max(1e-10, x))`. A clip
shorter than ten seconds is tiled whole as many times as fits and zero-padded (`repeatpad`).
The tokenizer is RoBERTa's byte-level BPE read from the checkpoint's own `vocab.json` and
`merges.txt` (`BytePairTokenizer`), and the checkpoint's logit scale comes from `scales.json`
beside them — the three files ship as `clap-support/` next to the asset.

One deliberate departure, stated because upstream's is random: `ClapFeatureExtractor` crops a
random ten seconds from anything longer (`rand_trunc`). `ClapEmbedder` instead embeds each
consecutive ten-second window and averages, then re-normalises — deterministic, and it hears the
whole file. A 48 kHz file is read sample for sample; other rates go through AVAudioConverter at
its best quality, which is a resampler difference, not a model one.

## Measured against upstream

`Tools/export_clap.py` writes the asset, the support files and fixtures from the Hugging Face
model in one run — the phrases' ids and embeddings, three clips' log-mels and embeddings, and
their zero-shot probabilities over twenty labels — and `swift test` compares:

| Stage | Agreement with `transformers` |
|---|---|
| Tokenizer, 20 phrases | ids and masks identical |
| Log-mel, 3 clips (3.0 s, 3.8 s, 10.0 s) | 116–156 dB PSNR; max 0.013 dB on any bin |
| Text embeddings, 20 phrases | 142 dB worst; cosine > 0.9999 |
| Audio embeddings, 3 clips | 144–147 dB; cosine 0.99999+ |
| Zero-shot probabilities, 20 labels | max difference 1e-8 |

On an M3 Max the audio encoder takes about 0.3 s per ten-second window and the text encoder
about 0.1 s per phrase, model load included in the first call.

**Why the general checkpoint and not `larger_clap_general`:** the music checkpoint's Hugging Face
conversion is broken — in `transformers` itself every clip maps to nearly the same vector
(audio–audio cosines 0.81–0.99 between a string loop, a kick and a full mix; every audio–text
score 0.01–0.04; stored logit scale 1.03). The first export reproduced it to 147 dB and was
useless. `larger_clap_general` (trained on general audio, music and speech; logit scale 38.7)
puts a kick file at 0.49 against "a kick drum" and a string loop at 0.22 against "a string
instrument", and `clap-htsat-unfused` behaves the same; the export takes either by `--name`.

## The model

Not bundled (797 MB). Export it yourself from the Hugging Face weights, or download the export:

```sh
uv run Tools/export_clap.py --install          # from laion/larger_clap_general, with fixtures
hf download arraypress/crate-clap-general --local-dir models
```

The command-line tool is [`crate`](https://github.com/arraypress/swift-crate-cli).

## Requirements

- macOS 27+ (Core AI), Apple silicon
- Swift 6

## License

MIT — see [LICENSE](LICENSE). The model is LAION's, Apache 2.0.
