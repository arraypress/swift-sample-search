# CLAUDE.md — swift-sample-search

CLAP (`laion/larger_clap_music`) on Core AI: `ClapEmbedder` (audio and text
vectors), `ClapFrontEnd` (log-mel), `BytePairTokenizer` (RoBERTa BPE),
`SampleIndex` (on-disk vectors, search/similar/tags). Module `SampleSearch`;
the CLI is `../swift-crate-cli` (binary `crate`). Added 2026-09-24.

## Build & test
```bash
swift build && swift test          # front end + tokenizer + index need no model
```
Model-backed tests (`EmbedderTests`) run only when the asset and
`clap-support/` are installed under `~/Library/Application Support/crate/`:
`uv run Tools/export_clap.py --install demo10s=… short3s=… drums6s=…` writes
both and the fixtures in one go (probe venv: coreai-torch 0.4.2, torch 2.13,
transformers 5.17, Python 3.12).

## Traps (all hit)
- **transformers 5 changed two APIs the export leans on.** `save_vocabulary`
  writes a single `tokenizer.model` (the vocab, no merges) — the script
  downloads the hub's `vocab.json`/`merges.txt` instead. `get_text_features`
  / `get_audio_features` return an output object; the normalised vector is
  `.pooler_output`.
- **`AVAudioFile.read(into:)` can stop short of `file.length`** — 788 of
  480,000 frames missing on a soundfile-written FLOAT WAV. Read in a loop
  until `framePosition == length` (`ClapEmbedder.load`). And a 48 kHz file
  must not go through AVAudioConverter at all: even a same-rate conversion
  drops frames, which shifts every repeat-pad tile and costs 100 dB.
- **`rand_trunc` is random.** Audio over 10 s cannot match upstream; the
  library averages consecutive windows. Fixtures are ≤ 10 s on purpose.
- **The checkpoint's logit scale is ~1** (`logit_scale_a` = 0.027 stored,
  exp → 1.03), so upstream's zero-shot softmax is nearly flat. Report it as
  is (`Tag.probability`, verified to 1e-8) but rank by cosine (`Tag.score`).
- Slaney filterbank (`norm="slaney"`, `mel_scale="slaney"`) is the one the
  `rand_trunc` path uses; the htk bank in the extractor is for `fusion`.
- Index dates are stored as `secondsSince1970` — ISO 8601 drops the fraction
  and breaks the round-trip equality test.
- Test fixtures are reached with `subdirectory: "Fixtures"` (the `.copy`
  keeps the folder name).
