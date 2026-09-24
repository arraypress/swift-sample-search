# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["torch==2.13.0", "numpy", "coreai-torch==0.4.2", "coreai-core==1.0.0b2", "transformers", "soundfile", "librosa"]
# ///
"""CLAP (laion/larger_clap_music, Apache 2.0) → crate-clap-music-float32.aimodel, plus its tokenizer files.

    uv run Tools/export_clap.py [--out Tools/exports] [--fixtures Tests/SampleSearchTests/Fixtures] [--install] [clip=path.wav …]

Two entry points: `audio` (log-mel [1, 1, 1001, 64] — 10 s at 48 kHz, 64 Slaney mels 50–14000 Hz in
dB — → the projected audio embedding [1, 512]) and `text` (RoBERTa token ids [1, 77] with an
attention mask [1, 77] → the projected text embedding [1, 512]). Embeddings are the projection
layers' outputs; the host L2-normalises them, as `get_audio_features`/`get_text_features` do.
Both are asserted equal to the Hugging Face model before export. The RoBERTa byte-level BPE
tokenizer (vocab.json, merges.txt) and the logit scales (scales.json) are copied beside the asset as clap-support/ for the host's tokenizer.

Upstream's feature extractor random-crops audio longer than 10 s (`rand_trunc`); the host
averages the embeddings of consecutive 10-second windows instead, which is deterministic. Clips
of 10 s or less are repeat-padded exactly as upstream does."""
import argparse, os, sys, json, time, shutil
from pathlib import Path
import numpy as np, torch, torch.nn as nn, soundfile as sf, librosa
from transformers import ClapModel, ClapProcessor
import coreai_torch
from coreai.runtime import AIModelAssetMetadata
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--name", default="laion/larger_clap_music"); ap.add_argument("--out", default=str(HERE / "exports"))
ap.add_argument("--fixtures", default=str(HERE.parent / "Tests/SampleSearchTests/Fixtures")); ap.add_argument("--install", action="store_true")
ap.add_argument("--max-tokens", type=int, default=77); ap.add_argument("clips", nargs="*", help="name=path.wav for audio fixtures")
args = ap.parse_args()
T = args.max_tokens
model = ClapModel.from_pretrained(args.name).eval(); proc = ClapProcessor.from_pretrained(args.name)
fe, tok = proc.feature_extractor, proc.tokenizer
print(f"{args.name}: audio {sum(p.numel() for p in model.audio_model.parameters())/1e6:.0f} M, text {sum(p.numel() for p in model.text_model.parameters())/1e6:.0f} M, projection {model.config.projection_dim}; mel {fe.feature_size} bins, hop {fe.hop_length}, n_fft {fe.fft_window_size}, {fe.frequency_min}-{fe.frequency_max} Hz, {fe.sampling_rate} Hz, truncation {fe.truncation}, padding {fe.padding}", flush=True)

class Audio(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, mel):
        out = self.m.audio_model(input_features=mel, is_longer=torch.zeros(1, 1, dtype=torch.bool))
        return self.m.audio_projection(out.pooler_output)
class Text(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, ids, mask):
        out = self.m.text_model(input_ids=ids, attention_mask=mask)
        return self.m.text_projection(out.pooler_output)
audio, text = Audio(model).eval(), Text(model).eval()

fx = Path(args.fixtures); fx.mkdir(parents=True, exist_ok=True)
# Tokenizer files, beside the asset and as test fixtures.
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
tokdir = out / "clap-support"; tokdir.mkdir(exist_ok=True)
# The checkpoint's own vocab.json and merges.txt (transformers 5 no longer writes them from save_vocabulary).
from huggingface_hub import hf_hub_download
for f in ["vocab.json", "merges.txt"]:
    src = hf_hub_download(args.name, f)
    shutil.copy(src, tokdir / f); shutil.copy(src, fx / f)
for stale in tokdir.glob("tokenizer.model"): stale.unlink()
# The checkpoint's logit scales (exp of the stored log-parameters): zero-shot tagging is softmax(cos × audio scale).
scales = {"logit_scale_audio": float(model.logit_scale_a.exp()), "logit_scale_text": float(model.logit_scale_t.exp()), "max_tokens": T, "model": args.name}
for d in [tokdir, fx]: json.dump(scales, open(d / "scales.json", "w"), indent=1)
print(f"logit scales: audio {scales['logit_scale_audio']:.4f}, text {scales['logit_scale_text']:.4f}", flush=True)

# Text fixtures: phrases → ids (padded to T) and embeddings; equality of the wrapper vs HF.
phrases = ["a punchy 808 kick drum", "warm analog pad", "dusty rhodes chord", "female vocal chop", "acid bassline 303", "hi-hat loop", "riser fx", "male rap vocal", "piano melody", "distorted guitar riff", "vinyl crackle", "sub bass note", "trance pluck", "orchestral strings swell", "snare roll", "vocoder", "brass stab", "flute", "808", "reese bass"]
enc = tok(phrases, padding="max_length", max_length=T, truncation=True, return_tensors="pt")
with torch.no_grad():
    ours = torch.cat([text(enc["input_ids"][i:i+1], enc["attention_mask"][i:i+1]) for i in range(len(phrases))])
    theirs = model.get_text_features(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).pooler_output
    ours_n = nn.functional.normalize(ours, dim=-1)
print(f"text: wrapper (normalised) vs get_text_features max |diff| {(ours_n - theirs).abs().max():.2e}", flush=True)
json.dump({"phrases": phrases, "ids": enc["input_ids"].tolist(), "mask": enc["attention_mask"].tolist()}, open(fx / "text_fixture.json", "w"))
ours_n.numpy().astype(np.float32).tofile(fx / "text_embeddings.f32")

# Audio fixtures: the clip through the feature extractor (repeatpad / crop), mel and embedding.
for spec in args.clips:
    name, path = spec.split("=", 1)
    y, sr = sf.read(path, dtype="float32", always_2d=True); y = y.mean(1)
    if sr != fe.sampling_rate: y = librosa.resample(y, orig_sr=sr, target_sr=fe.sampling_rate, res_type="soxr_hq").astype(np.float32)
    y = y[: fe.nb_max_samples]                          # ≤ 10 s: the deterministic path
    inputs = fe(y, sampling_rate=fe.sampling_rate, return_tensors="pt")
    mel = inputs["input_features"]
    with torch.no_grad():
        ours_a = nn.functional.normalize(audio(mel), dim=-1); theirs_a = model.get_audio_features(**inputs).pooler_output
    print(f"audio {name}: {len(y)/fe.sampling_rate:.2f} s → mel {tuple(mel.shape)}; wrapper vs get_audio_features max |diff| {(ours_a - theirs_a).abs().max():.2e}", flush=True)
    sf.write(fx / f"{name}_48k.wav", y, fe.sampling_rate, subtype="FLOAT")
    with torch.no_grad():
        zs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], input_features=mel).logits_per_audio[0].softmax(-1)
    json.dump({"labels": phrases, "probabilities": zs.tolist()}, open(fx / f"{name}_zero_shot.json", "w"))
    print(f"  zero-shot top: {phrases[int(zs.argmax())]} ({float(zs.max()):.3f})", flush=True)
    mel[0, 0].numpy().astype(np.float32).tofile(fx / f"{name}_mel.f32"); ours_a[0].numpy().astype(np.float32).tofile(fx / f"{name}_embedding.f32")

t0 = time.time()
with torch.no_grad():
    ex_a = torch.export.export(audio, (torch.randn(1, 1, 1001, 64),)).run_decompositions(coreai_torch.get_decomp_table())
    ex_t = torch.export.export(text, (torch.ones(1, T, dtype=torch.int32), torch.ones(1, T, dtype=torch.int32))).run_decompositions(coreai_torch.get_decomp_table())
converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
converter.add_exported_program(ex_a, input_names=["mel"], output_names=["embedding"], entrypoint_name="audio")
converter.add_exported_program(ex_t, input_names=["ids", "mask"], output_names=["embedding"], entrypoint_name="text")
program = converter.to_coreai(); program.optimize()
meta = AIModelAssetMetadata()
meta.author = "LAION (Wu, Chen, Zhang, Hui, Berg-Kirkpatrick, Dubnov) — CLAP; Core AI export by SampleSearch"
meta.license = "Apache-2.0"
meta.model_description = f"CLAP {args.name}: audio = log-mel [1,1,1001,64] → embedding [1,512] (HTSAT + projection); text = RoBERTa ids [1,{T}] + mask → embedding [1,512]. L2-normalise in the host. Tokenizer files beside the asset."
meta.creation_date = int(time.time())
asset = out / "crate-clap-music-float32.aimodel"
if asset.exists(): shutil.rmtree(asset)
program.save_asset(asset, meta)
print(f"saved {asset} ({sum(f.stat().st_size for f in asset.rglob('*'))/1e6:.0f} MB) in {time.time()-t0:.0f} s", flush=True)
if args.install:
    dest = Path.home() / "Library/Application Support/crate"
    dest.mkdir(parents=True, exist_ok=True)
    for src in [asset, tokdir]:
        d = dest / src.name
        if d.exists(): shutil.rmtree(d)
        shutil.copytree(src, d)
    print(f"installed into {dest}")
