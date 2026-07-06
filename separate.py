"""
Stem separation core (Demucs) with optional cascades.

  - base 4-stem / 6-stem separation  (official Meta models, trusted)
  - drum split: the 'drums' stem -> kick / snare / toms / cymbals
      (community 'drumsep' Hybrid-Demucs model; loaded with the full unpickler,
       which the user explicitly authorized)
  - vocal split: the 'vocals' stem -> lead / backing   (added on top of drums)

Files are copied, never modified; analysis runs only on copies.
"""
from __future__ import annotations
import os
import re
from typing import Iterable

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ZeroGPU: @spaces.GPU marks the GPU entry point so HF attaches a GPU for that
# call only. Off Spaces (local / CPU), `spaces` isn't installed -> no-op decorator.
try:
    import spaces
    _gpu = spaces.GPU(duration=300)
except Exception:
    def _gpu(fn):
        return fn

# ---- base models (official Demucs) ----
MODEL_STEMS = {
    "htdemucs":    ["drums", "bass", "other", "vocals"],
    "htdemucs_ft": ["drums", "bass", "other", "vocals"],
    "htdemucs_6s": ["drums", "bass", "other", "vocals", "guitar", "piano"],
    "mdx_extra":   ["drums", "bass", "other", "vocals"],
}
DEFAULT_MODEL = "htdemucs"

# ---- drum-split (community 'drumsep' model) ----
DRUM_REPO  = os.path.join(HERE, "models")
DRUM_MODEL = "49469ca8"
DRUM_HF    = "vincewin/drumsep"      # mirror the Space downloads from
DRUM_MAP   = {"bombo": "kick", "redoblante": "snare", "platillos": "cymbals", "toms": "toms"}
DRUM_ORDER = ["kick", "snare", "toms", "cymbals"]

# ---- RoFormer engine (newer architecture, via the 'audio-separator' package) ----
# BS-RoFormer vocals model: cleaner vocals/instrumental split than Demucs, 2 stems.
ROFORMER_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"

# Formats every engine decodes natively (libsndfile-backed readers). Anything
# else — AAC / ALAC inside .m4a, raw .aac, .opus, .wma, .caf, video containers —
# is transcoded to WAV with ffmpeg first, so uploads from phones (Apple formats)
# work with every engine, including RoFormer which doesn't use ffmpeg itself.
NATIVE_EXTS = {".wav", ".aiff", ".aif", ".aifc", ".flac", ".mp3", ".ogg"}


def prepare_input(input_path: str) -> str:
    """Return a path every engine can read, transcoding to WAV when needed.

    The original file is never touched (the WAV copy goes to a temp dir) and
    the song name is preserved so stems stay labeled '<song> - <stem>.wav'.
    Falls back to the original path if ffmpeg is missing or conversion fails.
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext in NATIVE_EXTS:
        return input_path
    import shutil
    import subprocess
    import tempfile
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return input_path
    song = _safe(os.path.splitext(os.path.basename(input_path))[0])
    out = os.path.join(tempfile.mkdtemp(prefix="input_"), song + ".wav")
    try:
        subprocess.run([ffmpeg, "-y", "-i", input_path, "-vn", "-acodec", "pcm_s16le", out],
                       check=True, capture_output=True)
        return out
    except Exception:
        return input_path

# Engines selectable in the app: label-friendly quality tiers.
ENGINES = ("demucs", "demucs_ft", "roformer")

# ---- modes offered in the app: how many stems and how to reach them ----
MODES = {
    "4": {"base": "htdemucs",    "drums": False, "vocals": False},
    "6": {"base": "htdemucs_6s", "drums": False, "vocals": False},
    "9": {"base": "htdemucs_6s", "drums": True,  "vocals": False},
}
DEFAULT_MODE = "4"


def available_models() -> list[str]:
    return list(MODEL_STEMS.keys())


def list_stems(model_name: str = DEFAULT_MODEL) -> list[str]:
    return MODEL_STEMS.get(model_name, MODEL_STEMS[DEFAULT_MODEL])


def mode_stems(mode: str) -> list[str]:
    """Final stem names a mode produces (for the checkbox UI), in display order."""
    cfg = MODES[mode]
    out: list[str] = []
    for s in MODEL_STEMS[cfg["base"]]:
        if s == "drums" and cfg["drums"]:
            out += DRUM_ORDER
        elif s == "vocals" and cfg["vocals"]:
            out += ["lead vocal", "backing vocal"]
        else:
            out.append(s)
    return out


def _safe(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return name or "track"


def _pick_device(device: str | None) -> str:
    if device:
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ---------------- model loading (cached) ----------------
_CACHE: dict = {}
_PATCHED = False


def _full_unpickle():
    """Allow torch.load to deserialize the community drumsep checkpoint.

    PyTorch >=2.6 defaults to weights_only=True, which refuses pickled model
    objects. The user explicitly authorized loading this third-party model.
    """
    global _PATCHED
    if _PATCHED:
        return
    import torch
    _orig = torch.load
    torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
    _PATCHED = True


def _ensure_drum_repo() -> str:
    path = os.path.join(DRUM_REPO, DRUM_MODEL + ".th")
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        os.makedirs(DRUM_REPO, exist_ok=True)
        hf_hub_download(repo_id=DRUM_HF, filename=DRUM_MODEL + ".th", local_dir=DRUM_REPO)
    return DRUM_REPO


def _load(name: str, repo: str | None = None):
    key = (name, repo)
    if key not in _CACHE:
        _full_unpickle()
        from pathlib import Path
        from demucs.pretrained import get_model
        m = get_model(name=name, repo=Path(repo) if repo else None)
        m.eval()
        _CACHE[key] = m
    return _CACHE[key]


def _read_audio(path: str, sr: int, ch: int):
    from demucs.audio import AudioFile
    return AudioFile(path).read(streams=0, samplerate=sr, channels=ch)


def _run(model, wav, device, shifts=0, overlap=0.25):
    """Normalize, run apply_model, de-normalize. wav: tensor [channels, samples].

    shifts > 0 enables Demucs's "shift trick" (test-time augmentation): the input
    is shifted a few times and the results averaged, which reduces artifacts at a
    roughly linear cost in time. overlap controls window blending at chunk edges.
    """
    import torch
    from demucs.apply import apply_model
    ref = wav.mean(0)
    x = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        out = apply_model(model, x[None].to(device), shifts=int(shifts), split=True,
                          overlap=overlap, progress=False, device=device)[0]
    return out * (ref.std() + 1e-8) + ref.mean()


def _write(tensor, out_path: str, sr: int):
    import soundfile as sf
    sf.write(out_path, np.clip(tensor.cpu().numpy().T, -1.0, 1.0), sr, subtype="PCM_16")


# ---------------- public API ----------------
def separate(input_path, out_dir, model_name=DEFAULT_MODEL, stems=None,
             device=None, progress=None, shifts=0, overlap=0.25) -> list[str]:
    """Base (single-model) separation. Used by the CLI and batch tool."""
    device = _pick_device(device)
    if progress:
        progress(0.05, f"Loading model '{model_name}' on {device}...")
    model = _load(model_name)
    sources = list(model.sources)
    wanted = [s for s in (list(stems) if stems else sources) if s in sources]
    if not wanted:
        raise ValueError(f"No valid stems for {model_name}. Choose from: {sources}")
    if progress:
        progress(0.1, "Reading audio...")
    wav = _read_audio(input_path, model.samplerate, model.audio_channels)
    if progress:
        progress(0.2, "Separating audio (slow on CPU)...")
    out = _run(model, wav, device, shifts=shifts, overlap=overlap)
    os.makedirs(out_dir, exist_ok=True)
    song = _safe(os.path.splitext(os.path.basename(input_path))[0])
    written = []
    for i, name in enumerate(sources):
        if name not in wanted:
            continue
        p = os.path.join(out_dir, f"{song} - {name}.wav")
        _write(out[i], p, model.samplerate)
        written.append(p)
    if progress:
        progress(1.0, "Done")
    return written


def separate_mode(input_path, out_dir, mode=DEFAULT_MODE, stems=None,
                  device=None, progress=None, base_override=None,
                  shifts=0, overlap=0.25) -> list[str]:
    """Cascaded separation by mode ('4', '6', '9'). Writes '<song> - <stem>.wav'.

    base_override swaps the mode's base model (e.g. the fine-tuned 'htdemucs_ft'
    for cleaner 4-stem output). shifts/overlap tune separation quality vs. speed.
    """
    cfg = MODES[mode]
    device = _pick_device(device)
    if progress:
        progress(0.05, "Loading model...")
    base = _load(base_override or cfg["base"])
    sr = base.samplerate
    if progress:
        progress(0.1, "Reading audio...")
    wav = _read_audio(input_path, sr, base.audio_channels)
    if progress:
        progress(0.15, "Separating main stems (slow on CPU)...")
    out = _run(base, wav, device, shifts=shifts, overlap=overlap)
    result = {name: out[i] for i, name in enumerate(base.sources)}

    if cfg["drums"] and "drums" in result:
        if progress:
            progress(0.6, "Splitting drums into kick / snare / toms / cymbals...")
        dm = _load(DRUM_MODEL, repo=_ensure_drum_repo())
        parts = _run(dm, result.pop("drums"), device, shifts=shifts, overlap=overlap)
        for i, src in enumerate(dm.sources):
            result[DRUM_MAP.get(src, src)] = parts[i]

    os.makedirs(out_dir, exist_ok=True)
    song = _safe(os.path.splitext(os.path.basename(input_path))[0])
    order = mode_stems(mode)
    wanted = [s for s in (list(stems) if stems else order) if s in result]
    written = []
    for name in order:
        if name not in wanted:
            continue
        p = os.path.join(out_dir, f"{song} - {name}.wav")
        _write(result[name], p, sr)
        written.append(p)
    if progress:
        progress(1.0, "Done")
    return written


def separate_roformer(input_path, out_dir, progress=None) -> list[str]:
    """High-quality vocals/instrumental split using a BS-RoFormer model.

    Uses the 'audio-separator' package, which downloads the model on first use.
    Produces two stems; files are renamed to the '<song> - <stem>.wav' convention.
    """
    from audio_separator.separator import Separator
    os.makedirs(out_dir, exist_ok=True)
    if progress:
        progress(0.1, "Loading RoFormer model (first run downloads it ~1 min)...")
    sep = Separator(output_dir=out_dir, output_format="WAV")
    sep.load_model(model_filename=ROFORMER_MODEL)
    if progress:
        progress(0.3, "Separating with RoFormer (slow on CPU)...")
    produced = sep.separate(input_path)  # list of output file names (in out_dir)
    song = _safe(os.path.splitext(os.path.basename(input_path))[0])
    written = []
    for fn in produced:
        src = fn if os.path.isabs(fn) else os.path.join(out_dir, fn)
        low = os.path.basename(fn).lower()
        stem = "vocals" if "vocal" in low else ("instrumental" if "instrument" in low else _safe(os.path.splitext(os.path.basename(fn))[0]))
        dst = os.path.join(out_dir, f"{song} - {stem}.wav")
        if os.path.abspath(src) != os.path.abspath(dst):
            os.replace(src, dst)
        written.append(dst)
    if progress:
        progress(1.0, "Done")
    return written


def run_separation(input_path, out_dir, engine="demucs", mode=DEFAULT_MODE,
                   stems=None, device=None, progress=None, shifts=0, overlap=0.25) -> list[str]:
    """Single entry point the app uses; dispatches on the chosen engine.

    - 'demucs'    : the mode's standard model (4/6/9-stem), with optional shifts.
    - 'demucs_ft' : fine-tuned model for 4-stem (falls back to standard for 6/9),
                    cleaner but ~4x slower.
    - 'roformer'  : BS-RoFormer (newest architecture) — best vocals/instrumental,
                    always 2 stems; mode/stem selection is ignored.

    shifts/overlap are quality knobs (higher = cleaner, slower).
    """
    input_path = prepare_input(input_path)
    if engine == "roformer":
        return separate_roformer(input_path, out_dir, progress=progress)
    base_override = None
    if engine == "demucs_ft" and MODES[mode]["base"] == "htdemucs":
        base_override = "htdemucs_ft"
    return separate_mode(input_path, out_dir, mode, stems, device=device,
                         progress=progress, base_override=base_override,
                         shifts=shifts, overlap=overlap)


@_gpu
def gpu_separate(input_path, out_dir, engine="demucs", mode=DEFAULT_MODE,
                 stems=None, shifts=0, overlap=0.25) -> list[str]:
    """GPU entry point for ZeroGPU. Takes only picklable args (no progress callback,
    which can't cross the GPU process boundary). Device auto-selects cuda when present."""
    return run_separation(input_path, out_dir, engine=engine, mode=mode,
                          stems=stems, device=None, shifts=shifts, overlap=overlap,
                          progress=None)


def merge_stems(paths, out_path) -> str:
    """Mix several stem .wav files back into one track by summing their samples.

    Demucs stems are additive (they sum to the original), so combining a subset
    is just a sample-wise sum. Inputs are aligned on length/channels defensively
    in case the caller mixes files from different sources.
    """
    import soundfile as sf
    paths = list(paths)
    if len(paths) < 2:
        raise ValueError("Select at least two stems to merge.")
    mix = None
    sr = None
    for p in paths:
        data, file_sr = sf.read(p, always_2d=True)  # shape [samples, channels]
        data = data.astype(np.float64)
        if mix is None:
            mix, sr = data, file_sr
            continue
        if file_sr != sr:
            raise ValueError("Stems have different sample rates; cannot merge.")
        if data.shape[1] != mix.shape[1]:          # mono vs stereo -> upmix mono
            wide = max(data.shape[1], mix.shape[1])
            if data.shape[1] == 1:
                data = np.repeat(data, wide, axis=1)
            if mix.shape[1] == 1:
                mix = np.repeat(mix, wide, axis=1)
        if data.shape[0] != mix.shape[0]:          # trim to the shorter one
            n = min(data.shape[0], mix.shape[0])
            mix, data = mix[:n], data[:n]
        mix = mix + data
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sf.write(out_path, np.clip(mix, -1.0, 1.0), sr, subtype="PCM_16")
    return out_path


def stem_of(path: str) -> str:
    """Recover the stem label from a '<song> - <stem>.wav' filename."""
    base = os.path.splitext(os.path.basename(path))[0]
    return base.split(" - ")[-1] if " - " in base else base


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Separate an audio file into stems.")
    ap.add_argument("input")
    ap.add_argument("-o", "--out", default="stems_out")
    ap.add_argument("--mode", default=None, choices=list(MODES.keys()),
                    help="4 / 6 / 9 stems (cascaded). Overrides --model.")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL, choices=available_models())
    ap.add_argument("-s", "--stems", nargs="*", default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    pr = lambda f, m: print(f"[{f*100:5.1f}%] {m}")
    if a.mode:
        paths = separate_mode(a.input, a.out, a.mode, a.stems, a.device, progress=pr)
    else:
        paths = separate(a.input, a.out, a.model, a.stems, a.device, progress=pr)
    print("\nWrote:")
    for p in paths:
        print("  ", p)
