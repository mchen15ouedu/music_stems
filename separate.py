"""
Stem separation core, built on Demucs (Meta, MIT-licensed, free pretrained models).

Nothing here needs training: htdemucs / htdemucs_6s are pretrained and downloaded
once (~80 MB each) into the torch hub cache on first use.
"""
from __future__ import annotations
import os
import re
from typing import Callable, Iterable

# Stems each model produces. Demucs separates ALL stems in a single forward pass,
# so "pick a subset" just means we save the chosen ones — the compute is identical.
MODEL_STEMS = {
    "htdemucs":     ["drums", "bass", "other", "vocals"],            # default, best quality
    "htdemucs_ft":  ["drums", "bass", "other", "vocals"],            # fine-tuned, ~4x slower
    "htdemucs_6s":  ["drums", "bass", "other", "vocals", "guitar", "piano"],  # 6 stems
    "mdx_extra":    ["drums", "bass", "other", "vocals"],
}
DEFAULT_MODEL = "htdemucs"


def available_models() -> list[str]:
    return list(MODEL_STEMS.keys())


def list_stems(model_name: str = DEFAULT_MODEL) -> list[str]:
    """Stems a given model can produce."""
    return MODEL_STEMS.get(model_name, MODEL_STEMS[DEFAULT_MODEL])


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


def separate(
    input_path: str,
    out_dir: str,
    model_name: str = DEFAULT_MODEL,
    stems: Iterable[str] | None = None,
    device: str | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> list[str]:
    """
    Separate `input_path` and write the requested `stems` as WAV files named
    "<song> - <stem>.wav" into `out_dir`. Returns the list of written paths.

    `stems=None` writes every stem the model produces. Uses demucs's stable
    lower-level API (get_model / apply_model / save_audio).
    """
    import torch
    import numpy as np
    import soundfile as sf
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from demucs.audio import AudioFile

    device = _pick_device(device)
    if progress:
        progress(0.05, f"Loading model '{model_name}' on {device}...")
    model = get_model(model_name)
    model.to(device)
    model.eval()

    sources = list(model.sources)  # true stem names/order, e.g. drums,bass,other,vocals
    wanted = [s for s in (list(stems) if stems else sources) if s in sources]
    if not wanted:
        raise ValueError(f"No valid stems requested for model {model_name}. "
                         f"Choose from: {sources}")

    if progress:
        progress(0.1, "Reading audio...")
    wav = AudioFile(input_path).read(
        streams=0, samplerate=model.samplerate, channels=model.audio_channels)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)  # normalize as demucs.separate does

    if progress:
        progress(0.2, "Separating audio (this is the slow part on CPU)...")
    with torch.no_grad():
        out = apply_model(model, wav[None].to(device),
                          split=True, overlap=0.25, progress=True, device=device)[0]
    out = out * (ref.std() + 1e-8) + ref.mean()  # de-normalize

    os.makedirs(out_dir, exist_ok=True)
    song = _safe(os.path.splitext(os.path.basename(input_path))[0])
    written: list[str] = []
    for i, name in enumerate(sources):
        if name not in wanted:
            continue
        out_path = os.path.join(out_dir, f"{song} - {name}.wav")
        # [channels, samples] -> [samples, channels], clamp, write 16-bit PCM WAV
        audio = np.clip(out[i].cpu().numpy().T, -1.0, 1.0)
        sf.write(out_path, audio, model.samplerate, subtype="PCM_16")
        written.append(out_path)
        if progress:
            progress(0.8 + 0.2 * (i + 1) / len(sources), f"Saved {name}")
    if progress:
        progress(1.0, "Done")
    return written


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Separate an audio file into stems with Demucs.")
    ap.add_argument("input", help="Path to an audio file (mp3/wav/flac/m4a/...)")
    ap.add_argument("-o", "--out", default="stems_out", help="Output folder")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL, choices=available_models())
    ap.add_argument("-s", "--stems", nargs="*", default=None,
                    help="Subset of stems to write (default: all)")
    ap.add_argument("--device", default=None, help="cpu / cuda (auto if omitted)")
    a = ap.parse_args()
    paths = separate(a.input, a.out, a.model, a.stems, a.device,
                     progress=lambda f, m: print(f"[{f*100:5.1f}%] {m}"))
    print("\nWrote:")
    for p in paths:
        print("  ", p)
