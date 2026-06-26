"""
Standalone batch stem separation — loop over a folder of songs and write stems
for each. Runs entirely locally and free (Demucs on your CPU/GPU).

Examples
--------
  # every song in your Music folder -> a "Stems" folder, all stems:
  python batch_separate.py "C:\\Users\\chenm\\Music" --recursive

  # only vocals + drums, 6-stem model, each song in its own subfolder:
  python batch_separate.py "D:\\songs" -m htdemucs_6s -s vocals drums --per-song-folder

Re-running is safe: songs whose stems already exist are skipped, so you can stop
and resume. Output files are named "<song> - <stem>.wav".
"""
import os
import sys
import argparse
import traceback

import separate

AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".wma", ".mp4"}


def find_audio(root, recursive, skip_dir):
    skip_dir = os.path.abspath(skip_dir)
    if recursive:
        for d, _, files in os.walk(root):
            if os.path.abspath(d).startswith(skip_dir):   # never recurse into the output tree
                continue
            for f in files:
                if os.path.splitext(f)[1].lower() in AUDIO_EXT:
                    yield os.path.join(d, f)
    else:
        for f in sorted(os.listdir(root)):
            p = os.path.join(root, f)
            if os.path.isfile(p) and os.path.splitext(f)[1].lower() in AUDIO_EXT:
                yield p


def already_done(out_dir, song, stems):
    return all(os.path.exists(os.path.join(out_dir, f"{song} - {s}.wav")) for s in stems)


def main():
    ap = argparse.ArgumentParser(description="Batch-separate a folder of songs into stems.")
    ap.add_argument("input", help="Folder containing songs")
    ap.add_argument("-o", "--out", default=None,
                    help="Output folder (default: <input>\\Stems)")
    ap.add_argument("-m", "--model", default=separate.DEFAULT_MODEL,
                    choices=separate.available_models())
    ap.add_argument("-s", "--stems", nargs="*", default=None,
                    help="Subset of stems to write (default: all for the model)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--per-song-folder", action="store_true",
                    help="Put each song's stems in its own subfolder")
    ap.add_argument("--device", default=None, help="cpu / cuda (auto if omitted)")
    a = ap.parse_args()

    if not os.path.isdir(a.input):
        sys.exit(f"Not a folder: {a.input}")
    out_root = a.out or os.path.join(a.input, "Stems")
    stems = a.stems or separate.list_stems(a.model)

    files = list(find_audio(a.input, a.recursive, out_root))
    print(f"Found {len(files)} audio file(s). model={a.model} stems={stems}")
    print(f"Output -> {out_root}\n")

    ok = skip = err = 0
    for i, p in enumerate(files, 1):
        song = separate._safe(os.path.splitext(os.path.basename(p))[0])
        out_dir = os.path.join(out_root, song) if a.per_song_folder else out_root
        try:
            if already_done(out_dir, song, stems):
                skip += 1
                print(f"[{i}/{len(files)}] skip (already done): {song}")
                continue
            print(f"[{i}/{len(files)}] separating: {song}", flush=True)
            separate.separate(p, out_dir, a.model, stems, device=a.device)
            ok += 1
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            err += 1
            print(f"[{i}/{len(files)}] ERROR on {song}: {e}")
            traceback.print_exc()

    print(f"\nDONE  ok={ok}  skipped={skip}  errors={err}  ->  {out_root}")


if __name__ == "__main__":
    main()
