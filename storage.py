"""
Best-effort archiving of each run to a PRIVATE Hugging Face dataset, plus helpers
for the per-user account page. Uploads run in a background thread so they never
delay the user's download, and failures are swallowed — storage never breaks the app.

Layout
------
Anonymous runs (auto-deleted after 14 days by the cleanup workflow):
    runs/YYYY-MM-DD/<id>/  input.<ext>  <stem>.wav ...  meta.json

Logged-in users (kept forever — cleanup only touches runs/):
    users/<user_id>/songs/<song_id>/        song_id = sha1(audio) so re-uploads group
        input.<ext>                         the original upload (stored once per song)
        song.json                           title, original filename, uploaded_at
        runs/<run_id>/                      one per separation of this song
            <stem>.wav ...
            meta.json                       engine, mode, shifts, overlap, timestamp

Env: HF_TOKEN (WRITE access to the dataset), STORAGE_DATASET, STORE_RUNS=0 to disable.
"""
from __future__ import annotations
import os
import io
import re
import json
import shutil
import uuid
import hashlib
import datetime
import tempfile
import threading

DATASET = os.environ.get("STORAGE_DATASET", "vincewin/stem-worker-data")


def _token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")


def enabled() -> bool:
    return os.environ.get("STORE_RUNS", "1") != "0" and bool(_token())


def _safe_id(s) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(s))[:64] or "user"


def _hash_file(path, n=12) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def save_run_async(input_path, stem_paths, meta=None, user_id=None):
    """Fire-and-forget archive of one run. user_id -> permanent per-user storage."""
    if not enabled():
        return
    threading.Thread(target=_save_run,
                     args=(input_path, list(stem_paths or []), dict(meta or {}), user_id),
                     daemon=True).start()


def _save_run(input_path, stem_paths, meta, user_id=None):
    stage = None
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=_token())
        now = datetime.datetime.now()
        run_id = now.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        has_input = bool(input_path and os.path.exists(input_path))
        in_ext = (os.path.splitext(input_path)[1] if has_input else "") or ".bin"

        if user_id:
            uid = _safe_id(user_id)
            song_id = _hash_file(input_path) if has_input else uuid.uuid4().hex[:12]
            song_dir = f"users/{uid}/songs/{song_id}"
            # upload the original + song.json once per song
            if has_input and not api.file_exists(repo_id=DATASET, repo_type="dataset",
                                                 filename=f"{song_dir}/input{in_ext}"):
                api.upload_file(path_or_fileobj=input_path, repo_id=DATASET, repo_type="dataset",
                                path_in_repo=f"{song_dir}/input{in_ext}",
                                commit_message=f"song {song_id} input")
                info = {"title": meta.get("title") or os.path.splitext(os.path.basename(input_path))[0],
                        "original_filename": os.path.basename(input_path),
                        "uploaded_at": now.isoformat(), "song_id": song_id}
                api.upload_file(path_or_fileobj=io.BytesIO(json.dumps(info, indent=2).encode()),
                                repo_id=DATASET, repo_type="dataset",
                                path_in_repo=f"{song_dir}/song.json",
                                commit_message=f"song {song_id} meta")
            run_prefix = f"{song_dir}/runs/{run_id}"
        else:
            run_prefix = f"runs/{now.strftime('%Y-%m-%d')}/{run_id}"

        # stage the run folder (stems + meta, plus input for anonymous) and upload once
        stage = tempfile.mkdtemp(prefix="runsave_")
        for p in stem_paths:
            if p and os.path.exists(p):
                shutil.copy2(p, os.path.join(stage, os.path.basename(p)))
        if not user_id and has_input:
            shutil.copy2(input_path, os.path.join(stage, "input" + in_ext))
        meta = {**meta, "run_id": run_id, "timestamp": now.isoformat(),
                "n_stems": len([p for p in stem_paths if p and os.path.exists(p)])}
        with open(os.path.join(stage, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        api.upload_folder(folder_path=stage, repo_id=DATASET, repo_type="dataset",
                          path_in_repo=run_prefix, commit_message=f"run {run_id}")
    except Exception:
        pass
    finally:
        if stage and os.path.isdir(stage):
            shutil.rmtree(stage, ignore_errors=True)


# ---------------- account-page helpers ----------------
def list_user_songs(user_id) -> list[dict]:
    """[{song_id, title, uploaded_at, runs:[{run_id, stems:[...]}]}] newest run first."""
    if not user_id or not _token():
        return []
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi(token=_token())
        uid = _safe_id(user_id)
        prefix = f"users/{uid}/songs/"
        files = [f for f in api.list_repo_files(repo_id=DATASET, repo_type="dataset")
                 if f.startswith(prefix)]
        songs: dict[str, dict] = {}
        for f in files:
            parts = f[len(prefix):].split("/")
            sid = parts[0]
            s = songs.setdefault(sid, {"song_id": sid, "title": sid, "uploaded_at": "", "runs": {}})
            if len(parts) >= 4 and parts[1] == "runs" and parts[3].endswith(".wav"):
                s["runs"].setdefault(parts[2], {"run_id": parts[2], "stems": []})["stems"].append(parts[3])
        # best-effort: read each song.json for a friendly title (cap to keep it snappy)
        for sid, s in list(songs.items())[:60]:
            try:
                p = hf_hub_download(repo_id=DATASET, repo_type="dataset",
                                    filename=f"{prefix}{sid}/song.json")
                with open(p, encoding="utf-8") as fh:
                    info = json.load(fh)
                s["title"] = info.get("title", sid)
                s["uploaded_at"] = info.get("uploaded_at", "")
            except Exception:
                pass
        out = []
        for s in songs.values():
            s["runs"] = sorted(s["runs"].values(), key=lambda r: r["run_id"], reverse=True)
            out.append(s)
        out.sort(key=lambda s: s.get("uploaded_at", ""), reverse=True)
        return out
    except Exception:
        return []


def fetch_run_stems(user_id, song_id, run_id) -> list[str]:
    """Download one run's stem files locally and return their paths (for the UI)."""
    if not (user_id and song_id and run_id and _token()):
        return []
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi(token=_token())
        uid = _safe_id(user_id)
        run_prefix = f"users/{uid}/songs/{song_id}/runs/{run_id}/"
        paths = []
        for f in api.list_repo_files(repo_id=DATASET, repo_type="dataset"):
            if f.startswith(run_prefix) and f.endswith(".wav"):
                paths.append(hf_hub_download(repo_id=DATASET, repo_type="dataset", filename=f))
        return paths
    except Exception:
        return []
