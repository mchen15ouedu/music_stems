"""
Best-effort archiving of each run (uploaded song + generated stems) to a PRIVATE
Hugging Face dataset. Uploads run in a background thread so they never delay the
user's download, and any failure is swallowed — storage must never break the app.

Layout in the dataset (date-prefixed so the bi-weekly cleanup can delete by age):
    runs/YYYY-MM-DD/HHMMSS-<id>/input.<ext>
                               /<song> - <stem>.wav ...
                               /meta.json

Env:
    HF_TOKEN          must have WRITE access to the dataset (not just inference).
    STORAGE_DATASET   override the dataset id (default vincewin/stem-worker-data).
    STORE_RUNS=0      disable archiving entirely.
"""
from __future__ import annotations
import os
import io
import json
import shutil
import uuid
import datetime
import threading

DATASET = os.environ.get("STORAGE_DATASET", "vincewin/stem-worker-data")


def _token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")


def enabled() -> bool:
    return os.environ.get("STORE_RUNS", "1") != "0" and bool(_token())


def save_run_async(input_path, stem_paths, meta=None):
    """Fire-and-forget: archive one run's input + stems to the private dataset."""
    if not enabled():
        return
    t = threading.Thread(target=_save_run,
                         args=(input_path, list(stem_paths or []), dict(meta or {})),
                         daemon=True)
    t.start()


def _save_run(input_path, stem_paths, meta):
    stage = None
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=_token())
        now = datetime.datetime.now()
        run_id = now.strftime("%H%M%S-") + uuid.uuid4().hex[:6]
        prefix = f"runs/{now.strftime('%Y-%m-%d')}/{run_id}"

        import tempfile
        stage = tempfile.mkdtemp(prefix="runsave_")
        if input_path and os.path.exists(input_path):
            ext = os.path.splitext(input_path)[1] or ".bin"
            shutil.copy2(input_path, os.path.join(stage, "input" + ext))
        for p in stem_paths:
            if p and os.path.exists(p):
                shutil.copy2(p, os.path.join(stage, os.path.basename(p)))
        meta = {**meta, "run_id": run_id, "timestamp": now.isoformat(),
                "n_stems": len([p for p in stem_paths if p and os.path.exists(p)])}
        with open(os.path.join(stage, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

        api.upload_folder(folder_path=stage, repo_id=DATASET, repo_type="dataset",
                          path_in_repo=prefix, commit_message=f"run {run_id}")
    except Exception:
        pass  # archiving is best-effort; never surface errors to the app
    finally:
        if stage and os.path.isdir(stage):
            shutil.rmtree(stage, ignore_errors=True)
