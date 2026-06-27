"""
Push this folder to a Hugging Face Space.

The token is read from the HF_TOKEN environment variable (never hardcoded).
Usage:
    set HF_TOKEN=<token>           # PowerShell: $env:HF_TOKEN = (...)
    set HF_SPACE=vincewin/stem_worker
    python deploy_space.py
"""
import os
from huggingface_hub import HfApi

REPO = os.environ.get("HF_SPACE", "vincewin/stem_worker")
token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("HF_TOKEN env var not set.")

api = HfApi(token=token)
here = os.path.dirname(os.path.abspath(__file__))
api.upload_folder(
    folder_path=here,
    repo_id=REPO,
    repo_type="space",
    commit_message="Deploy AI stem splitter",
    ignore_patterns=[
        "*.log", "*.log.err", "pip_pid.txt", "pip_install.*",
        "__pycache__/*", "*.pyc", "stems_*/*", "merge_*/*", "*.wav",
        ".git/*", ".venv/*", "deploy_space.py",
        # the ~160 MB drum model is downloaded at runtime from vincewin/drumsep
        "models/*", "models",
        "_*.py",   # scratch/test scripts
    ],
)
print(f"Uploaded -> https://huggingface.co/spaces/{REPO}")
