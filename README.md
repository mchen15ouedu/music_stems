---
title: AI Stem Splitter
emoji: 🎵
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
pinned: false
license: mit
hf_oauth: true
hf_oauth_expiration_minutes: 43200
---

# 🎵 AI Stem Splitter

Upload a song → pick the **stems** you want → download each as its own `.wav`.
Built on **[Demucs](https://github.com/facebookresearch/demucs)** (Meta, MIT-licensed) —
free, pretrained models, **no training required**. An LLM assistant helps you choose
which stems to generate for your goal.

## Stems

| Model | Stems |
|-------|-------|
| `htdemucs` (default) | vocals · drums · bass · other |
| `htdemucs_6s` | vocals · drums · bass · guitar · piano · other |

Demucs separates all stems in one pass; choosing a subset just selects which files to save.
Output files are named **`<song> - <stem>.wav`**.

## Run locally

```bash
conda create -n stems -c conda-forge python=3.11 ffmpeg -y
conda activate stems
pip install -r requirements.txt
python app.py                      # launches the Gradio UI
# or, headless:
python separate.py "song.mp3" -m htdemucs -s vocals drums
```

## LLM assistant — backends

Set `STEM_LLM_BACKEND`:

| Value | Use | Notes |
|-------|-----|-------|
| `hf` (default) | HF Inference API | Free with a token. On Spaces the `HF_TOKEN` secret is auto-injected. Set `STEM_LLM_MODEL` to choose the model. |
| `none` | No LLM | Deterministic rule-based suggestions; always works offline. |
| `vllm` | Self-hosted vLLM (**GPU only**) | Set `VLLM_BASE_URL` (OpenAI-compatible). Only worth it on a GPU Space. |
| `openai` | Any OpenAI-compatible API | `OPENAI_BASE_URL` / `OPENAI_API_KEY`. |

> **About vLLM:** it accelerates *language models*, not the audio model — Demucs runs on
> plain PyTorch and is unaffected by vLLM. vLLM is also GPU-only, so it only makes sense if
> you self-host the assistant LLM on a paid GPU Space. The default `hf` backend keeps the
> Space free on CPU.

## Hugging Face Space

This repo is Space-ready (`app.py` + `requirements.txt` + this header). Push it to a Gradio
Space. **CPU basic** is free; a 3–4 min song takes ~1–3 min to separate. Optional GPU
hardware speeds it up (and is the only place the `vllm` backend applies).
