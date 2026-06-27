"""
LLM helper that facilitates the stem-separation workflow: it explains stems and
recommends which ones to generate for a user's goal (karaoke, remix, practice...).

Pluggable backend (set env var STEM_LLM_BACKEND):
  - "hf"     (default) : Hugging Face Inference API via huggingface_hub.InferenceClient.
                         Free with an HF token; on HF Spaces the token is auto-injected
                         as the HF_TOKEN secret. Set STEM_LLM_MODEL to choose the model.
  - "vllm"            : OpenAI-compatible endpoint served by vLLM. ONLY makes sense on a
                         GPU deployment. Set VLLM_BASE_URL (e.g. http://localhost:8000/v1)
                         and STEM_LLM_MODEL to the served model id.
  - "openai"          : any OpenAI-compatible API (OPENAI_BASE_URL / OPENAI_API_KEY).
  - "none"            : no LLM call; deterministic rule-based replies (always available).

If a backend is configured but errors at runtime, we fall back to the rule-based
assistant so the UI never breaks.
"""
from __future__ import annotations
import os

BACKEND = os.environ.get("STEM_LLM_BACKEND", "hf").lower()
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # ungated + served by HF Inference Providers
MODEL = os.environ.get("STEM_LLM_MODEL", DEFAULT_HF_MODEL)

STEM_GLOSSARY = {
    "vocals": "the lead and backing voices",
    "drums": "the full drum kit / percussion",
    "bass": "the bassline (bass guitar / synth bass)",
    "other": "everything else — usually guitars, keys, synths, strings",
    "guitar": "guitar parts (htdemucs_6s only)",
    "piano": "piano parts (htdemucs_6s only)",
}

# Common goals -> which stems to KEEP (used by the rule-based fallback and to seed the LLM)
GOAL_RECIPES = {
    "karaoke": (["drums", "bass", "other"], "Remove vocals; keep the instrumental backing."),
    "instrumental": (["drums", "bass", "other"], "Everything except the vocals."),
    "acapella": (["vocals"], "Just the isolated vocals."),
    "vocals only": (["vocals"], "Just the isolated vocals."),
    "drumless": (["vocals", "bass", "other"], "Backing track with no drums — great for drummers to play along."),
    "bassless": (["vocals", "drums", "other"], "Backing track with no bass — for bass practice."),
    "drums only": (["drums"], "Isolated drums for sampling or transcription."),
    "bass only": (["bass"], "Isolated bass for transcription/practice."),
    "remix": (["vocals", "drums", "bass", "other"], "All stems, so you can rebalance and remix."),
    "stems": (["vocals", "drums", "bass", "other"], "All available stems."),
}


# Plain-language walkthrough of the app, shared by the LLM prompt and the
# rule-based fallback so the assistant can always explain "what do I do?".
APP_STEPS = [
    "Upload your song with the file box on the left (mp3, wav, flac, m4a...).",
    "Pick how many stems you want (4, 6, or 9). More stems take longer.",
    "Tick the stems you want to keep, then press 'Separate selected stems'.",
    "When it finishes, download each stem from the 'Download stems' box.",
    "Optional: in the 'Merge stems' section, tick 2-3 of the new stems and press "
    "'Merge selected into one file' to mix them back into a single track "
    "(e.g. drums + bass for a rhythm backing).",
]


def how_to_use() -> str:
    steps = "\n".join(f"{i}. {s}" for i, s in enumerate(APP_STEPS, 1))
    return "Here's how to use the app:\n\n" + steps


def _system_prompt(model_name: str, available: list[str], song: str | None) -> str:
    gloss = "\n".join(f"  - {s}: {STEM_GLOSSARY.get(s, s)}" for s in available)
    ctx = f"\nThe user loaded: \"{song}\"." if song else ""
    steps = "\n".join(f"  {i}. {s}" for i, s in enumerate(APP_STEPS, 1))
    return (
        "You are a friendly audio-engineering assistant embedded in a stem-separation app. "
        "The app uses the Demucs model to split a song into separate audio stems. "
        f"The selected model ('{model_name}') can produce these stems:\n{gloss}\n\n"
        "If the user is unsure what to do or asks how to use the app, walk them through "
        f"these steps in plain language:\n{steps}\n\n"
        "Otherwise, help the user decide which stems to generate for their goal (karaoke, "
        "remix, instrument practice, transcription, sampling), and explain which stems to "
        "merge if they want a custom backing track. When you recommend stems, name them "
        "exactly as listed above and keep answers short and practical."
        + ctx
    )


def recommend_stems(goal: str, available: list[str]) -> tuple[list[str], str]:
    """Deterministic mapping from a free-text goal to a stem selection (no LLM needed)."""
    g = (goal or "").lower()
    for key, (keep, why) in GOAL_RECIPES.items():
        if key in g:
            picks = [s for s in keep if s in available]
            if picks:
                return picks, why
    # keyword fallback: pick any stem name mentioned directly
    picks = [s for s in available if s in g]
    if picks:
        return picks, "Matched the stems you named."
    return list(available), "Not sure of your goal — defaulting to all stems."


_HELP_HINTS = ("how do i", "how to", "what do i do", "how does this", "get started",
               "instructions", "help", "where do i", "confused", "not sure what")


# --- quality-improvement loop ------------------------------------------------
# When a split looks dirty, the assistant escalates the engine/settings one rung
# up this ladder and re-runs. The decision is deterministic so it always changes
# something and works even with no LLM backend.
IMPROVE_HINTS = ("improve", "better", "cleaner", "clean it", "not clean", "not good",
                 "bad", "worse", "bleed", "bleeding", "noisy", "noise", "muddy",
                 "muffled", "artifact", "redo", "re-run", "rerun", "try again",
                 "fix", "leaking", "leak", "still hear")


def wants_improvement(message: str) -> bool:
    m = (message or "").lower()
    return any(h in m for h in IMPROVE_HINTS)


def improve_settings(complaint: str, engine: str, shifts: int,
                     overlap: float = 0.25) -> tuple[str, int, float, str]:
    """Deterministic escalation ladder -> (engine, shifts, overlap, why).

    Climbs the cheapest-first quality ladder: shifts -> overlap -> fine-tuned
    model -> RoFormer. Always changes something so a re-run is meaningful, and
    works with no LLM backend.
    """
    c = (complaint or "").lower()
    shifts = int(shifts)
    overlap = float(overlap)
    vocal_focus = any(w in c for w in ("vocal", "voice", "sing", "acapella", "karaoke"))
    if engine == "demucs":
        if shifts < 2:
            return ("demucs", 2, overlap,
                    "Turned on the **shift trick (shifts=2)** — averages a few passes to "
                    "cut artifacts. Re-running (~2× slower).")
        if overlap < 0.5:
            return ("demucs", shifts, 0.5,
                    "Raised **overlap to 0.5** to smooth the chunk-edge artifacts. Re-running.")
        return ("demucs_ft", max(shifts, 2), overlap,
                "Switched to the **fine-tuned model (Demucs FT)** — cleaner separation, "
                "~4× slower. Re-running.")
    if engine == "demucs_ft":
        if overlap < 0.5:
            return ("demucs_ft", shifts, 0.5,
                    "Raised **overlap to 0.5** on the fine-tuned model for smoother output. "
                    "Re-running.")
        extra = " RoFormer is especially strong on vocals." if vocal_focus else ""
        return ("roformer", shifts, overlap,
                "Switched to **RoFormer**, the cleanest engine (vocals/instrumental)."
                + extra + " Re-running.")
    # already on roformer
    if overlap < 0.5:
        return ("roformer", shifts, 0.5,
                "Raised RoFormer **overlap to 0.5** for a smoother result. Re-running.")
    return ("roformer", shifts, overlap,
            "You're already on **RoFormer** at high overlap — the top-quality setting. "
            "If it's still not clean, the source mix may be the limit (try a different "
            "section). Re-running once more.")


def _raw_completion(system_prompt: str, user_message: str, max_tokens: int = 200) -> str:
    """One-shot LLM completion via the configured backend. Raises on failure."""
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}]
    if BACKEND in ("vllm", "openai"):
        from openai import OpenAI
        base = os.environ.get("VLLM_BASE_URL" if BACKEND == "vllm" else "OPENAI_BASE_URL")
        key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        client = OpenAI(base_url=base, api_key=key)
        resp = client.chat.completions.create(model=MODEL, messages=messages,
                                              max_tokens=max_tokens, temperature=0.1)
        return resp.choices[0].message.content
    from huggingface_hub import InferenceClient
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    client = InferenceClient(model=MODEL, token=token)
    resp = client.chat_completion(messages=messages, max_tokens=max_tokens, temperature=0.1)
    return resp.choices[0].message.content


def improve_plan(complaint: str, engine: str, shifts: int, overlap: float,
                 available: list[str] | None = None) -> tuple[str, int, float, str]:
    """LLM-driven version: let the model analyze the complaint and choose the knobs.

    Returns (engine, shifts, overlap, why). Falls back to the deterministic ladder
    when there's no LLM backend/token or the call fails — so it always does something.
    """
    fallback = improve_settings(complaint, engine, shifts, overlap)
    if BACKEND == "none":
        return fallback
    try:
        import json
        sys_prompt = (
            "You tune an audio source-separation app to fix a user's complaint about stem "
            "quality. Choose the SINGLE best settings change and explain it in one sentence.\n"
            "Engines: 'demucs' (fast), 'demucs_ft' (cleaner, ~4x slower), "
            "'roformer' (newest, best for vocals/instrumental but only 2 stems).\n"
            "Knobs: shifts (int 0-10, higher=cleaner+slower), overlap (float 0.25-0.9, "
            "higher=smoother edges+slower).\n"
            "Guidance: vocal bleed/voice issues -> roformer; choppy/clicky/edge artifacts -> "
            "raise overlap; general noise/muddiness -> raise shifts then demucs_ft. Only pick "
            "roformer if the user cares about vocals or instrumental, since it drops to 2 stems.\n"
            f"Current: engine={engine}, shifts={int(shifts)}, overlap={float(overlap)}.\n"
            'Respond with ONLY JSON: {"engine":"...","shifts":0,"overlap":0.25,"why":"..."}'
        )
        raw = _raw_completion(sys_prompt, complaint or "Make it cleaner.")
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start:end + 1])
        eng = data.get("engine", engine)
        if eng not in ("demucs", "demucs_ft", "roformer"):
            eng = engine
        sh = max(0, min(10, int(data.get("shifts", shifts))))
        ov = max(0.25, min(0.9, float(data.get("overlap", overlap))))
        why = str(data.get("why", "")).strip() or "Adjusted the settings for your complaint."
        # nothing changed? nudge with the ladder so the re-run is meaningful
        if (eng, sh, round(ov, 2)) == (engine, int(shifts), round(float(overlap), 2)):
            return fallback
        return (eng, sh, ov, "🤖 " + why + " Re-running.")
    except Exception:
        return fallback


def _rule_based_reply(message: str, available: list[str]) -> str:
    m = (message or "").lower()
    if any(h in m for h in _HELP_HINTS):
        return how_to_use()
    picks, why = recommend_stems(message, available)
    listed = ", ".join(picks)
    return (f"For that, I'd generate: **{listed}**.\n_{why}_\n\n"
            f"(Tip: available stems are {', '.join(available)}.)")


def chat(message: str, history: list[dict] | None, model_name: str,
         available: list[str], song: str | None = None) -> str:
    """Return an assistant reply. Falls back to rule-based on any error/missing config."""
    history = history or []
    if BACKEND == "none":
        return _rule_based_reply(message, available)
    try:
        messages = [{"role": "system", "content": _system_prompt(model_name, available, song)}]
        for turn in history[-6:]:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": message})

        if BACKEND in ("vllm", "openai"):
            from openai import OpenAI
            base = os.environ.get("VLLM_BASE_URL" if BACKEND == "vllm" else "OPENAI_BASE_URL")
            key = os.environ.get("OPENAI_API_KEY", "EMPTY")
            client = OpenAI(base_url=base, api_key=key)
            resp = client.chat.completions.create(model=MODEL, messages=messages,
                                                  max_tokens=300, temperature=0.4)
            return resp.choices[0].message.content.strip()

        # default: Hugging Face Inference API
        from huggingface_hub import InferenceClient
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        client = InferenceClient(model=MODEL, token=token)
        resp = client.chat_completion(messages=messages, max_tokens=300, temperature=0.4)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return (_rule_based_reply(message, available)
                + f"\n\n_(LLM backend unavailable — used built-in rules. {type(e).__name__})_")
