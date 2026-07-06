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
# OpenAI is used as an automatic fallback when the HF call fails and OPENAI_API_KEY is set.
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

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
        "exactly as listed above and keep answers short and practical.\n\n"
        "The app also has quality settings you understand deeply:\n" + KNOB_GUIDE + "\n"
        "If the user asks what a setting does, explain it from the guide in plain language. "
        "If they complain that a stem sounds bad, tell them to describe the problem to you "
        "(e.g. 'the vocals have drum bleed') — you will adjust the settings and re-run "
        "automatically."
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
# What each knob actually does, symptom -> remedy. This is the assistant's
# domain knowledge: it goes verbatim into both the tuning prompt and the chat
# prompt, so the LLM adjusts the right knob and can explain any of them.
KNOB_GUIDE = """\
HOW THE QUALITY KNOBS WORK (and when to use each):

1. ENGINE — which separation model runs:
   - demucs: the standard hybrid model. Fast baseline. 4-stem mode uses one model;
     6- and 9-stem modes use a bigger 6-source model.
   - demucs_ft: fine-tuned variant of the 4-stem model. Noticeably cleaner vocals/
     drums/bass/other, ~4x slower. ONLY takes effect in 4-stem mode — in 6- or
     9-stem mode no fine-tuned model exists and choosing it changes NOTHING.
   - roformer: a newer band-split architecture. The cleanest vocal/instrumental
     split by a wide margin, but it ALWAYS outputs exactly 2 stems (vocals +
     instrumental) and ignores mode, stem selection, and shifts. Pick it only when
     the user cares about vocals or an instrumental/karaoke track.

2. SHIFTS (0-10): test-time augmentation. The song is shifted in time N ways,
   separated N times, and the results are averaged, which cancels random artifacts:
   hiss, warble, "underwater"/phasey/robotic textures, unstable or smeared notes.
   Cost is linear (shifts=4 is ~4x slower). Sweet spot 2-4; above 5 adds little.
   Helps every stem and both demucs engines; roformer ignores it.

3. OVERLAP (0.25-0.9): the song is processed in short chunks that are cross-faded.
   Higher overlap gives more context at chunk boundaries, fixing artifacts that
   repeat at regular intervals: clicks, seams, brief volume dips, choppy or "gated"
   moments. Raise to 0.5 for audible edge problems, 0.75 if they persist. Cost
   grows steeply above 0.75.

SYMPTOM -> BEST KNOB:
   - vocal bleed (music in the vocal stem, or voice left in the instrumental) ->
     roformer if a 2-stem result is acceptable; otherwise demucs_ft (4-stem mode
     only) plus shifts 2-4.
   - hiss / warble / underwater / robotic / phasey sound on a stem -> shifts 2-4.
   - clicks, seams, choppiness, dropouts at regular intervals -> overlap 0.5-0.75.
   - drums or bass weak / smeared / lacking punch -> shifts 2-4, then demucs_ft
     if in 4-stem mode.
   - guitar or piano stem is poor (6/9-stem modes) -> these are the model's weakest
     sources; raise shifts, and be honest that guitar/piano have inherent limits.
   - kick/snare/toms/cymbals complaints (9-stem mode) -> the drum split is a second
     model pass that also honors shifts/overlap, so raising them helps it too.
   - everything already maxed (roformer, or demucs_ft with shifts>=4 overlap>=0.75)
     -> say honestly that the model's ceiling is reached; suggest the other engine
     family or a different section of the song.
"""

# When a split looks dirty, the assistant escalates the engine/settings one rung
# up this ladder and re-runs. The decision is deterministic so it always changes
# something and works even with no LLM backend.
IMPROVE_HINTS = ("improve", "better", "cleaner", "clean it", "not clean", "not good",
                 "bad", "worse", "bleed", "bleeding", "noisy", "noise", "muddy",
                 "muffled", "artifact", "redo", "re-run", "rerun", "try again",
                 "fix", "leaking", "leak", "still hear", "still has", "still got",
                 "distort", "robotic", "metallic", "underwater", "watery", "phasey",
                 "echo", "choppy", "chopped", "click", "stutter", "glitch", "hiss",
                 "warble", "smear", "harsh", "hollow", "gated", "dropout",
                 "not great", "terrible", "awful", "horrible", "low quality",
                 "poor quality", "sounds off", "sounds wrong", "sounds weird")


def wants_improvement(message: str) -> bool:
    m = (message or "").lower()
    return any(h in m for h in IMPROVE_HINTS)


def improve_settings(complaint: str, engine: str, shifts: int,
                     overlap: float = 0.25, mode: str = "4") -> tuple[str, int, float, str]:
    """Deterministic escalation ladder -> (engine, shifts, overlap, why).

    Climbs the cheapest-first quality ladder: shifts -> overlap -> fine-tuned
    model -> RoFormer. Always changes something so a re-run is meaningful, and
    works with no LLM backend. The fine-tuned model only exists for 4-stem mode,
    so in 6/9-stem mode the ladder escalates shifts/overlap further instead.
    """
    c = (complaint or "").lower()
    shifts = int(shifts)
    overlap = float(overlap)
    mode = str(mode)
    vocal_focus = any(w in c for w in ("vocal", "voice", "sing", "acapella",
                                       "karaoke", "instrumental"))
    edge_focus = any(w in c for w in ("click", "chop", "seam", "stutter", "gap",
                                      "dropout", "gated", "cut out"))
    if engine == "demucs":
        if edge_focus and overlap < 0.5:
            return ("demucs", shifts, 0.5,
                    "That sounds like a chunk-boundary artifact — raised **overlap to 0.5** "
                    "to smooth the seams. Re-running.")
        if shifts < 2:
            return ("demucs", 2, overlap,
                    "Turned on the **shift trick (shifts=2)** — averages a few passes to "
                    "cut artifacts. Re-running (~2× slower).")
        if overlap < 0.5:
            return ("demucs", shifts, 0.5,
                    "Raised **overlap to 0.5** to smooth the chunk-edge artifacts. Re-running.")
        if mode == "4":
            return ("demucs_ft", max(shifts, 2), overlap,
                    "Switched to the **fine-tuned model (Demucs FT)** — cleaner separation, "
                    "~4× slower. Re-running.")
        # 6/9-stem mode: no fine-tuned model exists — push the averaging further
        if shifts < 4:
            return ("demucs", 4, overlap,
                    "Raised **shifts to 4** — more averaged passes for a cleaner result "
                    "(the fine-tuned model only exists for 4-stem mode). Re-running.")
        if vocal_focus:
            return ("roformer", shifts, overlap,
                    "Switched to **RoFormer** — the cleanest vocal/instrumental engine. "
                    "Note: it always outputs exactly 2 stems. Re-running.")
        if overlap < 0.75:
            return ("demucs", shifts, 0.75,
                    "Raised **overlap to 0.75** — maximum boundary smoothing. Re-running.")
        return ("demucs", shifts, overlap,
                "You're at the quality ceiling for this stem count. For vocals or an "
                "instrumental, **RoFormer** can do better (2 stems); or try **4-stem mode**, "
                "which has a cleaner fine-tuned model. Re-running once more.")
    if engine == "demucs_ft":
        if overlap < 0.5:
            return ("demucs_ft", shifts, 0.5,
                    "Raised **overlap to 0.5** on the fine-tuned model for smoother output. "
                    "Re-running.")
        if shifts < 4:
            return ("demucs_ft", 4, overlap,
                    "Raised **shifts to 4** on the fine-tuned model — more averaged passes. "
                    "Re-running.")
        extra = " RoFormer is especially strong on vocals." if vocal_focus else ""
        return ("roformer", shifts, overlap,
                "Switched to **RoFormer**, the cleanest engine (vocals/instrumental, "
                "always 2 stems)." + extra + " Re-running.")
    # already on roformer
    if overlap < 0.5:
        return ("roformer", shifts, 0.5,
                "Raised RoFormer **overlap to 0.5** for a smoother result. Re-running.")
    return ("roformer", shifts, overlap,
            "You're already on **RoFormer** at high overlap — the top-quality setting. "
            "If it's still not clean, the source mix may be the limit (try a different "
            "section). Re-running once more.")


# ---- provider fallback chain: primary (HF/Qwen) -> OpenAI -> rule-based ----
def _provider_order() -> list[str]:
    """LLM providers to try, in order. Adds OpenAI as a fallback when its key is set."""
    if BACKEND == "none":
        return []
    order = [BACKEND]
    if "openai" not in order and os.environ.get("OPENAI_API_KEY"):
        order.append("openai")
    return order


def _call_provider(provider: str, messages: list[dict], max_tokens: int, temperature: float) -> str:
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),
                        base_url=os.environ.get("OPENAI_BASE_URL") or None)
        resp = client.chat.completions.create(model=OPENAI_MODEL, messages=messages,
                                              max_tokens=max_tokens, temperature=temperature)
        return resp.choices[0].message.content
    if provider == "vllm":
        from openai import OpenAI
        client = OpenAI(base_url=os.environ.get("VLLM_BASE_URL"),
                        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
        resp = client.chat.completions.create(model=MODEL, messages=messages,
                                              max_tokens=max_tokens, temperature=temperature)
        return resp.choices[0].message.content
    # default: Hugging Face Inference API
    from huggingface_hub import InferenceClient
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    client = InferenceClient(model=MODEL, token=token)
    resp = client.chat_completion(messages=messages, max_tokens=max_tokens, temperature=temperature)
    return resp.choices[0].message.content


def _complete(messages: list[dict], max_tokens: int = 200, temperature: float = 0.4) -> str:
    """Try each provider in order (HF -> OpenAI); raise only if all fail."""
    errors = []
    for provider in _provider_order():
        try:
            return _call_provider(provider, messages, max_tokens, temperature)
        except Exception as e:
            errors.append(f"{provider}:{type(e).__name__}")
    raise RuntimeError("no LLM provider available (" + ", ".join(errors) + ")")


def _raw_completion(system_prompt: str, user_message: str, max_tokens: int = 200) -> str:
    """One-shot LLM completion via the provider chain. Raises if all providers fail."""
    return _complete([{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_message}],
                     max_tokens=max_tokens, temperature=0.1)


def improve_plan(complaint: str, engine: str, shifts: int, overlap: float,
                 mode: str = "4",
                 stems: list[str] | None = None) -> tuple[str, int, float, str]:
    """LLM-driven version: let the model analyze the complaint and choose the knobs.

    Gets the full KNOB_GUIDE plus the run context (mode, generated stems, current
    settings) so it picks the knob that actually addresses the symptom. Returns
    (engine, shifts, overlap, why). Falls back to the deterministic ladder when
    there's no LLM backend/token or the call fails — so it always does something.
    """
    mode = str(mode)
    fallback = improve_settings(complaint, engine, shifts, overlap, mode=mode)
    if BACKEND == "none":
        return fallback
    try:
        import json
        ctx = f"CURRENT STATE: {mode}-stem mode, engine={engine}, shifts={int(shifts)}, overlap={float(overlap)}."
        if stems:
            ctx += f" Stems already generated: {', '.join(stems)}."
        sys_prompt = (
            "You tune an audio source-separation app to fix a user's complaint about stem "
            "quality.\n\n" + KNOB_GUIDE + "\n" + ctx + "\n\n"
            "Choose the SINGLE best settings change for this complaint (change more than one "
            "knob only if the symptom clearly needs it). Hard rules:\n"
            "- NEVER pick 'demucs_ft' unless the mode is 4 — elsewhere it changes nothing.\n"
            "- Pick 'roformer' only for vocal/instrumental complaints, and warn in 'why' that "
            "the result will be exactly 2 stems (vocals + instrumental).\n"
            "- Always change something vs. the current state, and prefer the cheapest knob "
            "that addresses the symptom.\n"
            "In 'why' (1-2 short sentences, plain language): name the knob you changed, why it "
            "fixes THIS symptom, and the rough speed cost.\n"
            'Respond with ONLY JSON: {"engine":"...","shifts":0,"overlap":0.25,"why":"..."}'
        )
        raw = _raw_completion(sys_prompt, complaint or "Make it cleaner.")
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start:end + 1])
        eng = data.get("engine", engine)
        if eng not in ("demucs", "demucs_ft", "roformer"):
            eng = engine
        if eng == "demucs_ft" and mode != "4":
            eng = "demucs"      # guardrail: fine-tuned model doesn't exist for 6/9-stem
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
    if not _provider_order():
        return _rule_based_reply(message, available)
    try:
        messages = [{"role": "system", "content": _system_prompt(model_name, available, song)}]
        for turn in history[-6:]:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": message})
        return _complete(messages, max_tokens=300, temperature=0.4).strip()
    except Exception as e:
        return (_rule_based_reply(message, available)
                + f"\n\n_(LLM unavailable — used built-in rules. {type(e).__name__})_")
