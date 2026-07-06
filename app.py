"""
Gradio app: AI stem splitter (Demucs) with an LLM assistant.

Run locally:   python app.py
On Hugging Face Spaces: this file is the entry point (app_file: app.py).
"""
import os
import tempfile

import gradio as gr

import separate
import llm_assistant as assistant
import storage

# ---- Chartmetric-inspired dark theme (navy canvas, orange accent, teal/green) ----
_ORANGE = gr.themes.Color("#FFF1E9", "#FFE0CC", "#FFC3A0", "#FFA476", "#FF8552",
                          "#FF6A2B", "#E85A1E", "#C24A17", "#963811", "#6B270B", "#3A1607")
_TEAL = gr.themes.Color("#E0FBF4", "#B6F3E6", "#8AEAD7", "#59DDC5", "#33D1B5",
                        "#1FC8AE", "#17A892", "#118074", "#0C5F57", "#07403B", "#042320")
_NAVY = gr.themes.Color("#EDEFF3", "#C9CFDA", "#97A0AE", "#6E7686", "#4B5160",
                        "#3A4150", "#2B313D", "#1F242F", "#171B24", "#12161E", "#0E1116")

def _build_theme():
  return gr.themes.Base(
    primary_hue=_ORANGE, secondary_hue=_TEAL, neutral_hue=_NAVY,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
  ).set(
    # page + surfaces (light and dark set identically so it's always the dark look)
    body_background_fill="#0E1116", body_background_fill_dark="#0E1116",
    body_text_color="#EDEFF3", body_text_color_dark="#EDEFF3",
    background_fill_primary="#171B24", background_fill_primary_dark="#171B24",
    background_fill_secondary="#12161E", background_fill_secondary_dark="#12161E",
    block_background_fill="#171B24", block_background_fill_dark="#171B24",
    block_border_color="#2B313D", block_border_color_dark="#2B313D",
    block_label_text_color="#97A0AE", block_label_text_color_dark="#97A0AE",
    block_title_text_color="#EDEFF3", block_title_text_color_dark="#EDEFF3",
    border_color_primary="#2B313D", border_color_primary_dark="#2B313D",
    input_background_fill="#1F242F", input_background_fill_dark="#1F242F",
    input_border_color="#2B313D", input_border_color_dark="#2B313D",
    # orange primary buttons / teal links / orange controls
    button_primary_background_fill="#FF6A2B", button_primary_background_fill_dark="#FF6A2B",
    button_primary_background_fill_hover="#FF8552", button_primary_background_fill_hover_dark="#FF8552",
    button_primary_text_color="#2A1206", button_primary_text_color_dark="#2A1206",
    button_secondary_background_fill="#1F242F", button_secondary_background_fill_dark="#1F242F",
    button_secondary_text_color="#EDEFF3", button_secondary_text_color_dark="#EDEFF3",
    button_secondary_border_color="#3A4150", button_secondary_border_color_dark="#3A4150",
    slider_color="#FF6A2B", slider_color_dark="#FF6A2B",
    checkbox_background_color_selected="#FF6A2B", checkbox_background_color_selected_dark="#FF6A2B",
    link_text_color="#1FC8AE", link_text_color_dark="#1FC8AE",
  )


try:
    THEME = _build_theme()
except Exception:
    # Fallback if a theme token name differs in this Gradio version — still applies
    # the orange/teal/navy hues so the app boots with the right accent colors.
    THEME = gr.themes.Base(primary_hue=_ORANGE, secondary_hue=_TEAL, neutral_hue=_NAVY)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Anton&family=Oswald:wght@500;600;700&display=swap');
.gradio-container {max-width: 1080px !important; margin: 0 auto;}
footer {display: none !important;}
h1, h2, h3, .prose h1, .prose h2, .prose h3, .md h1, .md h2, .md h3 {
  font-family: 'Oswald', 'Arial Narrow', sans-serif !important;
  font-weight: 600 !important;
  letter-spacing: .01em;
}
.ok-msg, .ok-msg * { color: #2FD08A !important; }
/* Phone layout: let button rows wrap, keep touch targets tappable, trim padding. */
@media (max-width: 680px) {
  .gradio-container {padding: 6px !important;}
  .gradio-container .row {flex-wrap: wrap !important;}
  .gradio-container .row > * {min-width: 0 !important;}
  .gradio-container button {min-height: 44px;}
}
"""

# Green-forward waveforms on every player (teal track, green progress fill).
try:
    WAVEFORM = gr.WaveformOptions(waveform_color="#1FC8AE", waveform_progress_color="#2FD08A")
except Exception:
    WAVEFORM = None

# Billboard-style masthead: heavy black wordmark on a light band, with a SQUARED
# green strip directly beneath it that carries the tagline + intro text.
HEADER = """
<div style="background:#FFFFFF;padding:16px 20px 0;">
  <div style="font-family:'Anton',Impact,sans-serif;color:#0A0A0A;
              font-size:clamp(28px, 8vw, 44px);
              line-height:1.0;letter-spacing:.01em;">🎵 AI STEM SPLITTER</div>
</div>
<div style="background:#2FD08A;color:#06231A;padding:11px 20px 14px;border-radius:0;margin:0 0 12px;">
  <div style="font-family:'Oswald',sans-serif;font-weight:600;font-size:13px;
              letter-spacing:.12em;text-transform:uppercase;">
    Demucs · RoFormer · AI assistant — split any song into studio stems, free
  </div>
  <div style="font-family:Inter,system-ui,sans-serif;font-size:14px;line-height:1.55;margin-top:6px;">
    Upload a song, choose which <b>stems</b> (vocals, drums, bass, ...) you want, and download
    each as its own audio file. The assistant on the right can recommend which stems to generate
    for your goal. New here? Press <b>📖 Instructions</b>.
  </div>
</div>
"""

# --- step-by-step "figure" slideshow shown by the Instructions button ---
SLIDES = [
    ("🎵", "Step 1 — Upload your song",
     "Drop a file into the upload box (on a phone, tap it to pick from your Files). "
     "MP3, WAV, AIFF, FLAC, ALAC, AAC and M4A all work."),
    ("🔢", "Step 2 — Choose how many stems",
     "Pick 4, 6, or 9 stems. More stems take longer; 9 also splits the drum kit into "
     "kick, snare, toms and cymbals."),
    ("✅", "Step 3 — Select & separate",
     "Tick the stems you want to keep, then press “🎚️ Separate selected stems” and wait."),
    ("⬇️", "Step 4 — Download your stems",
     "When it finishes, each stem appears in the “Download stems” box — click to save them."),
    ("🎚️", "Step 5 — Merge (optional)",
     "In “Merge stems”, tick 2 or 3 of the new stems and press “Merge selected into one "
     "file” to combine them into a single custom track."),
    ("🤖", "Step 6 — Not clean? Ask the assistant",
     "If a stem sounds messy, tell the assistant (e.g. “vocals have bleed”). It bumps the "
     "quality settings — shifts, then the fine-tuned model, then RoFormer — and re-runs."),
]


def slide_html(i: int) -> str:
    i = max(0, min(len(SLIDES) - 1, i))
    emoji, title, text = SLIDES[i]
    return f"""
    <div style="border:1px solid #FF6A2B;border-radius:14px;padding:28px 24px;
                text-align:center;background:#12161E;min-height:180px;">
      <div style="font-size:64px;line-height:1;margin-bottom:12px;">{emoji}</div>
      <div style="font-family:'Oswald',sans-serif;font-size:22px;font-weight:600;margin-bottom:8px;color:#EDEFF3;">{title}</div>
      <div style="font-size:15px;color:#97A0AE;max-width:560px;margin:0 auto;">{text}</div>
    </div>
    """


def nav_slide(idx: int, delta: int):
    new = max(0, min(len(SLIDES) - 1, idx + delta))
    return new, slide_html(new), f"Slide {new + 1} of {len(SLIDES)}"


def on_mode_change(mode):
    stems = separate.mode_stems(mode)
    return gr.update(choices=stems, value=stems)


def on_upload(file_path):
    if not file_path:
        return "", None
    name = os.path.splitext(os.path.basename(file_path))[0]
    return name, file_path  # song name -> State, file path -> preview player


def _run_and_pack(file_path, mode, chosen, engine, shifts, overlap, history, progress, user_id=None):
    """Run separation with the chosen engine/shifts/overlap and build the output tuple.

    Shared by the Separate button and the assistant's improve-and-re-run flow.
    user_id (a logged-in HF username) routes the archive to permanent per-user storage.
    """
    if not file_path:
        raise gr.Error("Please upload an audio file first.")
    out_dir = tempfile.mkdtemp(prefix="stems_")
    # Transcode phone/Apple formats (AAC/ALAC/...) to WAV on CPU, before the GPU call.
    progress(0.05, desc="Preparing audio…")
    src_path = separate.prepare_input(file_path)
    progress(0.1, desc=f"Separating ({engine})… first RoFormer run downloads the model.")
    paths = separate.gpu_separate(src_path, out_dir, engine, mode, chosen, int(shifts), float(overlap))
    progress(1.0, desc="Done")
    # archive this run (input + stems) — permanent per-user if signed in, else the 14-day pool
    storage.save_run_async(file_path, paths,
                           {"mode": mode, "engine": engine, "shifts": int(shifts),
                            "overlap": float(overlap)}, user_id=user_id)
    names = ", ".join(separate.stem_of(p) for p in paths)
    merge_choices = [(separate.stem_of(p), p) for p in paths]
    history = (history or []) + [{
        "role": "assistant",
        "content": (f"✅ Stems are ready: {names}.\n\n**Not clean enough?** Tell me what's "
                    "wrong (e.g. *“vocals have drum bleed”*) and I'll boost the settings and re-run."),
    }]
    return (
        paths,                                          # files_out
        f"✅ Done — generated {len(paths)} stem(s): {names}",   # status
        gr.update(choices=merge_choices, value=[]),     # merge_cg
        gr.update(visible=len(paths) >= 2),             # merge_box
        None,                                           # merge_out (clear latest-merge player)
        paths,                                          # stem_paths -> preview players
        history,                                        # chatbot
        None,                                           # merged_files (clear old merges)
        [],                                             # merged_paths state (reset)
    )


def do_separate(file_path, mode, chosen, engine, shifts, overlap, history,
                progress=gr.Progress(), profile: gr.OAuthProfile | None = None):
    if engine != "roformer" and not chosen:
        raise gr.Error("Select at least one stem to generate.")
    uid = profile.username if profile else None
    return _run_and_pack(file_path, mode, chosen, engine, shifts, overlap, history, progress, user_id=uid)


def improve_rerun(flag, file_path, mode, chosen, engine, shifts, overlap, history,
                  progress=gr.Progress(), profile: gr.OAuthProfile | None = None):
    """Chained after a chat turn: re-run separation only when the assistant asked to."""
    if not flag:
        return gr.skip()
    uid = profile.username if profile else None
    return _run_and_pack(file_path, mode, chosen, engine, shifts, overlap, history, progress, user_id=uid)


def do_reset():
    """Clear everything back to the initial state so a new file can be dropped in."""
    default_stems = separate.mode_stems(separate.DEFAULT_MODE)
    return (
        None,                                            # audio_in
        None,                                            # input_preview
        separate.DEFAULT_MODE,                           # mode_dd
        gr.update(choices=default_stems, value=default_stems),  # stems_cg
        "",                                              # status
        None,                                            # files_out
        gr.update(visible=False),                        # merge_box
        gr.update(choices=[], value=[]),                 # merge_cg
        None,                                            # merge_out
        "",                                              # merge_status
        [],                                              # stem_paths (clears preview players)
        "",                                              # song_name
        "demucs",                                        # engine_dd
        0,                                               # shifts_sl
        0.25,                                            # overlap_sl
        False,                                           # improve_flag
        [],                                              # chatbot (clear chat)
        None,                                            # merged_files
        [],                                              # merged_paths
    )


def do_merge(selected, merged_paths, progress=gr.Progress()):
    if not selected or len(selected) < 2:
        raise gr.Error("Tick at least two stems to merge.")
    progress(0.3, desc="Mixing stems...")
    song = os.path.basename(selected[0]).split(" - ")[0]
    labels = "+".join(separate.stem_of(p) for p in selected)
    out_dir = tempfile.mkdtemp(prefix="merge_")
    out_path = os.path.join(out_dir, separate._safe(f"{song} - mix ({labels})") + ".wav")
    separate.merge_stems(selected, out_path)
    progress(1.0, desc="Done")
    merged_paths = (merged_paths or []) + [out_path]
    return (
        out_path,                       # merge_out (latest merged track)
        merged_paths,                   # merged_files (all merges, downloadable)
        merged_paths,                   # merged_paths state
        gr.update(value=[]),            # merge_cg -> clear selection for the next merge
        f"✅ Merged {len(selected)} stems → {os.path.basename(out_path)}. "
        f"Pick another set and merge again — {len(merged_paths)} mix(es) so far.",
    )


def suggest_stems(goal, mode):
    available = separate.mode_stems(mode)
    picks, why = assistant.recommend_stems(goal, available)
    return gr.update(value=picks), f"Suggested **{', '.join(picks)}** — {why}"


def chat_fn(message, history, mode, song, engine, shifts, overlap, paths):
    history = history or []
    # If a split exists and the user complains about quality, let the LLM analyze the
    # complaint, pick the best engine/shifts/overlap, and flag an automatic re-run.
    if paths and assistant.wants_improvement(message):
        new_engine, new_shifts, new_overlap, why = assistant.improve_plan(
            message, engine, int(shifts), float(overlap),
            mode=mode, stems=[separate.stem_of(p) for p in paths])
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": why},
        ]
        return (history, "", gr.update(value=new_engine), gr.update(value=new_shifts),
                gr.update(value=new_overlap), True)
    available = separate.mode_stems(mode)
    reply = assistant.chat(message, history, mode, available, song)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return history, "", gr.update(), gr.update(), gr.update(), False


# ---------------- account / my-data (HF OAuth) ----------------
def account_view(profile: gr.OAuthProfile | None = None):
    if profile is None:
        return ("**Not signed in.** The app works fully without an account — but anonymous "
                "uploads are auto-deleted after 14 days. **Sign in** with the button above to keep "
                "your songs and stems permanently and browse them here.")
    return (f"### 👋 Signed in as **{profile.username}**\n"
            "Your uploads and stems are saved **permanently** under your account. "
            "Manage your password / account at "
            "[Hugging Face settings](https://huggingface.co/settings/account). "
            "Press **Load my saved songs** below to browse them.")


def load_my_data(profile: gr.OAuthProfile | None = None):
    if profile is None:
        return ([], gr.update(choices=[], value=None), gr.update(choices=[], value=None),
                "Sign in to see your saved songs.")
    songs = storage.list_user_songs(profile.username)
    if not songs:
        return ([], gr.update(choices=[], value=None), gr.update(choices=[], value=None),
                "No saved songs yet — separate a track while signed in and it'll appear here.")
    song_choices = [(f"{s['title']}  ·  {len(s['runs'])} version(s)", s["song_id"]) for s in songs]
    lines = [f"You have **{len(songs)}** saved song(s):"]
    for s in songs:
        nstems = sum(len(r["stems"]) for r in s["runs"])
        lines.append(f"- **{s['title']}** — {len(s['runs'])} version(s), {nstems} stem files")
    runs = songs[0]["runs"]
    run_choices = [(f"{r['run_id']}  ·  {len(r['stems'])} stems", r["run_id"]) for r in runs]
    return (songs,
            gr.update(choices=song_choices, value=song_choices[0][1]),
            gr.update(choices=run_choices, value=run_choices[0][1] if run_choices else None),
            "\n".join(lines))


def on_song_select(song_id, songs):
    s = next((x for x in (songs or []) if x["song_id"] == song_id), None)
    if not s:
        return gr.update(choices=[], value=None)
    run_choices = [(f"{r['run_id']}  ·  {len(r['stems'])} stems", r["run_id"]) for r in s["runs"]]
    return gr.update(choices=run_choices, value=run_choices[0][1] if run_choices else None)


def fetch_stems(song_id, run_id, profile: gr.OAuthProfile | None = None):
    if profile is None or not song_id or not run_id:
        return None
    return storage.fetch_run_stems(profile.username, song_id, run_id) or None


with gr.Blocks(title="AI Stem Splitter", theme=THEME, css=CSS) as demo:
    gr.HTML(HEADER)
    song_name = gr.State("")
    slide_idx = gr.State(0)
    stem_paths = gr.State([])   # output stem file paths -> drives the preview players
    improve_flag = gr.State(False)  # set by chat when the user asks to improve the split
    merged_paths = gr.State([])  # accumulates every merged-track file for download
    songs_state = gr.State([])   # cached list of the signed-in user's saved songs

    instr_btn = gr.Button("📖 Instructions", size="sm")

    with gr.Accordion("👤 Account & my data", open=False):
        gr.LoginButton()
        account_md = gr.Markdown()
        load_btn = gr.Button("🔄 Load my saved songs", size="sm")
        with gr.Row():
            song_dd = gr.Dropdown(label="Your songs", choices=[], scale=3)
            run_dd = gr.Dropdown(label="Version (run)", choices=[], scale=3)
            fetch_btn = gr.Button("⬇️ Get these stems", scale=1)
        account_data_md = gr.Markdown()
        account_files = gr.File(label="Your saved stems", file_count="multiple", interactive=False)

    # collapsible step-by-step "figure" slideshow
    with gr.Group(visible=False) as instr_box:
        slide_view = gr.HTML(slide_html(0))
        with gr.Row():
            prev_btn = gr.Button("← Back")
            slide_counter = gr.Markdown(f"Slide 1 of {len(SLIDES)}")
            next_btn = gr.Button("Next →")
        close_btn = gr.Button("✕ Close instructions", size="sm")

    with gr.Row():
        with gr.Column(scale=3):
            # "audio" alone becomes accept="audio/*", which iOS Safari uses to gray
            # out FLAC/AIFF/etc. in the Files picker — list extensions explicitly too.
            audio_in = gr.File(
                label="Upload song (MP3 · WAV · AIFF · FLAC · ALAC · AAC · M4A …)",
                file_types=["audio", ".mp3", ".wav", ".aiff", ".aif", ".aifc", ".flac",
                            ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wma", ".caf"],
                type="filepath")
            input_preview = gr.Audio(label="▶ Preview uploaded song", interactive=False,
                                     waveform_options=WAVEFORM)
            mode_dd = gr.Dropdown(
                choices=[
                    ("4 stems · vocals, drums, bass, other", "4"),
                    ("6 stems · + guitar, piano", "6"),
                    ("9 stems · 6 + drums split into kick, snare, toms, cymbals", "9"),
                ],
                value=separate.DEFAULT_MODE,
                label="How many stems?",
                info="More stems take longer. 9 splits the drum kit into its pieces.",
            )
            stems_cg = gr.CheckboxGroup(
                choices=separate.mode_stems(separate.DEFAULT_MODE),
                value=separate.mode_stems(separate.DEFAULT_MODE),
                label="Stems to generate",
            )
            with gr.Accordion("⚙️ Quality (engine & shifts) — adjust, then Re-run", open=True):
                engine_dd = gr.Dropdown(
                    choices=[
                        ("Demucs — fast", "demucs"),
                        ("Demucs FT — cleaner, ~4× slower", "demucs_ft"),
                        ("RoFormer — best vocals/instrumental (2 stems)", "roformer"),
                    ],
                    value="demucs", label="Engine",
                    info="Higher quality = slower. RoFormer ignores the stem count "
                         "(always vocals + instrumental).",
                )
                shifts_sl = gr.Slider(
                    0, 10, value=0, step=1, label="Shifts",
                    info="0 = fastest. Higher averages more passes for cleaner output "
                         "(linear time cost). The assistant can raise this for you.",
                )
                overlap_sl = gr.Slider(
                    0.25, 0.9, value=0.25, step=0.05, label="Overlap",
                    info="Window overlap. Higher = smoother, fewer chunk-edge artifacts "
                         "(slower). The assistant can raise this too.",
                )
            with gr.Row():
                go = gr.Button("🎚️ Get stems", variant="primary")
                rerun = gr.Button("🔁 Re-run", variant="secondary")
                reset = gr.Button("🔄 Reset", variant="secondary")
            status = gr.Markdown("", elem_classes=["ok-msg"])
            files_out = gr.File(label="Download stems", file_count="multiple", interactive=False)

            gr.Markdown("#### ▶ Preview stems")

            @gr.render(inputs=stem_paths)
            def _stem_players(paths):
                if not paths:
                    gr.Markdown("*Separate a song to preview each stem here.*")
                    return
                for p in paths:
                    gr.Audio(value=p, label=separate.stem_of(p), interactive=False,
                             waveform_options=WAVEFORM)

            with gr.Group(visible=False) as merge_box:
                gr.Markdown("### 🎚️ Merge stems\nTick 2–3 stems and combine them into one file "
                            "(e.g. *drums + bass* for a rhythm track). You can merge again with a "
                            "different set — each mix is added to the list below.")
                merge_cg = gr.CheckboxGroup(choices=[], label="Stems to merge")
                merge_btn = gr.Button("Merge selected into one file", variant="primary")
                merge_status = gr.Markdown("", elem_classes=["ok-msg"])
                merge_out = gr.Audio(label="▶ Latest merged track", interactive=False,
                                     waveform_options=WAVEFORM)
                merged_files = gr.File(label="Download merged tracks", file_count="multiple",
                                       interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### 🤖 Assistant\nAsk what each stem is, or describe your goal "
                        "(e.g. *“I want a karaoke track”*) and I'll suggest stems.")
            goal_box = gr.Textbox(label="Your goal", placeholder="e.g. karaoke / drumless practice / acapella")
            suggest_btn = gr.Button("Suggest stems for this goal")
            chatbot = gr.Chatbot(height=300, label="Chat")  # messages format is default in gradio 6
            msg = gr.Textbox(label="Message", placeholder="Ask the assistant anything about stems...")

    # wiring
    mode_dd.change(on_mode_change, mode_dd, stems_cg)
    audio_in.change(on_upload, audio_in, [song_name, input_preview])
    sep_outputs = [files_out, status, merge_cg, merge_box, merge_out, stem_paths, chatbot,
                   merged_files, merged_paths]
    sep_inputs = [audio_in, mode_dd, stems_cg, engine_dd, shifts_sl, overlap_sl, chatbot]
    go.click(do_separate, sep_inputs, sep_outputs)
    rerun.click(do_separate, sep_inputs, sep_outputs)
    merge_btn.click(do_merge, [merge_cg, merged_paths],
                    [merge_out, merged_files, merged_paths, merge_cg, merge_status])
    reset.click(do_reset, None,
                [audio_in, input_preview, mode_dd, stems_cg, status, files_out,
                 merge_box, merge_cg, merge_out, merge_status, stem_paths, song_name,
                 engine_dd, shifts_sl, overlap_sl, improve_flag, chatbot, merged_files, merged_paths])
    suggest_btn.click(suggest_stems, [goal_box, mode_dd], [stems_cg, status])
    # chat: answer; if it was an "improve" request, the LLM picks engine/shifts/overlap, then re-run
    msg.submit(chat_fn,
               [msg, chatbot, mode_dd, song_name, engine_dd, shifts_sl, overlap_sl, stem_paths],
               [chatbot, msg, engine_dd, shifts_sl, overlap_sl, improve_flag]).then(
        improve_rerun,
        [improve_flag, audio_in, mode_dd, stems_cg, engine_dd, shifts_sl, overlap_sl, chatbot],
        sep_outputs)

    # instructions slideshow
    instr_btn.click(lambda: (gr.update(visible=True), 0, slide_html(0),
                             f"Slide 1 of {len(SLIDES)}"),
                    None, [instr_box, slide_idx, slide_view, slide_counter])
    close_btn.click(lambda: gr.update(visible=False), None, instr_box)
    prev_btn.click(lambda i: nav_slide(i, -1), slide_idx,
                   [slide_idx, slide_view, slide_counter])
    next_btn.click(lambda i: nav_slide(i, 1), slide_idx,
                   [slide_idx, slide_view, slide_counter])

    # account / my-data (HF OAuth); profile is auto-injected (None when anonymous)
    demo.load(account_view, None, account_md)
    load_btn.click(load_my_data, None, [songs_state, song_dd, run_dd, account_data_md])
    song_dd.change(on_song_select, [song_dd, songs_state], run_dd)
    fetch_btn.click(fetch_stems, [song_dd, run_dd], account_files)

if __name__ == "__main__":
    demo.launch()
