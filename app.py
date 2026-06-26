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

DESCRIPTION = """
# 🎵 AI Stem Splitter
Upload a song, choose which **stems** (vocals, drums, bass, ...) you want, and download
each as its own audio file. The assistant on the right can recommend which stems to
generate for your goal. New here? Press **📖 Instructions**.
"""

# --- step-by-step "figure" slideshow shown by the Instructions button ---
SLIDES = [
    ("🎵", "Step 1 — Upload your song",
     "Drop a file into the upload box on the left. mp3, wav, flac and m4a all work."),
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
]


def slide_html(i: int) -> str:
    i = max(0, min(len(SLIDES) - 1, i))
    emoji, title, text = SLIDES[i]
    return f"""
    <div style="border:1px solid #6366f1;border-radius:14px;padding:28px 24px;
                text-align:center;background:linear-gradient(180deg,#1e1b4b22,#312e8122);
                min-height:180px;">
      <div style="font-size:64px;line-height:1;margin-bottom:12px;">{emoji}</div>
      <div style="font-size:20px;font-weight:700;margin-bottom:8px;">{title}</div>
      <div style="font-size:15px;opacity:.85;max-width:560px;margin:0 auto;">{text}</div>
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
        return ""
    return os.path.splitext(os.path.basename(file_path))[0]


def do_separate(file_path, mode, chosen, progress=gr.Progress()):
    if not file_path:
        raise gr.Error("Please upload an audio file first.")
    if not chosen:
        raise gr.Error("Select at least one stem to generate.")
    out_dir = tempfile.mkdtemp(prefix="stems_")
    paths = separate.separate_mode(
        file_path, out_dir, mode, chosen, device=None,
        progress=lambda f, m: progress(f, desc=m),
    )
    names = ", ".join(os.path.basename(p) for p in paths)
    # populate the merge picker with the stems we just created (label -> file path)
    merge_choices = [(separate.stem_of(p), p) for p in paths]
    return (
        paths,
        f"✅ Done — generated {len(paths)} stem(s): {names}",
        gr.update(choices=merge_choices, value=[]),
        gr.update(visible=len(paths) >= 2),
        None,
    )


def do_merge(selected, progress=gr.Progress()):
    if not selected or len(selected) < 2:
        raise gr.Error("Tick at least two stems to merge.")
    progress(0.3, desc="Mixing stems...")
    song = os.path.basename(selected[0]).split(" - ")[0]
    labels = "+".join(separate.stem_of(p) for p in selected)
    out_dir = tempfile.mkdtemp(prefix="merge_")
    out_path = os.path.join(out_dir, separate._safe(f"{song} - mix ({labels})") + ".wav")
    separate.merge_stems(selected, out_path)
    progress(1.0, desc="Done")
    return out_path, f"✅ Merged {len(selected)} stems → {os.path.basename(out_path)}"


def suggest_stems(goal, mode):
    available = separate.mode_stems(mode)
    picks, why = assistant.recommend_stems(goal, available)
    return gr.update(value=picks), f"Suggested **{', '.join(picks)}** — {why}"


def chat_fn(message, history, mode, song):
    available = separate.mode_stems(mode)
    reply = assistant.chat(message, history, mode, available, song)
    history = (history or []) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return history, ""


with gr.Blocks(title="AI Stem Splitter") as demo:
    gr.Markdown(DESCRIPTION)
    song_name = gr.State("")
    slide_idx = gr.State(0)

    instr_btn = gr.Button("📖 Instructions", size="sm")

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
            audio_in = gr.File(label="Upload song (mp3 / wav / flac / m4a ...)",
                               file_types=["audio"], type="filepath")
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
            go = gr.Button("🎚️ Separate selected stems", variant="primary")
            status = gr.Markdown("")
            files_out = gr.File(label="Download stems", file_count="multiple", interactive=False)

            with gr.Group(visible=False) as merge_box:
                gr.Markdown("### 🎚️ Merge stems\nTick 2–3 of the stems above and combine "
                            "them into one file (e.g. *drums + bass* for a rhythm track).")
                merge_cg = gr.CheckboxGroup(choices=[], label="Stems to merge")
                merge_btn = gr.Button("Merge selected into one file", variant="primary")
                merge_status = gr.Markdown("")
                merge_out = gr.Audio(label="Merged track", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### 🤖 Assistant\nAsk what each stem is, or describe your goal "
                        "(e.g. *“I want a karaoke track”*) and I'll suggest stems.")
            goal_box = gr.Textbox(label="Your goal", placeholder="e.g. karaoke / drumless practice / acapella")
            suggest_btn = gr.Button("Suggest stems for this goal")
            chatbot = gr.Chatbot(height=300, label="Chat")  # messages format is default in gradio 6
            msg = gr.Textbox(label="Message", placeholder="Ask the assistant anything about stems...")

    # wiring
    mode_dd.change(on_mode_change, mode_dd, stems_cg)
    audio_in.change(on_upload, audio_in, song_name)
    go.click(do_separate, [audio_in, mode_dd, stems_cg],
             [files_out, status, merge_cg, merge_box, merge_out])
    merge_btn.click(do_merge, merge_cg, [merge_out, merge_status])
    suggest_btn.click(suggest_stems, [goal_box, mode_dd], [stems_cg, status])
    msg.submit(chat_fn, [msg, chatbot, mode_dd, song_name], [chatbot, msg])

    # instructions slideshow
    instr_btn.click(lambda: (gr.update(visible=True), 0, slide_html(0),
                             f"Slide 1 of {len(SLIDES)}"),
                    None, [instr_box, slide_idx, slide_view, slide_counter])
    close_btn.click(lambda: gr.update(visible=False), None, instr_box)
    prev_btn.click(lambda i: nav_slide(i, -1), slide_idx,
                   [slide_idx, slide_view, slide_counter])
    next_btn.click(lambda i: nav_slide(i, 1), slide_idx,
                   [slide_idx, slide_view, slide_counter])

if __name__ == "__main__":
    demo.launch()
