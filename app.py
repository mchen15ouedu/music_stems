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
each as its own `.wav`. Powered by **Demucs** (free, pretrained — no training needed).
The assistant on the right can recommend which stems to generate for your goal.
"""


def on_model_change(model_name):
    stems = separate.list_stems(model_name)
    return gr.update(choices=stems, value=stems)


def on_upload(file_path):
    if not file_path:
        return ""
    return os.path.splitext(os.path.basename(file_path))[0]


def do_separate(file_path, model_name, chosen, progress=gr.Progress()):
    if not file_path:
        raise gr.Error("Please upload an audio file first.")
    if not chosen:
        raise gr.Error("Select at least one stem to generate.")
    out_dir = tempfile.mkdtemp(prefix="stems_")
    paths = separate.separate(
        file_path, out_dir, model_name, chosen, device=None,
        progress=lambda f, m: progress(f, desc=m),
    )
    names = ", ".join(os.path.basename(p) for p in paths)
    return paths, f"✅ Done — generated {len(paths)} stem(s): {names}"


def suggest_stems(goal, model_name):
    available = separate.list_stems(model_name)
    picks, why = assistant.recommend_stems(goal, available)
    return gr.update(value=picks), f"Suggested **{', '.join(picks)}** — {why}"


def chat_fn(message, history, model_name, song):
    available = separate.list_stems(model_name)
    reply = assistant.chat(message, history, model_name, available, song)
    history = (history or []) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return history, ""


with gr.Blocks(title="AI Stem Splitter") as demo:
    gr.Markdown(DESCRIPTION)
    song_name = gr.State("")

    with gr.Row():
        with gr.Column(scale=3):
            audio_in = gr.File(label="Upload song (mp3 / wav / flac / m4a ...)",
                               file_types=["audio"], type="filepath")
            model_dd = gr.Dropdown(
                choices=separate.available_models(), value=separate.DEFAULT_MODEL,
                label="Separation model",
                info="htdemucs = 4 stems (best quality). htdemucs_6s = 6 stems (adds guitar & piano).",
            )
            stems_cg = gr.CheckboxGroup(
                choices=separate.list_stems(separate.DEFAULT_MODEL),
                value=separate.list_stems(separate.DEFAULT_MODEL),
                label="Stems to generate",
            )
            go = gr.Button("🎚️ Separate selected stems", variant="primary")
            status = gr.Markdown("")
            files_out = gr.File(label="Download stems", file_count="multiple", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### 🤖 Assistant\nAsk what each stem is, or describe your goal "
                        "(e.g. *“I want a karaoke track”*) and I'll suggest stems.")
            goal_box = gr.Textbox(label="Your goal", placeholder="e.g. karaoke / drumless practice / acapella")
            suggest_btn = gr.Button("Suggest stems for this goal")
            chatbot = gr.Chatbot(type="messages", height=300, label="Chat")
            msg = gr.Textbox(label="Message", placeholder="Ask the assistant anything about stems...")

    # wiring
    model_dd.change(on_model_change, model_dd, stems_cg)
    audio_in.change(on_upload, audio_in, song_name)
    go.click(do_separate, [audio_in, model_dd, stems_cg], [files_out, status])
    suggest_btn.click(suggest_stems, [goal_box, model_dd], [stems_cg, status])
    msg.submit(chat_fn, [msg, chatbot, model_dd, song_name], [chatbot, msg])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
