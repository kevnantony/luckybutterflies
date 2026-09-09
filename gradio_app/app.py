import gradio as gr
from document_parser import extract_text_from_file
from llm_handler import generate_learning_outcomes_prompt, stream_chat_response

# Developer encoded system prompt
DEV_SYSTEM_PROMPT = (
    "You are an empathetic, encouraging virtual teacher. Your goal is to guide the student towards "
    "the answer without just giving it away. You adapt your teaching style to the student's needs. "
    "Always maintain a supportive and constructive tone. "
)

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root & body ── */
:root {
    --bg-base:    #0d0e14;
    --bg-surface: #13151f;
    --bg-card:    #1a1d2e;
    --bg-input:   #1f2235;
    --border:     rgba(99, 102, 241, 0.18);
    --border-hover: rgba(139, 92, 246, 0.45);
    --accent-1:   #6366f1;
    --accent-2:   #8b5cf6;
    --accent-3:   #a78bfa;
    --text-primary:   #f0f0ff;
    --text-secondary: #9ca3af;
    --text-muted:     #4b5563;
    --success: #10b981;
    --glow: 0 0 30px rgba(99, 102, 241, 0.25);
}

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container, .main, footer {
    font-family: 'Inter', sans-serif !important;
    background: var(--bg-base) !important;
    color: var(--text-primary) !important;
}

/* Hide default footer */
footer { display: none !important; }

/* ── Header ── */
#app-header {
    text-align: center;
    padding: 28px 20px 8px;
}
#app-header h1 {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 60%, #e879f9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 6px;
    letter-spacing: -0.5px;
}
#app-header p {
    color: var(--text-secondary);
    font-size: 1rem;
    margin: 0;
    font-weight: 400;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 8px 0 20px;
}

/* ── Panels ── */
.left-panel, .right-panel {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    padding: 24px !important;
    transition: border-color 0.3s ease;
}
.left-panel:hover, .right-panel:hover {
    border-color: var(--border-hover) !important;
}

/* ── Section titles ── */
.section-label {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--accent-3) !important;
    margin-bottom: 14px !important;
}

/* ── File upload zone ── */
.file-upload-zone {
    background: var(--bg-input) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 14px !important;
    transition: border-color 0.3s, background 0.3s !important;
    cursor: pointer !important;
}
.file-upload-zone:hover {
    border-color: var(--accent-1) !important;
    background: rgba(99, 102, 241, 0.07) !important;
}
.file-upload-zone * { color: var(--text-secondary) !important; }

/* ── Buttons ── */
.extract-btn {
    background: linear-gradient(135deg, var(--accent-1) 0%, var(--accent-2) 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 12px 20px !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    width: 100% !important;
    margin-top: 10px !important;
}
.extract-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(139, 92, 246, 0.45) !important;
}
.extract-btn:active { transform: translateY(0) !important; }

.send-btn {
    background: linear-gradient(135deg, var(--accent-1) 0%, var(--accent-2) 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 10px 22px !important;
    height: 48px !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.25) !important;
}
.send-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4) !important;
}

/* ── Status output ── */
.status-out p {
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    padding: 10px 14px !important;
    border-radius: 10px !important;
    background: rgba(16, 185, 129, 0.1) !important;
    border: 1px solid rgba(16, 185, 129, 0.25) !important;
    color: #6ee7b7 !important;
    margin: 10px 0 0 !important;
}

/* ── Accordion ── */
.prompt-accordion {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-top: 12px !important;
}
.prompt-accordion > .label-wrap {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 10px 14px !important;
}
.prompt-accordion textarea {
    background: var(--bg-base) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-family: 'Inter', monospace !important;
}

/* ── Chatbot ── */
.chat-window {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    overflow: hidden !important;
}

/* Bot bubble */
.chat-window .message.bot {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 4px 16px 16px 16px !important;
    font-size: 0.92rem !important;
    line-height: 1.65 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
}

/* User bubble */
.chat-window .message.user {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #fff !important;
    border-radius: 16px 4px 16px 16px !important;
    font-size: 0.92rem !important;
    line-height: 1.65 !important;
    box-shadow: 0 2px 12px rgba(99, 102, 241, 0.35) !important;
}

/* ── Chat input row ── */
.chat-input-row {
    display: flex;
    gap: 10px;
    align-items: flex-end;
    margin-top: 12px;
}
.msg-textbox textarea {
    background: var(--bg-input) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    font-size: 0.92rem !important;
    transition: border-color 0.2s !important;
    padding: 12px 16px !important;
    resize: none !important;
}
.msg-textbox textarea:focus {
    border-color: var(--accent-1) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
    outline: none !important;
}
.msg-textbox textarea::placeholder { color: var(--text-muted) !important; }

/* ── Inputs/labels globally ── */
label, .block > label span {
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

/* ── Row spacing ── */
.gradio-row { gap: 18px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--accent-1); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-2); }

/* ── Fade-in animation ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
#app-header, .left-panel, .right-panel {
    animation: fadeUp 0.5s ease forwards;
}
.left-panel  { animation-delay: 0.1s; }
.right-panel { animation-delay: 0.2s; }
"""


def process_handbook(file_obj):
    if file_obj is None:
        return "⚠️ Please upload a file first.", "", gr.update(visible=False)
    file_path = file_obj.name
    try:
        text = extract_text_from_file(file_path)
    except Exception as e:
        return f"❌ Error extracting text: {e}", "", gr.update(visible=False)
    try:
        extracted_prompt = generate_learning_outcomes_prompt(text)
    except Exception as e:
        return f"❌ Error generating prompt: {e}", "", gr.update(visible=False)
    return "✅ Extraction complete! The Virtual Teacher is ready.", extracted_prompt, gr.update(visible=True)


def extract_text(content) -> str:
    """Normalize Gradio 6 content to a plain string for Ollama.
    
    Gradio 6.26 stores content as a list of {'text': ..., 'type': 'text'} dicts.
    Ollama requires a plain string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part["text"] for part in content if isinstance(part, dict) and "text" in part
        )
    return str(content)


def user(user_message, history):
    if not user_message.strip():
        return "", history
    return "", history + [{"role": "user", "content": user_message}]


def bot(history, extracted_prompt):
    if not history:
        yield history
        return
    combined_system_prompt = DEV_SYSTEM_PROMPT + "\n\n" + (extracted_prompt or "")
    # Build the messages list to pass to Ollama (all prior turns, fully resolved)
    messages = []
    for msg in history[:-1]:
        messages.append({"role": msg["role"], "content": extract_text(msg["content"])})
    # Add the latest user message
    messages.append({"role": "user", "content": extract_text(history[-1]["content"])})
    # Append a placeholder assistant message that we'll stream into
    history = history + [{"role": "assistant", "content": ""}]
    for chunk in stream_chat_response(messages, combined_system_prompt):
        history[-1]["content"] = chunk
        yield history


dark_theme = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(
    body_background_fill="#0d0e14",
    body_text_color="#f0f0ff",
    border_color_primary="rgba(99,102,241,0.18)",
    background_fill_primary="#13151f",
    background_fill_secondary="#1a1d2e",
    color_accent_soft="#1f2235",
    button_primary_background_fill="linear-gradient(135deg,#6366f1,#8b5cf6)",
    button_primary_text_color="#ffffff",
    block_label_text_color="#9ca3af",
    input_background_fill="#1f2235",
    input_border_color="rgba(99,102,241,0.18)",
)


with gr.Blocks() as app:
    extracted_system_prompt = gr.State("")

    # ── Header ──
    gr.HTML("""
        <div id="app-header">
            <h1>🎓 Virtual Teacher</h1>
            <p>Upload your course handbook and get an AI tutor tailored to your curriculum.</p>
        </div>
        <div class="divider"></div>
    """)

    with gr.Row(equal_height=True):
        # ── Left: Upload Panel ──
        with gr.Column(scale=1, min_width=300, elem_classes="left-panel"):
            gr.HTML("<p class='section-label'>① Setup</p>")
            file_input = gr.File(
                label="Course Handbook",
                file_types=[".pdf", ".txt"],
                elem_classes="file-upload-zone",
            )
            process_btn = gr.Button(
                "⚡ Extract Learning Objectives",
                variant="primary",
                elem_classes="extract-btn",
            )
            status_out = gr.Markdown("", elem_classes="status-out")

            with gr.Accordion(
                "📋 View Extracted System Prompt",
                open=False,
                visible=False,
                elem_classes="prompt-accordion",
            ) as prompt_accordion:
                prompt_out = gr.Textbox(
                    label="Generated prompt",
                    lines=10,
                    interactive=False,
                )

        # ── Right: Chat Panel ──
        with gr.Column(scale=2, elem_classes="right-panel"):
            gr.HTML("<p class='section-label'>② Chat with your Virtual Teacher</p>")
            chatbot = gr.Chatbot(
                height=460,
                show_label=False,
                elem_classes="chat-window",
                layout="bubble",
                placeholder="<strong>No messages yet.</strong><br>Upload a handbook and start asking questions!",
            )
            with gr.Row(elem_classes="chat-input-row"):
                msg_input = gr.Textbox(
                    show_label=False,
                    placeholder="Ask your virtual teacher anything…",
                    scale=8,
                    container=False,
                    elem_classes="msg-textbox",
                    lines=1,
                    max_lines=4,
                )
                submit_btn = gr.Button(
                    "Send ➤",
                    variant="primary",
                    scale=1,
                    elem_classes="send-btn",
                )

    # ── Event handlers ──
    process_btn.click(
        fn=process_handbook,
        inputs=[file_input],
        outputs=[status_out, extracted_system_prompt, prompt_accordion],
    ).then(
        fn=lambda p: p,
        inputs=[extracted_system_prompt],
        outputs=[prompt_out],
    )

    msg_input.submit(
        fn=user,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot],
        queue=False,
    ).then(
        fn=bot,
        inputs=[chatbot, extracted_system_prompt],
        outputs=[chatbot],
    )

    submit_btn.click(
        fn=user,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot],
        queue=False,
    ).then(
        fn=bot,
        inputs=[chatbot, extracted_system_prompt],
        outputs=[chatbot],
    )


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=custom_css,
        theme=dark_theme,
        debug=True,
    )
