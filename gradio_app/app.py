import base64
from pathlib import Path
import gradio as gr
from document_parser import extract_text_from_file
from llm_handler import (
    generate_learning_outcomes_prompt,
    stream_learning_outcomes_prompt,
    stream_chat_response,
    generate_progress_report,
)

# Developer encoded system prompt
DEV_SYSTEM_PROMPT = (
    "You are a teacher who helps a student study without directly giving answers but rather by giving hints and support in a clear and consise manner and tries to make a student undrstand a concept in a pedagogically exceptional way rather than just answering the question." 
)

# Helper to read local images as base64 data URLs
def load_image_base64(rel_path: str) -> str:
    base_dir = Path(__file__).parent
    img_path = base_dir / rel_path
    if img_path.exists():
        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            suffix = img_path.suffix.lstrip(".").lower()
            mime = "jpeg" if suffix in ["jpg", "jpeg"] else suffix
            return f"data:image/{mime};base64,{encoded}"
    return ""

AVATAR_B64 = load_image_base64("assets/teacher_avatar.jpg")
BANNER_B64 = load_image_base64("assets/analytics_banner.jpg")

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root Design System ── */
:root {
    --bg-base:      #090a10;
    --bg-surface:   #0f111d;
    --bg-card:      #151828;
    --bg-card-alt:  #1b1f33;
    --bg-input:     #171a2d;
    --border:       rgba(99, 102, 241, 0.16);
    --border-hover: rgba(139, 92, 246, 0.45);
    --border-focus: rgba(99, 102, 241, 0.7);
    --accent-1:     #6366f1;
    --accent-2:     #8b5cf6;
    --accent-3:     #a78bfa;
    --emerald-1:    #059669;
    --emerald-2:    #10b981;
    --emerald-3:    #34d399;
    --text-primary:   #f3f4fd;
    --text-secondary: #9aa3bf;
    --text-muted:     #525b7a;
    --glow-indigo:  0 0 24px rgba(99, 102, 241, 0.28);
    --glow-emerald: 0 0 24px rgba(16, 185, 129, 0.28);
    --radius-sm:    8px;
    --radius-md:    12px;
    --radius-lg:    18px;
    --radius-xl:    24px;
}

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container, .main, footer {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: var(--bg-base) !important;
    color: var(--text-primary) !important;
}

footer { display: none !important; }

/* ── App Header ── */
#app-header-wrap {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 24px;
    background: linear-gradient(180deg, rgba(21, 24, 40, 0.85) 0%, rgba(15, 17, 29, 0.4) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    margin-bottom: 20px;
    backdrop-filter: blur(14px);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
}

.header-left {
    display: flex;
    align-items: center;
    gap: 16px;
}

.header-avatar {
    width: 52px;
    height: 52px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid var(--accent-1);
    box-shadow: var(--glow-indigo);
}

.header-title-group h1 {
    font-size: 1.6rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #a5b4fc 0%, #c084fc 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.header-title-group p {
    margin: 2px 0 0;
    font-size: 0.85rem;
    color: var(--text-secondary);
}

.header-badges {
    display: flex;
    align-items: center;
    gap: 10px;
}

.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #c7d2fe;
}

.status-chip-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 10px #10b981;
    animation: pulseDot 2s infinite ease-in-out;
}

@keyframes pulseDot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.5; transform: scale(0.85); }
}

/* ── Layout & Containerization Fix ── */
.main-content-row {
    align-items: flex-start !important;
    gap: 20px !important;
}

/* Left panel stays firmly pinned at top, never stretching or distorting */
.left-panel {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-xl) !important;
    padding: 22px !important;
    align-self: flex-start !important;
    position: sticky !important;
    top: 16px !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25) !important;
}
.left-panel:hover {
    border-color: var(--border-hover) !important;
}

.right-panel {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-xl) !important;
    padding: 22px !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25) !important;
}
.right-panel:hover {
    border-color: var(--border-hover) !important;
}

/* ── Mentor Profile Card (Left Panel) ── */
.mentor-card {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 12px 14px;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.05) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: var(--radius-md);
    margin-bottom: 18px;
}

.mentor-card img {
    width: 46px;
    height: 46px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid var(--accent-2);
}

.mentor-info h3 {
    margin: 0;
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-primary);
}

.mentor-info p {
    margin: 2px 0 0;
    font-size: 0.76rem;
    color: var(--text-secondary);
}

/* ── Section Titles ── */
.section-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--accent-3) !important;
    margin-bottom: 12px !important;
}

/* ── File Upload Zone ── */
.file-upload-zone {
    background: var(--bg-input) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius-md) !important;
    transition: all 0.25s ease !important;
}
.file-upload-zone:hover {
    border-color: var(--accent-1) !important;
    background: rgba(99, 102, 241, 0.06) !important;
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.12) !important;
}
.file-upload-zone * {
    color: var(--text-secondary) !important;
}

/* ── UNIFORM BUTTON DESIGN SYSTEM ── */
/* All action buttons share matching dimensions, font size, and border radius */
.btn-uniform {
    height: 44px !important;
    min-height: 44px !important;
    max-height: 44px !important;
    border-radius: var(--radius-md) !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    padding: 0 18px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
    transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    line-height: 1 !important;
    border: none !important;
    text-decoration: none !important;
}

.btn-primary {
    background: linear-gradient(135deg, var(--accent-1) 0%, var(--accent-2) 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.32) !important;
}
.btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.45) !important;
}
.btn-primary:active {
    transform: translateY(0) !important;
}

.btn-success {
    background: linear-gradient(135deg, var(--emerald-1) 0%, var(--emerald-2) 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3) !important;
}
.btn-success:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.45) !important;
}
.btn-success:active {
    transform: translateY(0) !important;
}

.btn-ghost {
    background: var(--bg-card-alt) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
}
.btn-ghost:hover {
    color: var(--text-primary) !important;
    border-color: var(--border-hover) !important;
    background: rgba(99, 102, 241, 0.1) !important;
    transform: translateY(-1px) !important;
}

/* ── Status output ── */
.status-out p {
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    padding: 10px 14px !important;
    border-radius: var(--radius-md) !important;
    background: rgba(16, 185, 129, 0.1) !important;
    border: 1px solid rgba(16, 185, 129, 0.25) !important;
    color: #6ee7b7 !important;
    margin: 10px 0 0 !important;
}

/* ── Accordion (Left Panel Prompt) ── */
.prompt-accordion {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    margin-top: 14px !important;
}
.prompt-accordion > .label-wrap {
    color: var(--text-secondary) !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    padding: 10px 14px !important;
}
.prompt-accordion textarea {
    background: var(--bg-base) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.80rem !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Tabs Styling ── */
.workspace-tabs {
    border: none !important;
    background: transparent !important;
}
.workspace-tabs > div:first-child {
    border-bottom: 1px solid var(--border) !important;
    margin-bottom: 16px !important;
    gap: 8px !important;
}
.workspace-tabs button {
    font-size: 0.90rem !important;
    font-weight: 600 !important;
    border-radius: var(--radius-md) var(--radius-md) 0 0 !important;
    padding: 10px 18px !important;
    color: var(--text-secondary) !important;
    transition: all 0.2s ease !important;
    border: none !important;
}
.workspace-tabs button.selected {
    color: var(--text-primary) !important;
    background: rgba(99, 102, 241, 0.14) !important;
    border-bottom: 2px solid var(--accent-1) !important;
}

/* ── Quick Prompt Pills ── */
.prompt-chips-container {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 12px;
}
.chip-btn {
    height: 32px !important;
    min-height: 32px !important;
    max-height: 32px !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 0 12px !important;
    border-radius: 999px !important;
    background: var(--bg-input) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
    transition: all 0.2s ease !important;
}
.chip-btn:hover {
    color: var(--text-primary) !important;
    border-color: var(--accent-1) !important;
    background: rgba(99, 102, 241, 0.12) !important;
    transform: translateY(-1px) !important;
}

/* ── Chatbot Window ── */
.chat-window {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
}

.chat-window .message.bot {
    background: var(--bg-card-alt) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 4px 18px 18px 18px !important;
    font-size: 0.92rem !important;
    line-height: 1.65 !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25) !important;
}

.chat-window .message.user {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    border-radius: 18px 4px 18px 18px !important;
    font-size: 0.92rem !important;
    line-height: 1.65 !important;
    box-shadow: 0 3px 14px rgba(99, 102, 241, 0.35) !important;
}

/* ── Chat Input Row ── */
.chat-input-row {
    display: flex;
    gap: 10px;
    align-items: center;
    margin-top: 12px;
}

.msg-textbox textarea {
    height: 44px !important;
    min-height: 44px !important;
    background: var(--bg-input) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    font-size: 0.92rem !important;
    padding: 10px 16px !important;
    transition: all 0.2s ease !important;
}
.msg-textbox textarea:focus {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
    outline: none !important;
}
.msg-textbox textarea::placeholder {
    color: var(--text-muted) !important;
}

/* Action Toolbar */
.chat-action-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--border);
}

/* ── Progress & Analytics Workspace ── */
.analytics-hero-banner {
    position: relative;
    width: 100%;
    height: 140px;
    border-radius: var(--radius-lg);
    overflow: hidden;
    margin-bottom: 16px;
    border: 1px solid var(--border);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
}

.analytics-hero-banner img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    filter: brightness(0.75) contrast(1.1);
}

.analytics-banner-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, rgba(15, 17, 29, 0.92) 0%, rgba(15, 17, 29, 0.35) 100%);
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 0 24px;
}

.analytics-banner-overlay h2 {
    margin: 0;
    font-size: 1.4rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.01em;
}

.analytics-banner-overlay p {
    margin: 4px 0 0;
    font-size: 0.84rem;
    color: #9aa3bf;
}

.report-controls-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 14px;
}

/* Bounded scroll container so report never distorts height or page */
.report-scroll-container {
    max-height: 520px !important;
    overflow-y: auto !important;
    padding: 20px !important;
    background: var(--bg-surface) !important;
    border: 1px solid rgba(16, 185, 129, 0.22) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.3) !important;
}

.report-scroll-container h2 {
    color: #34d399 !important;
    font-size: 1.35rem !important;
    font-weight: 800 !important;
    margin-top: 0 !important;
    margin-bottom: 14px !important;
    border-bottom: 1px solid rgba(16, 185, 129, 0.2) !important;
    padding-bottom: 8px !important;
}

.report-scroll-container h3 {
    color: #e0e7ff !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    margin-top: 18px !important;
    margin-bottom: 8px !important;
}

.report-scroll-container ul {
    padding-left: 20px !important;
    margin-bottom: 14px !important;
}

.report-scroll-container li {
    margin-bottom: 6px !important;
    color: #d1d5db !important;
    font-size: 0.90rem !important;
    line-height: 1.6 !important;
}

.report-scroll-container p {
    color: #d1d5db !important;
    font-size: 0.90rem !important;
    line-height: 1.65 !important;
}

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-1); }

/* ── Animations ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
/* ── Suppress Any Native Gradio Progress Bars ── */
.progress-container, div.progress-level, div[role="progressbar"], .progress-bar, .meta-text {
    display: none !important;
}

/* ── In-Place Animated Loading Loop ── */
.loader-box {
    margin-top: 14px;
    padding: 14px 18px;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(139, 92, 246, 0.08) 100%);
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-radius: var(--radius-md);
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
    display: flex;
    align-items: center;
    gap: 14px;
    animation: fadeIn 0.25s ease-out;
}

.loader-loop {
    width: 26px;
    height: 26px;
    min-width: 26px;
    border: 3px solid rgba(99, 102, 241, 0.22);
    border-top-color: #a78bfa;
    border-right-color: #6366f1;
    border-radius: 50%;
    animation: spinLoop 0.8s linear infinite;
    box-shadow: 0 0 14px rgba(99, 102, 241, 0.35);
}

@keyframes spinLoop {
    0%   { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.loader-text-group {
    display: flex;
    flex-direction: column;
}

.loader-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #e0e7ff;
    letter-spacing: -0.01em;
}

.loader-desc {
    font-size: 0.76rem;
    color: #9aa3bf;
    margin-top: 2px;
}

.status-success-box {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 14px;
    padding: 12px 14px;
    border-radius: var(--radius-md);
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #6ee7b7;
}

.status-success-title {
    font-size: 0.84rem;
    font-weight: 700;
    color: #6ee7b7;
}

.status-success-desc {
    font-size: 0.75rem;
    color: #a7f3d0;
    margin-top: 1px;
}

.status-error-box {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 14px;
    padding: 12px 14px;
    border-radius: var(--radius-md);
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.35);
    color: #fca5a5;
    font-size: 0.82rem;
    font-weight: 600;
}

#app-header-wrap, .left-panel, .right-panel {
    animation: fadeInUp 0.45s ease forwards;
}
.left-panel  { animation-delay: 0.08s; }
.right-panel { animation-delay: 0.16s; }
"""

def process_handbook(file_obj):
    if file_obj is None:
        err_html = """
        <div class="status-error-box">
            <span>⚠️</span>
            <span>Please upload a course handbook or syllabus file first.</span>
        </div>
        """
        yield err_html, "", gr.update(visible=False), ""
        return

    loader_html_1 = """
    <div class="loader-box">
        <div class="loader-loop"></div>
        <div class="loader-text-group">
            <div class="loader-title">Ingesting Handbook...</div>
            <div class="loader-desc">Extracting document text from uploaded file</div>
        </div>
    </div>
    """
    yield loader_html_1, "", gr.update(visible=False), ""

    file_path = file_obj.name
    try:
        text = extract_text_from_file(file_path)
    except Exception as e:
        err_html = f"""
        <div class="status-error-box">
            <span>❌</span>
            <span>Error extracting text: {e}</span>
        </div>
        """
        yield err_html, "", gr.update(visible=False), ""
        return

    loader_html_2 = """
    <div class="loader-box">
        <div class="loader-loop"></div>
        <div class="loader-text-group">
            <div class="loader-title">Synthesizing with Llama 3.2:3b...</div>
            <div class="loader-desc">Extracting syllabus learning outcomes</div>
        </div>
    </div>
    """
    yield loader_html_2, "", gr.update(visible=True, open=True), "Connecting to Llama 3.2:3b..."

    final_prompt = ""
    try:
        for chunk in stream_learning_outcomes_prompt(text):
            final_prompt = chunk
            yield loader_html_2, final_prompt, gr.update(visible=True, open=True), final_prompt
    except Exception as e:
        err_html = f"""
        <div class="status-error-box">
            <span>❌</span>
            <span>Error generating prompt: {e}</span>
        </div>
        """
        yield err_html, "", gr.update(visible=False), ""
        return

    success_html = """
    <div class="status-success-box">
        <span style="font-size: 1.2rem;">✅</span>
        <div>
            <div class="status-success-title">Curriculum Synchronized!</div>
            <div class="status-success-desc">Virtual Teacher is ready with extracted learning outcomes.</div>
        </div>
    </div>
    """
    yield success_html, final_prompt, gr.update(visible=True, open=True), final_prompt


def extract_text(content) -> str:
    """Normalize Gradio 6 content to a plain string for Ollama."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part["text"] for part in content if isinstance(part, dict) and "text" in part
        )
    return str(content)


def user(user_message, history):
    if not user_message or not user_message.strip():
        return "", history
    return "", history + [{"role": "user", "content": user_message}]


def bot(history, extracted_prompt):
    if not history:
        yield history
        return
    combined_system_prompt = DEV_SYSTEM_PROMPT + "\n\n" + (extracted_prompt or "")
    messages = []
    for msg in history[:-1]:
        messages.append({"role": msg["role"], "content": extract_text(msg["content"])})
    messages.append({"role": "user", "content": extract_text(history[-1]["content"])})
    history = history + [{"role": "assistant", "content": ""}]
    for chunk in stream_chat_response(messages, combined_system_prompt):
        history[-1]["content"] = chunk
        yield history


# Initial placeholder for reports before one is generated
EMPTY_REPORT_MD = """## 📊 Student Progress Report

> 💡 **No assessment generated yet.**
> 
> Chat with **Prof. Nova** in the **💬 Tutoring Studio** tab, then click **"📊 Generate Progress Report"** to produce an intelligent analysis of strengths, focus areas, and learning outcomes!
"""

dark_theme = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "sans-serif"],
).set(
    body_background_fill="#090a10",
    body_text_color="#f3f4fd",
    border_color_primary="rgba(99,102,241,0.16)",
    background_fill_primary="#0f111d",
    background_fill_secondary="#151828",
    color_accent_soft="#171a2d",
    button_primary_background_fill="linear-gradient(135deg,#6366f1,#8b5cf6)",
    button_primary_text_color="#ffffff",
    block_label_text_color="#9aa3bf",
    input_background_fill="#171a2d",
    input_border_color="rgba(99,102,241,0.16)",
)


with gr.Blocks(title="Virtual Teacher AI | Intelligent Curriculum Tutor") as app:
    extracted_system_prompt = gr.State("")

    # ── Top Bar / Header with Graphic Mascot & System Badges ──
    header_html = f"""
    <div id="app-header-wrap">
        <div class="header-left">
            <img class="header-avatar" src="{AVATAR_B64}" alt="Prof. Nova AI Mascot" />
            <div class="header-title-group">
                <h1>🎓 Virtual Teacher Studio</h1>
                <p>AI-Powered Curriculum Guidance & Real-Time Student Mastery Tracking</p>
            </div>
        </div>
        <div class="header-badges">
            <div class="status-chip">
                <span class="status-chip-dot"></span>
                <span>Llama 3.2 3B Connected</span>
            </div>
            <div class="status-chip" style="border-color: rgba(16,185,129,0.3); background: rgba(16,185,129,0.1); color: #6ee7b7;">
                <span>⚡ Ollama Local Engine</span>
            </div>
        </div>
    </div>
    """
    gr.HTML(header_html)

    # ── Main Two-Column Layout (NO equal_height, prevents stretching LHS!) ──
    with gr.Row(equal_height=False, elem_classes="main-content-row"):

        # ── Left Column: Sticky Setup & AI Persona Sidebar ──
        with gr.Column(scale=1, min_width=320, elem_classes="left-panel"):
            
            # Mentor Profile Card
            mentor_card_html = f"""
            <div class="mentor-card">
                <img src="{AVATAR_B64}" alt="Prof. Nova" />
                <div class="mentor-info">
                    <h3>Prof. Nova</h3>
                    <p>Adaptive Socratic AI Mentor</p>
                </div>
            </div>
            """
            gr.HTML(mentor_card_html)

            gr.HTML("<div class='section-badge'>① Handbook Ingestion</div>")
            file_input = gr.File(
                label="Upload Syllabus or Handbook (.pdf, .txt)",
                file_types=[".pdf", ".txt"],
                elem_classes="file-upload-zone",
            )
            
            process_btn = gr.Button(
                "⚡ Extract Learning Objectives",
                elem_classes="btn-uniform btn-primary",
            )
            status_out = gr.HTML("", elem_classes="status-out")

            with gr.Accordion(
                "📋 Extracted Objectives & Teacher Instructions",
                open=False,
                visible=False,
                elem_classes="prompt-accordion",
            ) as prompt_accordion:
                prompt_out = gr.Textbox(
                    label="Curriculum System Prompt",
                    lines=8,
                    interactive=False,
                )

        # ── Right Column: Containerized Workspaces (Tabs) ──
        with gr.Column(scale=2, min_width=500, elem_classes="right-panel"):
            with gr.Tabs(elem_classes="workspace-tabs") as tabs:
                
                # ── Tab 1: Tutoring Studio ──
                with gr.Tab("💬 Tutoring Studio", id="tutoring_tab"):
                    
                    # Quick Prompt Suggestion Pills
                    gr.HTML("<div class='section-badge'>Suggested Explorations</div>")
                    with gr.Row(elem_classes="prompt-chips-container"):
                        chip_1 = gr.Button("🎯 Explain core syllabus objectives", elem_classes="chip-btn")
                        chip_2 = gr.Button("❓ Quiz me on key concepts", elem_classes="chip-btn")
                        chip_3 = gr.Button("💡 Give a practical real-world example", elem_classes="chip-btn")

                    chatbot = gr.Chatbot(
                        height=480,
                        show_label=False,
                        elem_classes="chat-window",
                        layout="bubble",
                        placeholder="<strong>Welcome to your private tutoring studio!</strong><br>Upload a course handbook on the left to initialize Prof. Nova, or ask any question to get started.",
                    )

                    with gr.Row(elem_classes="chat-input-row"):
                        msg_input = gr.Textbox(
                            show_label=False,
                            placeholder="Ask Prof. Nova anything about your curriculum…",
                            scale=8,
                            container=False,
                            elem_classes="msg-textbox",
                            lines=1,
                            max_lines=3,
                        )
                        submit_btn = gr.Button(
                            "Send ➤",
                            scale=2,
                            elem_classes="btn-uniform btn-primary",
                        )

                    # Uniform action toolbar below chat
                    with gr.Row(elem_classes="chat-action-bar"):
                        report_btn = gr.Button(
                            "📊 Generate Progress Report",
                            elem_classes="btn-uniform btn-success",
                            scale=3,
                        )
                        clear_btn = gr.Button(
                            "🗑️ Clear Session",
                            elem_classes="btn-uniform btn-ghost",
                            scale=1,
                        )

                # ── Tab 2: Learning Analytics & Student Report ──
                with gr.Tab("📊 Progress & Analytics", id="report_tab"):
                    
                    # Hero Graphic Banner for Analytics
                    banner_html = f"""
                    <div class="analytics-hero-banner">
                        <img src="{BANNER_B64}" alt="Learning Analytics Banner" />
                        <div class="analytics-banner-overlay">
                            <h2>Student Competency & Mastery Report</h2>
                            <p>Evaluated against course handbook learning outcomes</p>
                        </div>
                    </div>
                    """
                    gr.HTML(banner_html)

                    with gr.Row(elem_classes="report-controls-row"):
                        refresh_report_btn = gr.Button(
                            "🔄 Re-evaluate Transcript",
                            elem_classes="btn-uniform btn-success",
                        )
                        back_to_chat_btn = gr.Button(
                            "💬 Return to Tutoring Studio",
                            elem_classes="btn-uniform btn-ghost",
                        )

                    # Bounded scrollable report card container
                    with gr.Column(elem_classes="report-scroll-container"):
                        report_out = gr.Markdown(EMPTY_REPORT_MD)

    # ── Event Handlers & Dynamic Interactions ──
    
    # Handbook processing with live spinning loop and streaming prompt
    process_btn.click(
        fn=process_handbook,
        inputs=[file_input],
        outputs=[status_out, extracted_system_prompt, prompt_accordion, prompt_out],
        show_progress="hidden",
    )

    # Chat submission
    chat_submit_event = msg_input.submit(
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

    # Quick prompt chips auto-fill input
    chip_1.click(fn=lambda: "Explain the core syllabus objectives in simple terms.", outputs=[msg_input])
    chip_2.click(fn=lambda: "Quiz me on key concepts from the handbook to test my understanding.", outputs=[msg_input])
    chip_3.click(fn=lambda: "Give me a practical real-world example illustrating the first major topic.", outputs=[msg_input])

    # Clear chat session
    clear_btn.click(
        fn=lambda: ([], ""),
        outputs=[chatbot, msg_input],
    )

    # Generate Report from Chat Action Bar:
    def generate_and_switch(history, extracted_prompt):
        report_md = generate_progress_report(history, extracted_prompt)
        return report_md, gr.update(selected="report_tab")

    report_btn.click(
        fn=generate_and_switch,
        inputs=[chatbot, extracted_system_prompt],
        outputs=[report_out, tabs],
        show_progress="hidden",
    )

    # Refresh report inside the analytics tab
    refresh_report_btn.click(
        fn=generate_progress_report,
        inputs=[chatbot, extracted_system_prompt],
        outputs=[report_out],
        show_progress="hidden",
    )

    # Back to Chat button inside analytics tab
    back_to_chat_btn.click(
        fn=lambda: gr.update(selected="tutoring_tab"),
        outputs=[tabs],
    )


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=custom_css,
        theme=dark_theme,
        debug=True,
    )
