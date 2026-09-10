import ollama
'''
#this should be moved to teacher_student_conversation.py a function that generates a suitable prompts based on dropped file

def generate_learning_outcomes_prompt(handbook_text: str) -> str:
    """Uses Ollama to extract learning outcomes and generate a system prompt."""
    prompt = (
        "You are an expert curriculum designer. Extract the key learning outcomes and objectives from the following course handbook text. "
        "Format your output as a set of instructions for a virtual teacher. For example: 'Your goal is to help the student achieve the following learning outcomes: [List outcomes]'.\n\n"
        f"Course Handbook Text:\n{handbook_text}"
    )

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.message.content
    '''

def stream_learning_outcomes_prompt(handbook_text: str):
    """Uses Ollama to extract learning outcomes and stream tokens as they arrive."""
    prompt = (
        "You are an expert curriculum designer. Extract the key learning outcomes and objectives from the following course handbook text. "
        "Format your output as a set of instructions for a virtual teacher. For example: 'Your goal is to help the student achieve the following learning outcomes: [List outcomes]'.\n\n"
        f"Course Handbook Text:\n{handbook_text}"
    )

    response_stream = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    partial_text = ""
    for chunk in response_stream:
        content = chunk.message.content
        if content:
            partial_text += content
            yield partial_text

def stream_chat_response(messages: list, system_prompt: str):
    """Streams the chat response from Ollama, prepending the system prompt.
    
    `messages` must be a list of {"role": ..., "content": str} dicts.
    """
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    response_stream = ollama.chat(
        model="llama3.2:3b",
        messages=full_messages,
        stream=True,
    )

    partial_message = ""
    for chunk in response_stream:
        content = chunk.message.content
        if content:
            partial_message += content
            yield partial_message


def generate_progress_report(history: list, extracted_prompt: str) -> str:
    """Analyses the full chat history and produces a structured student progress report.

    Args:
        history: List of {role, content} dicts from the Gradio chatbot state.
        extracted_prompt: The learning-objectives system prompt extracted from the handbook.

    Returns:
        A markdown-formatted progress report string.
    """
    if not history:
        return (
            "## 📊 Student Progress Report\n\n"
            "> ⚠️ **No conversation yet.** "
            "Start chatting with the Virtual Teacher first, then generate a report."
        )

    # Flatten history into a readable transcript for the LLM
    transcript_lines = []
    for msg in history:
        role = msg.get("role", "")
        raw_content = msg.get("content", "")
        # Handle Gradio 6 rich-content lists
        if isinstance(raw_content, list):
            content = " ".join(
                part["text"] for part in raw_content
                if isinstance(part, dict) and "text" in part
            )
        else:
            content = str(raw_content)
        label = "Student" if role == "user" else "Teacher"
        transcript_lines.append(f"**{label}**: {content}")

    transcript = "\n\n".join(transcript_lines)

    learning_objectives = extracted_prompt.strip() if extracted_prompt.strip() else (
        "No specific learning objectives were provided."
    )

    prompt = (
        "You are an expert educational assessor. "
        "Based on the following chat transcript between a student and a virtual teacher, "
        "generate a concise, structured progress report for the student.\n\n"
        f"### Learning Objectives\n{learning_objectives}\n\n"
        f"### Chat Transcript\n{transcript}\n\n"
        "Produce the report in the following markdown format EXACTLY:\n\n"
        "## 📊 Student Progress Report\n\n"
        "**Session Summary**: (1–2 sentence overview of what was discussed)\n\n"
        "### ✅ Strengths\n"
        "- (bullet points of what the student did well)\n\n"
        "### 🔧 Areas for Improvement\n"
        "- (bullet points of concepts the student struggled with or needs to revisit)\n\n"
        "### 📈 Overall Understanding\n"
        "(A short paragraph rating and explaining the student's grasp of the material)\n\n"
        "### 💡 Recommendations\n"
        "- (bullet points of specific next steps or resources)\n\n"
        "Be honest, constructive, and encouraging. Base your assessment strictly on the transcript."
    )

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {"role": "system", "content": "You are an expert educational assessor."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.message.content
