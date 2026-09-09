import ollama

def generate_learning_outcomes_prompt(handbook_text: str) -> str:
    """Uses Ollama to extract learning outcomes and generate a system prompt."""
    prompt = (
        "You are an expert curriculum designer. Extract the key learning outcomes and objectives from the following course handbook text. "
        "Format your output as a set of instructions for a virtual teacher. For example: 'Your goal is to help the student achieve the following learning outcomes: [List outcomes]'.\n\n"
        f"Course Handbook Text:\n{handbook_text}"
    )

    response = ollama.chat(
        model="qwen2.5:14b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.message.content

def stream_chat_response(messages: list, system_prompt: str):
    """Streams the chat response from Ollama, prepending the system prompt.
    
    `messages` must be a list of {"role": ..., "content": str} dicts.
    """
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    response_stream = ollama.chat(
        model="qwen2.5:14b",
        messages=full_messages,
        stream=True,
    )

    partial_message = ""
    for chunk in response_stream:
        content = chunk.message.content
        if content:
            partial_message += content
            yield partial_message
