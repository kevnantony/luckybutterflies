import os
from datasets import load_dataset
from ollama import chat
from gradio_app.document_parser import extract_text_from_file
from database import Database

teacher_system_prompt = (
    "You are a teacher who helps a student study without directly giving answers but rather "
    "by giving hints and support in a clear and concise manner, and tries to make a student "
    "understand a concept in a pedagogically exceptional way rather than just answering the question."
)

student_system_prompt = "You are a high school student trying to learn something"

additional_knowledge_system_prompt = (
    "You are an expert curriculum designer. Extract the key learning outcomes and objectives from the following course handbook text. "
    "Format your output as a set of instructions for a virtual teacher. For example: 'Your goal is to help the student achieve the following learning outcomes: [List outcomes]'.\n\n"
)


def generate_learning_outcomes_prompt(handbook_text: str) -> str:
    """Uses Ollama to extract learning outcomes and generate a system prompt."""
    prompt = additional_knowledge_system_prompt + f"Course Handbook Text:\n{handbook_text}"
    response = chat(
        model="llama3.2:3b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.message.content


HANDBOOK_DIR = "handbooks"


def build_combined_prompt(file_path: str) -> str:
    text = extract_text_from_file(file_path)
    outcomes = generate_learning_outcomes_prompt(text)
    return teacher_system_prompt + f"\n\nCourse learning outcomes to focus on:\n{outcomes}\n"


HIGH_SCHOOL_SUBJECTS = [
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
]


def subject_to_handbook(subject: str) -> str:
    """Map an MMLU subject key to its handbook file path.
    e.g. 'high_school_computer_science' -> 'handbooks/Computer_Science_Handbook.pdf'
    """
    name = subject.replace("high_school_", "")
    name = "_".join(w.capitalize() for w in name.split("_"))
    return os.path.join(HANDBOOK_DIR, f"{name}_Handbook.pdf")


def format_messages(history, current_agent):
    messages = []
    for turn in history:
        role = "assistant" if turn["role"] == current_agent else "user"
        messages.append({"role": role, "content": turn["content"]})
    return messages


def main():
    db = Database()

    # Load all subject datasets once.
    datasets = {}
    for subject in HIGH_SCHOOL_SUBJECTS:
        print(f"Loading {subject}...")
        datasets[subject] = load_dataset("cais/mmlu", subject)

    for index, (subject, dataset) in enumerate(datasets.items()):
        print(f"\n=== {subject} ===")

        # Build the combined teacher prompt (teacher + extracted learning outcomes).
        handbook_path = subject_to_handbook(subject)
        if os.path.exists(handbook_path):
            combined_teacher_prompt = build_combined_prompt(handbook_path)
        else:
            print(f"[warn] no handbook for {subject} at {handbook_path}, using base teacher prompt")
            combined_teacher_prompt = teacher_system_prompt

        for n in range(2):
            q = dataset["test"][n]
            question = (
                f"I don't understand this task. Please help me.\n"
                f"question: {q['question']}\n"
                f"choices: {q['choices']}"
            )

            conversation_id = db.create_conversation(index * 2 + n, "teacher with additional KG")
            history = [{"role": "student", "content": question}]
            print(question)

            for sequence in range(6):
                if sequence % 2 == 0:
                    system_prompt = combined_teacher_prompt
                    agent = "teacher"
                    messages = format_messages(history, "teacher")
                else:
                    system_prompt = student_system_prompt
                    agent = "student"
                    messages = format_messages(history, "student")

                response = chat(
                    model="llama3.2:3b",
                    messages=[{"role": "system", "content": system_prompt}, *messages],
                )
                content = response.message.content

                history.append({"role": agent, "content": content})
                db.add_message(conversation_id, sequence + 1, agent, content)
                print(f"[{agent}] {content}\n")

    print("Hello from luckybutterflies!")


if __name__ == "__main__":
    main()