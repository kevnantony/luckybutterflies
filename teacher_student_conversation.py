from datasets import load_dataset
from ollama import chat

from database import Database

teacher_system_prompt = "You are a helpful teacher."
teacher_system_prompt = (
    "You are a teacher who explains answers and tries to make a student understand a concept in a pedagogically exceptional way rather than just answering the question.\n\n"
    "Give minimal input so the student can learn by itself."
)
student_system_prompt = "You are a high school student trying to learn something"

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


def format_messages(history, current_agent):
    messages = []

    for dict in history:
        if dict["role"] == current_agent:
            role = "assistant"
        else:
            role = "user"

        messages.append(
            {
                "role": role,
                "content": dict["content"],
            }
        )

    return messages


def main():

    db = Database()

    dataset = load_dataset("cais/mmlu", "high_school_biology")

    datasets = {}

    for subject in HIGH_SCHOOL_SUBJECTS:
        print(f"Loading {subject}...")
        datasets[subject] = load_dataset("cais/mmlu", subject)

    for index, dataset in enumerate(datasets.values()):
        print(dataset)
        print(dataset["test"][0])

        question = f"I dont understand this task. Please help me. \nquestion: {dataset['test'][0]['question']}\nchoices: {dataset['test'][0]['choices']}"

        conversation_id = db.create_conversation(index, "Original")
        history = []
        history.append(
            {
                "role": "student",
                "content": question,
            }
        )
        print(question)
        for sequence in range(6):
            if sequence % 2 == 0:
                system_prompt = teacher_system_prompt
                agent = "teacher"
                messages = format_messages(history, "teacher")
            else:
                system_prompt = student_system_prompt
                agent = "student"
                messages = format_messages(history, "student")

            # print(messages)
            response = chat(
                model="llama3.2:3b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *messages,
                ],
            )

            content = response.message.content

            history.append(
                {
                    "role": agent,
                    "content": content,
                }
            )

            db.add_message(
                conversation_id,
                sequence + 1,
                agent,
                content,
            )
            print(response.message.content)

    print("Hello from luckybutterflies!")


if __name__ == "__main__":
    main()
