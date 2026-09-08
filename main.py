from ollama import chat


def main():
    response = chat(
        model="qwen3:8b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
    )

    print(response.message.content)
    print("Hello from luckybutterflies!")


if __name__ == "__main__":
    main()
