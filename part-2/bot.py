import sys

import openai


def generate_response(message):
    response = openai.Completion.create(
        model="text-davinci-003", prompt=message, temperature=0.5, max_tokens=30
    )
    return response["choices"][0]["text"]


def main():
    while True:
        print("User: ")

        user_message = ""
        try:
            while True:
                user_message += input() + "\n"
        except EOFError:
            pass

        print("-" * 40)
        print("Generating response...")
        print("-" * 40)

        print()
        response = generate_response(user_message)
        print()

        print("Bot:")
        print(response)
        print()


if __name__ == "__main__":
    main()