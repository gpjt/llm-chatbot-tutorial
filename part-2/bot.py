import sys
from textwrap import dedent

import openai


def generate_response(message):
    prompt = dedent(f"""
        The following is the transcript of a chat between "Bot", a chatbot, and "User", a human using it.

        User:
        {message}

        Bot:
    """)
    response = openai.Completion.create(
        model="text-davinci-003", prompt=prompt, temperature=0.5, max_tokens=30, stop="User:"
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

        response = generate_response(user_message)

        print("Response received...")
        print("-" * 40)

        print("Bot:")
        print(response)
        print()

        print("-" * 40)
        print("...response ends")
        print("-" * 40)


if __name__ == "__main__":
    main()