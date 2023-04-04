import sys
from textwrap import dedent

import openai


def get_completion(model, prompt, temperature, max_tokens, stop):
    print("*" * 40)
    print("Getting completion for prompt:")
    print(">>>")
    print(prompt)
    print("<<<")
    response = openai.Completion.create(
        model=model, prompt=prompt, temperature=temperature, max_tokens=max_tokens, stop=stop
    )
    result = response["choices"][0]["text"]
    print("Completion is:")
    print(">>>")
    print(result)
    print("<<<")
    print("*" * 40)
    return result


class Message:
    def __init__(self, sender, content):
        self.sender = sender
        self.content = content

    def __str__(self):
        return f"{self.sender}:\n{self.content}"


conversation_history = []


def generate_response(user_message_content):
    conversation_history.append(Message("User", user_message_content))

    prompt = 'The following is the transcript of a chat between "Bot", a chatbot, and "User", a human using it.\n\n'
    for message in conversation_history:
        prompt += str(message) + "\n\n"

    prompt += "Bot:\n"

    bot_message_content = get_completion(
        model="text-davinci-003", prompt=prompt,
        temperature=0.5, max_tokens=30, stop="User:"
    )
    bot_message = Message("Bot", bot_message_content)
    conversation_history.append(bot_message)
    return bot_message_content


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