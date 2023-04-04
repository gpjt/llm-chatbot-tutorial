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


USER_INTRO = "051db2a5-b725-4b14-86c3-bce9f207834f"
BOT_INTRO = "4ce7886e-1500-4f45-b5be-3253c75872aa"


class Message:
    def __init__(self, sender, content):
        self.sender = sender
        self.content = content

    def __str__(self):
        if self.sender == "User":
            intro = USER_INTRO
        elif self.sender == "Bot":
            intro = BOT_INTRO
        else:
            raise Exception(f"Unknown sender {self.sender}")
        return f"{intro}:\n{self.content}"


conversation_history = []


def generate_response(user_message_content):
    conversation_history.append(Message("User", user_message_content))

    prompt = dedent(f"""
        The following is the transcript of a chat between a chatbot and a human using it.

        The user's messages start with the following text: {USER_INTRO}

        The bot's messages start with the following text: {BOT_INTRO}

    """)
    for message in conversation_history:
        prompt += str(message) + "\n\n"

    prompt += f"{BOT_INTRO}:\n"

    bot_message_content = get_completion(
        model="text-davinci-003", prompt=prompt,
        temperature=0.5, max_tokens=30, stop=USER_INTRO
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