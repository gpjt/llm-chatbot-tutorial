import sys


def generate_response(message):
    return "Dummy response"


def main():
    while True:
        print("User: ")

        user_message = ""
        try:
            while True:
                user_message += input() + "\n"
        except EOFError:
            pass

        print()
        response = generate_response(user_message)
        print()

        print("Bot:")
        print(response)
        print()


if __name__ == "__main__":
    main()