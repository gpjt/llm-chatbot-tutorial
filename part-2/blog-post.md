Welcome to the first part of my tutorial in how to build a chatbot using OpenAI's interface to their Large Language Models (LLMs)!  You can read the introduction [here](TBD).

The goal in this post is to write a basic bot script that accepts user input, and just bounces it off an OpenAI LLM.  We'll start with this simple CLI-based code to prompt the user, read a multi-line input (terminated with control-D on an empty line on Linux or Mac, or with a control-Z on an empty line for Windows), and then just present the user with a dummy response.  You can use control-C to exit (on all three platforms).

```python
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
```

I'm assuming that you know Python, so I won't explain this code, it's pretty simple.  The function `generate_response`, is, of course, where all of the interesting stuff will happen anyway.

So now let's connect up to OpenAI!  If you haven't already, you'll need to [sign up on their site](https://platform.openai.com/signup).  As of time of writing, this will give you a US$18 credit for API calls, which should be ample for anything in these tutorials.

The next step is to get an API key.  I'm sure they'll rearrange their site over time, but as I write this, [the page to get one is here](https://platform.openai.com/account/api-keys).  The key will start with `sk-` and then have a long string.  It allows anyone with it to use OpenAI resources and have them billed to your account, so keep it secret.

Next, we have to have a way to call the APIs.  They're REST-based, so would be easy enough to call directly, but OpenAPI have made a Python binding available that we can just install:

```bash
pip install openai
```

Next, we'll add an `import openai` to the start of the file, and then put some code in the `generate_response` function:

```python
def generate_response(message):
    response = openai.Completion.create(
        model="ada", prompt=message, temperature=0.5, max_tokens=30
    )
    return response["choices"][0]["text"]
```

What we're doing is calling the OpenAI API and asking for a completion to the message stored in `message`, using the language model `ada` (their oldest, cheapest one).  The other parameters we'll get into later on.  The response we get back has multiple choices -- you can ask the API for a bunch of alternative responses, so they would come back in that list, but the default is just one response, which is the one we use.

Let's try running that.  In a console, set the environment variable `OPENAI_API_KEY` to your secret key, and run the bot.  Type in a message for it, terminate with ^D (or ^Z on Windows), and...

```
$ export OPENAI_API_KEY=sk-blahblahblah
$ python bot.py
User:
Hello
----------------------------------------
Generating response...
----------------------------------------


Bot:

I am a new user here, and I just wanted to let you know that I received my order today and it is perfect!!

I

User:
```

The response you get will almost certainly be different, but it's pretty much certain to be equally random.  So we got something, but it's not much of a reply.  Maybe we'd do better with a different language model?  `babbage` is the one after `ada`, so change the `model` parameter to use that, and...

```
$ python bot.py
User:
Hello
----------------------------------------
Generating response...
----------------------------------------


Bot:

I am a student in the University of California, Berkeley. I have been interested in computers since I was a child. I have been working on

User:
```

OK, how about `curie`?

```
$ python bot.py
User:
Hello
----------------------------------------
Generating response...
----------------------------------------


Bot:

I have an issue with my iMac.

I bought it with the intention of upgrading the RAM to 4 GB and the hard drive to

User:
```

Maybe we can try jumping ahead, to a GPT 3.5 model.  `text-davinci-003` is described on the OpenAI website like this:

> Can do any language task with better quality, longer output, and consistent instruction-following than the curie, babbage, or ada models

Let's give it a go.

```
$ python bot.py
User:
Hello
----------------------------------------
Generating response...
----------------------------------------


Bot:

Hi there! How are you doing?

User:
```

That's pretty cool!  In fact, I was pretty surprised to get that response, as I didn't think that there was enough code in order to get a coherent response yet.  Unfortunately, it's still not perfect:

```
User:
This is a test
----------------------------------------
Generating response...
----------------------------------------


Bot:

This is a test sentence used to demonstrate how a sentence can be used for testing purposes. A test sentence can be used to check the accuracy of

User:
```

Now, there is a newer model called `gpt-3.5-turbo`, which is the one that underlies ChatGPT.  We could try that, but OpenAI have updated their API for this model so that it actually handles a lot of the interesting stuff underlying implementing chat.  That's useful if you're just trying to bang out a chatbot as efficiently as possible, but less useful if you're trying to learn how these bots work under the hood.

So what's going on here?  The LLM is generating words, but it's not really replying to us except in that first reply from the `text-davinci-003` model.

The core thing to understand about LLMs is that they are trying to complete the text that they've been provided with.  They were trained with a massive amount of text, and "know" that when some text starts with X, it's most likely to continue with Y.  That explains all of the responses from the older models.  As to how the last one knew to respond to hello with a chat-style message, I must admit but I'm unsure, but I suspect that it was trained on a large number of chat transcripts as well as everything on Wikipedia, Reddit, and heaven knows what else.

So if you prompt the LLM with something with one well-known completion (well-known if you studied Shakespeare at school, at least), you get the obvious response (still using `text-davinci-003`):

```
User:
Once more unto
----------------------------------------
Generating response...
----------------------------------------


Bot:

The breach, dear friends, once more;
Or close the wall up with our English dead.
In peace there's nothing so becomes a

User:
```

So what's the difference between an LLM and the auto-complete feature on a phone's keyboard?  Essentially it's just scale.  Your phone is trying to predict the most likely next words from looking back at the previous two or three, whereas the LLM is trying to predict the next *n* words -- where *n* can be in the thousands -- based on the full input that you sent it -- which can be as long as 4,096 "tokens" (for which we can read "words" for now).  It's the same problem, but the scale is so different that it becomes qualitatively different.  You can see how a simple statistical model could work out good solutions to the phone problem, but with the larger-scale problems LLMs are designed to solve, neural networks will billions of parameters -- that is, connections between their neurons -- are required.

Another way to think of it is that in order to suggest the next word based on the three previous ones, you can use statistics.  But in order to suggest the next 4,000 words based on a starting 4,000, then you need to do something that is very much like "thinking".  Whether it's real thinking or not is a really interesting philosophical question, and as such I'm going to completely ignore it.

Anyway, what that all tells us is that if we want to write a chatbot, we're going to need to provide the LLM with a prompt that logically could only be completed with a chatbot-like response.  So let's try this (adding a :

```python
def generate_response(message):
	prompt = dedent(f"""
		The
	""")
    response = openai.Completion.create(
        model="ada", prompt=message, temperature=0.5, max_tokens=30
    )
    return response["choices"][0]["text"]
```

