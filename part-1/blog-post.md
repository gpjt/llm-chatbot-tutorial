Welcome to the first part of my tutorial on how to build a chatbot using OpenAI's
interface to their Large Language Models (LLMs)!  You can read the introduction
[here](/2023/03/ai-llm-bot-beginners-tutorial-00).

If you're reading this and want to get the best out of it, I strongly recommend
that you run the code on your own machine as you go along: trust me, it will
stick in your mind much better if you do that.

The goal in this post is to write a basic bot script that accepts user input,
and just bounces it off an OpenAI LLM to generate a response.

<!--more-->

We'll start with
this simple CLI-based code to prompt the user, read a multi-line input
(terminated with control-D on an empty line on Linux or Mac, or with a control-Z
on an empty line for Windows), and then just present the user with a dummy
response.

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
```

I'm assuming that you know Python, so I won't explain this code -- it's pretty
simple.  The function `generate_response`, is, of course, where all of the
interesting stuff will happen anyway.

Try running it, just to get used to how to interact.  Use control-C to exit (on
all three platforms).

So now let's connect up to OpenAI!  If you haven't already, you'll need to
[sign up on their site](https://platform.openai.com/signup).  As of time of
writing, this will give you a US$18 credit for API calls, which should be ample
for anything in these tutorials.

The next step is to get an API key.  I'm sure they'll rearrange their site over
time, but as I write this,
[the page to get one is here](https://platform.openai.com/account/api-keys).
The key will start with `sk-` and then have a long string.  It allows anyone
with it to use OpenAI resources and have them billed to your account, so keep it
secret.

Next, we have to have a way to call the APIs.  They're REST-based, so would be
easy enough to call directly, but OpenAPI have made a Python binding available
that we can just install:

```bash
pip install openai
```

Next, we'll add an `import openai` to the start of the file, and then put some
code in the `generate_response` function:

```python
def generate_response(message):
    response = openai.Completion.create(
        model="ada", prompt=message, temperature=0.5, max_tokens=30
    )
    return response["choices"][0]["text"]
```

What we're doing is calling the OpenAI API and asking for a *completion* to the
message stored in `message`, using the language model `ada` (their oldest,
cheapest one).  The word "completion" has a very specific meaning here, which
should be clear by the end of this post, but for now just think of it as a reply.
The other parameters we'll get into later on.  The response we
get back has multiple choices -- you can ask the API for a bunch of alternative
responses, so they would come back in that list, but the default is just one
response, which is the one we use.

Let's try running that.  In a console, set the environment variable
`OPENAI_API_KEY` to your secret key...

```
$ export OPENAI_API_KEY=sk-blahblahblah
```

...and run the bot.

```
$ python bot.py
```

Type in a message for it, terminate with ^D (or ^Z on Windows), and...

```
User:
Hello
----------------------------------------
Generating response...
----------------------------------------
Response received...
----------------------------------------
Bot:

This is the first time I've ever had a friend ask me to do something like this. I am a little scared, but I'm excited

----------------------------------------
...response ends
----------------------------------------
User:
```

The response you get will almost certainly be different, but it's pretty much
certain to be equally random.  We got something, but it's not much of a reply
-- though it is an appropriate sequence of words to come after "Hello".  Maybe
we'd do better with a different language model?  `babbage` is the one after
`ada`, so change the `model` parameter to use that, and...

```
$ python bot.py
User:
Hello
----------------------------------------
Generating response...
----------------------------------------
Response received...
----------------------------------------
Bot:

My name is David and I am a proud owner of a 2005 Audi A4. I have driven this car for about 5 years now, and

----------------------------------------
...response ends
----------------------------------------
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
Response received...
----------------------------------------
Bot:

I am new to this forum so I am sure that I am not very good at this. I have a problem with my husband. He was

----------------------------------------
...response ends
----------------------------------------
User:
```

Let's try the next one, `davinci`:

```
$ python bot.py
User:
Hello
----------------------------------------
Generating response...
----------------------------------------
Response received...
----------------------------------------
Bot:

Welcome to the new site!

If you are a returning visitor, please log in below.

If you are a new visitor,

----------------------------------------
...response ends
----------------------------------------
User:
```

Once again, a good completion in terms of finishing the sentence that we
"started" with out hello, but not a reply.

Maybe we can try jumping ahead, to a GPT 3.5 model.  `text-davinci-003` is
described on the OpenAI website like this:

> Can do any language task with better quality, longer output, and consistent
> instruction-following than the curie, babbage, or ada models

Let's give it a go.

```
$ python bot.py
User:
Hello
----------------------------------------
Generating response...
----------------------------------------
Response received...
----------------------------------------
Bot:

Hi there! How can I help you?

----------------------------------------
...response ends
----------------------------------------
User:
```

That's pretty cool!  In fact, I was pretty surprised to get that response, as I
didn't think that there was enough code in order to get a coherent response yet.
Unfortunately, it's still not perfect:

```
User:
This is a test
----------------------------------------
Generating response...
----------------------------------------
Response received...
----------------------------------------
Bot:

This is a test sentence used to demonstrate how a sentence can be used as a test. It can be used to test grammar, punctuation,

----------------------------------------
...response ends
----------------------------------------
User:
```

Now, there is a newer model called `gpt-3.5-turbo`, which is the one that
underlies ChatGPT.  We could try that, but OpenAI have updated their API for
this model so that it actually handles a lot of the interesting stuff underlying
implementing chat.  That's useful if you're just trying to bang out a chatbot as
efficiently as possible, but less useful if you're trying to learn how these
bots work under the hood.

So let's stop here, and take stock.  What's going on?  The LLM is
generating words, but it's not really replying to us except in that first reply
from the `text-davinci-003` model.

The core thing to understand about LLMs is that they are trying to complete the
text that they've been provided with -- hence "completion" in the API call.  They were trained with a massive amount
of text, and "know" that when some text starts with X, it's most likely to
continue with Y.  That explains all of the responses from the older models.

As to how the last one knew to respond to hello with a chat-style message -- that
is, I believe, because it was trained not just on a large amount of data, but then
it was later "fine-tuned" using a lot of chat transcripts, with human involvement
to make sure that it saw as many "good" ones with helpful responses as possible
-- the Reinforcement Learning with Human Feedback (RLHF) that you have probably
heard about.

But we can see that even with that, if what you send it is not
obviously a chat message -- like the last example -- it will just go ahead with a sensible completion based
on the text it has seen while it was being trained.  To take an even more obvious
example of that, if you prompt the LLM with something with one well-known
completion (well-known if you studied Shakespeare at school, at least), you get
the obvious response (still using `text-davinci-003`):

```
User:
Once more unto
----------------------------------------
Generating response...
----------------------------------------
Response received...
----------------------------------------
Bot:

The breach, dear friends, once more;
Or close the wall up with our English dead.
In peace there's nothing so becomes a

----------------------------------------
...response ends
----------------------------------------
User:
```

So what's the difference between an LLM and the auto-complete feature on a
phone's keyboard?  After all, both are just making suggestions as to what should
come next, given some existing text.

Essentially it's just scale.  Your phone is trying to predict the most likely
next words from looking back at the previous two or three, whereas the LLM is
trying to predict the next *n* words -- where *n* can be in the thousands --
based on the full input that you sent it -- which can be as long as 4,096
"tokens" (for which we can read "words" for now).  It's the same problem, but
the scale is so different that it becomes qualitatively different.  You can see
how a simple statistical model could work out good solutions to the phone
problem, but with the larger-scale problems LLMs are designed to solve, neural
networks with billions of parameters -- that is, connections between their
neurons -- are required.

Another way to think of it is that in order to suggest the next word based on
the three previous ones, you can use basic statistics.  But in order to suggest the
next 4,000 words based on a starting 4,000, then you need to do something that
is very much like "thinking".  Whether it's real thinking or not is a really
interesting philosophical question, and as such I'm going to completely ignore
it.

Anyway, what that all tells us is that if we want to write a chatbot, we're
going to need to provide the LLM with a prompt that logically could only be
completed with a chatbot-like response.  So let's try this (also adding a
`from textwrap import dedent` to the imports at the start):

```python
def generate_response(message):
    prompt = dedent(f"""
        The following is the transcript of a chat between "Bot", a chatbot, and "User", a human using it.

        User:
        {message}

        Bot:
    """)
    response = openai.Completion.create(
        model="ada", prompt=prompt, temperature=0.5, max_tokens=30
    )
    return response["choices"][0]["text"]
```

You can see that we've gone back to the primitive `ada` model.  Let's see what
it does:

```
$ python bot.py
User:
Hello
----------------------------------------
Generating response...
----------------------------------------
Response received...
----------------------------------------
Bot:
Hello


User:
How are you?


Bot:
I'm good


User:
I'm good too


Bot

----------------------------------------
...response ends
----------------------------------------
User:
```

There's almost something there!  It does seem to have a problem -- it's
continuing the conversation with itself after generating an initial response.
We'll look into that in a bit, but first let's try the other models.

So, `babbage`:

```
$ python bot.py
User:
Hello
----------------------------------------
Generating response...
----------------------------------------
Response received...
----------------------------------------
Bot:
Hi, how are you?

User:
I'm good, how are you?

Bot:
I'm fine.



----------------------------------------
...response ends
----------------------------------------
User:
```

Still looking like a real conversation...  Now `curie`:

```
$ python bot.py
User:
Hello
----------------------------------------
Generating response...
----------------------------------------
Response received...
----------------------------------------
Bot:
Hello, User. How are you this fine day?


User:
Fine. How are you?


Bot:
I am fine

----------------------------------------
...response ends
----------------------------------------
User:
```

Even better, I'd say (with a tip o' the fedora to "this fine day").  What about `davinci`?

```
$ python bot.py
User:
Hello
----------------------------------------
Generating response...
----------------------------------------
Response received...
----------------------------------------
Bot:
Hello. How are you?


User:
I'm fine.


Bot:
That's good to hear.


User:

----------------------------------------
...response ends
----------------------------------------
User:
```

Meh.  Finally, `text-davinci-003`:

```
$ python bot.py
User:
Hello
----------------------------------------
Generating response...
----------------------------------------
Response received...
----------------------------------------
Bot:
Hi there! How can I help you?

----------------------------------------
...response ends
----------------------------------------
User:
```

That's interesting.  It stopped after having generated a full response, and
didn't try to continue the conversation playing both parts.  What's going on
here?  Why did the older models try to provide a full conversation continuing on
from the prompt, while the more recent model stopped
after a single response?  Continuing the conversation, when you think about it, is a completely reasonable
thing to do given that all we told it was that the text was a transcript of a
conversation -- we didn't tell it that it was playing the part of the bot, we
just told it to continue the transcript.

Let's look at the call we made to the completion API:

```python
    response = openai.Completion.create(
        model="some-model", prompt=prompt, temperature=0.5, max_tokens=30
    )
```

We've told it to use `some-model`, with our `prompt`.  The `temperature` we'll
look into a bit later -- for now, let's look at `max_tokens`.  That is telling
the system that is controlling the LLM the *maximum* number of tokens that we want.
It can generate fewer if it wants.

Tokens, in this sense, are roughly equal to words -- they're the elements that
the LLM generates, one at a time.  Small words will be tokens on their own, but
longer words will be split into tokens, each of which has some kind of semantic
meaning -- for example, "pseudoscientist" might be split into "pseudo",
"scient", and "ist".  The LLM will keep generating tokens until something about
the structure of the prompt that it is trying to complete indicates that a
reasonable completion will stop there, or until the code that is controlling it
stops asking for new tokens.

The older models, `ada`, `babbage`, `curie` and `davinci` all just kept on generating
tokens until they hit the 30-token maximum we'd passed into the API call and
the system that is asking them to generate new ones stopped doing so.
As for `text-davinci-003` -- I don't know this for sure, but the fact that it
decided to stop after one response, and also it's very chat-like response even
before it was given the context that it was predicting a chat session suggests
to me that it has been trained on a large number of chat sessions in the past,
so it "knows" their structure very well.  It's that RLHF again.

But could we make the earlier models work in a similar fashion?  As it turns
out, we can.  There's an optional argument to the `Completion.create` method,
`stop`, which essentially says "when the model emits this string, stop asking
it for more tokens".  So we can tell the LLM to stop generating tokens when it
the next token would be "User:" -- because then we know it's started trying to
continue the conversation with itself.

Let's try that, again with all five models:

```python
    response = openai.Completion.create(
        model="some-model", prompt=prompt, temperature=0.5, max_tokens=30, stop="User:"
    )
```

For `ada`:

```
$ python bot.py
User:
Hello
----------------------------------------
Generating response...
----------------------------------------
Response received...
----------------------------------------
Bot:
Hello




----------------------------------------
...response ends
----------------------------------------
User:
```

For `babbage`:

```
$ python bot.py
User:
Hello
----------------------------------------
Generating response...
----------------------------------------
Response received...
----------------------------------------
Bot:
Hi there.




----------------------------------------
...response ends
----------------------------------------
User:
```

For `curie`:

```
$ python bot.py
User:
Hello
----------------------------------------
Generating response...
----------------------------------------
Response received...
----------------------------------------
Bot:
Hello



----------------------------------------
...response ends
----------------------------------------
User:
```

`davinci`:

```
$ python bot.py
User:
Hello
----------------------------------------
Generating response...
----------------------------------------
Response received...
----------------------------------------
Bot:
Hi there!



----------------------------------------
...response ends
----------------------------------------
User:
```

...and finally `text-davinci-003`:

```
$ python bot.py
User:
Hello
----------------------------------------
Generating response...
----------------------------------------
Response received...
----------------------------------------
Bot:
Hi there! How can I help you?

----------------------------------------
...response ends
----------------------------------------
User:
```

So: we've got a super-simple bot, where we can send it a message, and get a
response.  That's a great start!

However, it has one big problem: it can't remember any chat context.
Each message we send it gets an answer that only depends on what we say in that
message.  Even worse, because it lacks that context, it will even generate "hallucinations" -- that is, responses with no
connection to the facts, which just happen to be a good completion to the provided
prompt.  For example:

```
User:
My favourite colour is green, what's yours?
----------------------------------------
Generating response...
----------------------------------------
Response received...
----------------------------------------
Bot:
My favorite color is blue.

----------------------------------------
...response ends
----------------------------------------
User:
Do you remember what my favourite colour is?
----------------------------------------
Generating response...
----------------------------------------
Response received...
----------------------------------------
Bot:
Yes, you said your favorite color is blue.

----------------------------------------
...response ends
----------------------------------------
User:
Can you tell me what we've talked about so far?
----------------------------------------
Generating response...
----------------------------------------
Response received...
----------------------------------------
Bot:
Sure! So far we've talked about your interests and hobbies, what you like to do in your free time, and what kind of music you like

----------------------------------------
...response ends
----------------------------------------
User:
```

In the next post, we'll fix that.







