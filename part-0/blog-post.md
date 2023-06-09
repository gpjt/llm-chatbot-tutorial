---
slug: ai-llm-bot-beginners-tutorial-00
title: Building an AI chatbot for beginners: part 0
date: 2023-03-19 20:45:00+00:00
state: published
categories: programming, python, ai
format: md
---

Like a lot of people, I've been blown away by the capabilities of Large Language
Model (LLM) based systems over the last few months.  I'm using ChatGPT regularly
for all kinds of things, from generating basic code to debugging errors to writing
emails.

I wanted to understand more about how these tools worked, and feel strongly that
there's no better way to learn something than by doing it.  Building an LLM is, at least
right now, super-expensive -- in the millions of dollars (although
[maybe that will be coming down fast](https://simonwillison.net/2023/Mar/17/beat-chatgpt-in-a-browser/)?).
It also requires a lot of deep knowledge to get to something interesting.
Perhaps something to try in the future, but not right now.

However, *using* LLMs to create something interesting -- that's much easier,
especially because OpenAI have a powerful [API](https://platform.openai.com/overview),
which provides ways to do all kinds of stuff.  Most relevantly, they provide
access to a [Completion API](https://platform.openai.com/docs/api-reference/completions).
That, as I understand it, is the lowest-level way of interacting with an LLM, so
building something out of it is probably the best bang for the buck for learning.

Over the last few weeks I've put together a bunch of things I found interesting,
and have learned a lot.  But it occurred to me that an even better way to learn
stuff than by building it is to build it, and then explain it to someone else,
even if that person is an abstract persona for "someone out there on the Internet".
So: time for a LLM chatbot tutorial!

<!--more-->

I'm writing this addressing myself as I was a few months back, so if it's of
interest to anyone, it'll be for people who know Python reasonably well, but who
haven't spent much or any time coding for LLMs.  If you already know AI or
Natural Language Processing well, you probably won't find anything new here
(though I would be very grateful for any corrections!).

My goal is to keep things as simple as possible.  The bots I've built in the past have been
web-based or use Telegram to interact with the user.  That adds on unnecessary complexity
from the viewpoint of a tutorial, so I'll show how to
build a basic chatbot that runs in a console as a single process.  You'll run

```
python bot.py
```

...and be able to type in your messages. The bot will respond,
using OpenAI as a backend.

So far I've planned out the steps required to go from a basic framework that
bounces stuff off OpenAI, to something that can actually have a conversation, to
something that avoids the obvious forms of
[prompt injection](https://research.nccgroup.com/2022/12/05/exploring-prompt-injection-attacks/),
to something that can have long-running conversations and understand the
context, even if a user's message refers to something they said some time ago.

I'll be posting these as I write them.  Hopefully there are people out there
who'll find them useful!

* [Part 1: connecting to the OpenAI API and creating a bot that replies to messages](/2023/03/ai-llm-bot-beginners-tutorial-01)
* [Part 2: keeping a history of the conversation](/2023/04/ai-llm-bot-beginners-tutorial-02)