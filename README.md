# videochatbot

This is a quickly hacked together project for a hackathon.
In this day and age of YouTube and TikTok, everyone has some sort of attention deficit disorder and shrivelled dopamine receptors.
I cannot handle how boring TYPING and READING is, I need a VIDEO.
Well here it is, you can now lipsync a video to the output of an LLM.

## How it works

It's as easy as it sounds.
A langchain chain is used to make a really simple chatbot.
It answers your questions, and it is fed into the microsoft speecht5 TTS model.
This voice is then used with wav2lip to lipsync a video, and you can profit.

## How to use

See the notebook for a quick demo.
The server is a bunch of spaghetti code, ignore it.
If you do use the server, the client is an example of how to connect to it.
To use the server you need an `.env` file like the example in `.env_example`.