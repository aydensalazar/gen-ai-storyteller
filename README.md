# Python Generative AI Video Creator 🦇

👨🏾‍💻 Author: Ayden Salazar, AI Engineer

See the Youtube channel (100% AI-generated content) in action: https://www.youtube.com/@FangedFrights

## Goal 🎯
The goal of this project is to build an AI pipeline that generates scary storytelling videos for Youtube. The videos should follow a standardized format:

- A collection of 3 scary stories narrated in human-realistic voice
- Pictures that relate to the story
- Background music that adds to creepy atmosphere
- Title cards that introduce the beginning of each story

## AI Pipeline 🤖

Necessary components to generate video:

1. Story Generation - ChatGPT
2. Speech-to-Text - ElevenLabs
3. Image Generation - DALL·E 2
4. Video editing - MoviePy

## Repository Contents 📂

- `3_Part_Story_Pipeline.ipynb` - Python notebook which can be run to generate a scary storytelling video. A topic can be either automatically or manually entered. Running these functions in series should produce a video. 
- `3_Part_Story_Pipeline_Demo.ipynb` - Presentation notebook for presenting the step-by-step breakdown of the code (not intended for actual video creation purposes)
- `storytelling_utils.py` - Python file that contains all the utility functions used to generate a story. 
- `api_keys.json` - used to store API keys for usage in notebook; notebook code will not run until you enter your own api keys for OpenAI and ElevenLabs! 
- `data/story_topics.csv` - contains a list of pre-populated story topics to allow user to automatically choose a topic rather than manually entering one 
- `data/music/` - contains background music assets for the video

## How to Generate a Scary Storytelling Video 🚀
To use this code repository, follow these steps:

1. Clone the repository to your local machine.
2. Ensure you have Python installed along with required libraries mentioned in the `requirements.txt` file.
3. Retrieve your `OpenAI` and `ElevenLabs` API keys and place inside the `api_keys.json`. 
4. Run every cell in the entire notebook `3_Part_Story_Pipeline.ipynb`.
5. Congrats! You now have a 100% AI generated video that is Youtube-ready! 😁

## Resources 🔗
- [Watch the Youtube Channel in Action](https://www.youtube.com/@FangedFrights)
- [How to get an OpenAI API Key](https://www.maisieai.com/help/how-to-get-an-openai-api-key-for-chatgpt)
- [How to get an ElevenLabs API Key](https://github.com/AndrewCPU/elevenlabs-api)
- [ChatGPT in Python for Beginners](https://www.youtube.com/watch?v=pGOyw_M1mNE&t=139s&ab_channel=TheAIAdvantage)






