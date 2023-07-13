import os
import json
import time
import pandas as pd
import numpy as np
import io
import sys
import requests

import openai
from elevenlabslib import *

from pydub import AudioSegment
from moviepy.editor import *
from moviepy.editor import ImageClip


# STORY GENERATION UTILITY FUNCTIONS

def retrieve_api_key_dict(filepath):
    """ Takes a file path to JSON file and retrieves the API keys in the form of a dictionary
        
        Parameters
        ----------
        filepath : str
            Filepath to JSON file containing API keys
    
        Returns
        api_key_dict : dictionary
            Dictionary containing API keys (E.G api_keys_dict['openai'] would return OpenAI API Key)
        --------
        
    """
    
    # Opening JSON file
    f = open(filepath)
    
    # Returns JSON object as a dictionary
    data = json.load(f)

    # Grab portion relating to API keys
    api_key_dict = data['api_keys']

    return api_key_dict

def convert_to_bool(x, reset=False):
    """ Takes a string "True" or "False" and converts datatype to boolean
        
        Parameters
        ----------
        x : str
            "True" or "False"
        reset : boolean 
            Set to true if you want to reset all row values under "Published" to False
        Returns 
        --------
        True or False : boolean
    """
    # Reset every row to False (optional)
    if reset:
        return False
    
    elif isinstance(x, bool):
      return x
    
    elif x.lower() == "true":
      return True
    
    elif x.lower() == "false":
      return False
    
    else:
       return True

def reset_story_topics_df():  
    # Read the story DataFrame
    story_topics = pd.read_csv("data/story_topics.csv")

    story_topics['Published'] = story_topics['Published'].apply(lambda x: convert_to_bool(x, reset=True))

    # Save updated story_topics DataFrame as CSV, replacing the old one
    story_topics.to_csv("data/story_topics.csv", index=False)


def generate_story_prompt(prompt=None):
    """ Generates story prompt from story_topics DataFrame (either via prompt DataFrame or manually)
        
        Parameters
        ----------
        prompt : str 
            Manually input a prompt to use (optional)

        Returns 
        --------
        user_content: str
            Story prompt

        extracted_topic : str
            Story topic
    """
    if prompt:
        # Assign prompt to topic
        extracted_topic = prompt
        
        # Build prompt using the topic 
        user_content = "Write me the beginning of a 4000 character, realistic scary story told in first-person about the following topic: " + extracted_topic

        return user_content, extracted_topic
    else:

        # Read the story DataFrame
        story_topics = pd.read_csv("data/story_topics.csv")

        story_topics['Published'] = story_topics['Published'].apply(lambda x: convert_to_bool(x))
        
        # Find the index of the next unpublished story topic in the story topics DataFrame
        nxt_story_index = story_topics[story_topics['Published'] == False].index[0]

        # Extract the story topic string from the story topics DataFrame
        extracted_topic = story_topics.iloc[nxt_story_index, 0]

        # Build prompt using the extracted story topic 
        user_content = "Write me the beginning of a 4000 character, realistic scary story told in first-person about the following topic: " + extracted_topic

        # Update "Published" column to True to reflect story has been used
        story_topics.iloc[nxt_story_index, 1] = True

        # Save updated story_topics DataFrame as CSV, replacing the old one
        story_topics.to_csv("data/story_topics.csv", index=False)

        return user_content, extracted_topic


def generate_short_story(user_content):
    """ Takes a text prompt and generates a short story from text prompt
        
        Parameters
        ----------
        user_content : str
            Short story prompt
        Returns 
        --------
        story_text: str
            Text of generated story
    """
    print("Generating story...")
    # Initialize story text
    story_text = ''

    # Generate the beginning of the story
    messages = ([
        {"role": "user", "content": user_content}
    ])
    chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, temperature = 1.15, max_tokens = 450)

    reply = chat.choices[0].message.content
    print(f"ChatGPT: {reply}")
    story_text += (reply)

    # Generate the ending of the story
    messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user", "content": "Great. Now write me the ending of the story."})
    chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, temperature = 1.15, max_tokens = 450)
    reply = chat.choices[0].message.content
    print(f"ChatGPT: {reply}")
    story_text+=(reply)

    return story_text


def generate_n_stories_dict(user_content, n=3):
    """ Takes a text prompt and generates three stories from text prompt
        
        Parameters
        ----------
        user_content : str
            Short story prompt
    
        Returns 
        --------
        story_dict: dictionary
            Dictionary containing each story with keys 'Story 1', 'Story 2', ..., 'Story n' and values as story text
    """
     
    # Generate 3 short stories centered around the story theme
    story_dict = {}
    for i in range(1, n + 1):

        # Generate story text and story title 
        story_text = generate_short_story(user_content)

        # Save output to a dictionary
        story_dict['Story ' + str(i)] = story_text

        # Wait until running next iteration (for API usage limit purposes)
        time.sleep(30)
    
    return story_dict

# STORY CLEANING/CHUNKING UTILITY FUNCTIONS

def remove_newlines_backslashes(string):
    """ Given a string, replace all instances of "\n" and "\" with an empty string.
        
        Parameters
        ----------
        string : str
            The string you want to clean
    
        Returns 
        --------
        cleaned_string : str
            String with all instances of "\n" and "\" removed
    """

    cleaned_string = string.replace('\n\n', ' ')
    cleaned_string = cleaned_string.replace('\\', '')
    return cleaned_string

def chunk_text(text, n=2499):
    """ Given a string, split into chunks of length n
        
        Parameters
        ----------
        text : str
            The text you want to chunk
    
        Returns 
        --------
        chunks : list
            List that contains all chunks
    """

    # Remove last period from text to allow proper replacement
    text = text[0:-1]

    # Split the text into individual sentences.
    sentences = text.split('. ')
    
    # Initialize a variable to keep track of the current chunk of text.
    current_chunk = ''
    
    # Initialize a list to hold all of the chunks of text we generate.
    chunks = []
    
    for sentence in sentences:
        # If the current sentence is longer than the maximum chunk length,
        # split it into chunks of the maximum length and add them to the list.
        if len(current_chunk + sentence) > n:
            # words = current_chunk.split(' ')
            # last_word = words[-1]
            # current_chunk = ' '.join(words[:-1])
            chunks.append(current_chunk)
            current_chunk = sentence + '. '
            #current_chunk = last_word + ' ' + sentence
        # Otherwise, add the sentence to the current chunk of text.
        else:
            current_chunk += sentence + '. '
    
    # Add any remaining text to the list of chunks.
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def clean_chunk_save(story_dict, extracted_topic, n=4999):
    """ Pipeline for cleaning stories inside story_dict. Cleans newlines/backslashes and chunks text into chunks of length n.
        Saves text as .txt files into project directory for future reference. 
        
        Parameters
        ----------
        story_dict : dictionary
            The story dictionary before data cleaning
    
        Returns 
        --------
        story_dict : dictionary
            The story dictionary after data cleaning
    """
    # Define the main filepath
    filepath = "data/" + extracted_topic + "/"

    # Define the text filepath
    text_filepath = "data/" + extracted_topic + "/text/"

    # Make the main directory
    os.mkdir(filepath)

    # Make the text directory
    os.mkdir(text_filepath)

    # Clean, chunk, and save the stories
    for i, story in enumerate(story_dict.keys()):

        # Clean 
        story_text_cleaned = remove_newlines_backslashes(story_dict[story])
        
        # Chunk
        chunks = chunk_text(story_text_cleaned, n)

        # Reassign back to dictionary
        story_dict[story] = chunks

        # Save the story to a .txt file in the new directory
        with open(text_filepath + "Story_" + str(i + 1) + '.txt', 'w') as f:
            # Write the string to the file
            f.write(story_dict[story][0])

    # Save story_dict as JSON
    with open('cleaned_story_dict.json', 'w') as convert_file:
        convert_file.write(json.dumps(story_dict))

    return story_dict

# NARRATION UTILITY FUNCTIONS

def generate_narration(story_dict, api_key, extracted_topic):

    # Define user using API key
    user = ElevenLabsUser(api_key)

    # initialize the voice object
    voice = user.get_voices_by_name("Antoni")[0]

    # Define the filepath
    filepath = "data/" + extracted_topic + "/audio/"

    # Create audio directory to save audio files
    os.mkdir(filepath)

    for story in story_dict.keys():

        # Grab the story title
        story_title = story

        # Grab the text chunks pertaining to the story
        chunks = story_dict[story]

        # Iterate through the story chunks and generate audio
        for i, chunk in enumerate(chunks):
            
            audio = voice.generate_audio_bytes(chunk)
            audio_saved = audio
            audio_saved = AudioSegment.from_file(io.BytesIO(audio_saved), format="mp3")

            story_title_cleaned = story_title.replace(".", "").replace(":", "").replace(" ", "_")
            file_path_and_name = filepath + story_title_cleaned + " - Part " + str(i + 1) + ".mp3"
            audio_saved.export(file_path_and_name, format="mp3")


# IMAGE-GENERATION UTILITY FUNCTIONS

def generate_images(story_dict, extracted_topic, n=4):

    image_object_dict = {}

    # Iterate through each story 
    for story in story_dict.keys():

        # Iterate through picture prompts and create images
        image_object_list = []
        for i in np.arange(1, n + 1):

            pic_prompt = "Scary image about the following topic: " + extracted_topic

            image_object = openai.Image.create(
            prompt=pic_prompt,
            n=1,
            size="1024x1024"
            )
            image_object_list.append(image_object)
            time.sleep(15)
        print(image_object_list)

        # Save the image object list to the associated story's key
        image_object_dict[story] = image_object_list
    return image_object_dict


def save_images(image_object_dict, extracted_topic):
    
    # Make the directory to save the pic files
    os.mkdir("data/" + extracted_topic + "/pics")

    # Iterate through each story 
    for story in image_object_dict.keys():

        # Grab the pic objects associated with this specific story 
        image_object_list = image_object_dict[story]


        # pic_num_within_chunk = 1
        # chunk_num = 1
        for i, image_object in enumerate(image_object_list):
            
            # Specify photo filename
            photo_title = story + "_pic" + str(i + 1)
    
            # Grab photo object data
            photo = image_object['data'][0]
            url = photo['url']
            response = requests.get(url)

            # Check that the response was successful
            if response.status_code == 200:
                # Open a file for writing in binary mode
                with open("data/" + extracted_topic + "/pics/" + photo_title + '.png', 'wb') as f:
                    # Write the contents of the response to the file
                    f.write(response.content)


def generate_images_pipeline(story_dict, extracted_topic, n=4):
    
    image_object_dict = generate_images(story_dict, extracted_topic, n)

    save_images(image_object_dict, extracted_topic)


# VIDEO-GENERATION UTILITY FUNCTIONS

def generate_video(story_dict, extracted_topic):

    # Make the directory to save the pic files
    os.mkdir("data/" + extracted_topic + "/video")

    # Get the screen size (assuming you want the image to fill the entire screen)
    screen_size = (1920, 1080)

    # Grab the scary music file
    scary_music = AudioFileClip("data/music/Cicada KIller - Coyote Hearing.mp3")

    # Set the music volume to 15% of the original volume
    scary_music = scary_music.volumex(0.15)

    # Define the list of clips (text + images + fades)
    clips = []

    # Define the current time to keep track of the current position in the video
    current_time = 0

    # Create each story as a clip one at a time
    for i, story in enumerate(story_dict.keys()):

        # Grab the story title
        story_title = story

        # Define the text clip
        txt = TextClip(story_title.replace(".", ""), font='Arial', fontsize=70, color='white')

        # Set the text clip duration
        txt_duration = 6

        # Set the image directory and get a list of image files
        img_dir = 'data/' + extracted_topic + '/pics'
        img_files = os.listdir(img_dir)

        # Sort the image files
        img_files = sorted(img_files)

        # Add the text clip to the list of clips
        clips.append(txt.set_duration(txt_duration))

        # Add to the current time
        current_time += txt_duration

        previous_audio_end_time = txt_duration

        part_of_story = story_title 

        # Find audio file path
        story_title_cleaned = story_title.replace(".", "").replace(":", "").replace(" ", "_")
        audio_file_path_and_name = "data/" + extracted_topic + "/audio/" + story_title_cleaned + " - Part 1" + ".mp3"

        # Initialize audio clip
        audioclip = AudioFileClip(audio_file_path_and_name)

        # Set the audio clip start time to the end time of the previous audio clip + the fade duration
        audioclip = audioclip.set_start(previous_audio_end_time)

        # Get the duration of the audio clip
        audio_duration = audioclip.duration

        # Find the appropriate image file names for this section
        matched_file_names = [file for file in img_files if part_of_story in file]

        # Calculate the duration of each image clip based on the duration of the audio clip
        img_duration = ((audio_duration) / len(matched_file_names))

        # Create a list of ImageClips for the images in this section
        img_clips = []

        # Loop through the image files and add each image 
        for j, matched_img_file in enumerate(matched_file_names):
            # Load the image and create an ImageClip
            img = ImageClip(os.path.join(img_dir, matched_img_file))

            # Resize the image to match the screen size, while maintaining the aspect ratio
            img_resized = img.resize(screen_size)

            # Center the image in the frame
            img_centered = img_resized.set_position(("center", "center"))

            # Add the image with a fade to black transition to the list of clips
            img_clip = img_centered.set_duration(img_duration)

            img_clips.append(img_clip)

        # Concatenate the list of ImageClips into a single VideoClip
        img_clip = concatenate_videoclips(img_clips, method='compose')

        # Assign the audio to the image clip
        img_clip = img_clip.set_audio(audioclip)

        # Add the image clip to the list of clips
        clips.append(img_clip)

        # Update the current time and previous audio end time
        current_time += audio_duration 
        previous_audio_end_time += audio_duration

    # Concatenate the list of clips into a single CompositeVideoClip
    video = concatenate_videoclips(clips, method='compose')

    # Set black screen time at end
    blackscreen_time = 5

    # Set the duration of the video clip based on the current time
    video = video.set_duration(current_time + blackscreen_time)

    # Get the current audio of the video
    current_audio = video.audio

    # Make the music loop for the duration of the video
    scary_music = afx.audio_loop(scary_music, duration=video.duration)

    # Combine the current audio and the new audio into a composite audio clip
    composite_audio = CompositeAudioClip([current_audio, scary_music])

    # Combine the video and music
    video = video.set_audio(composite_audio)

    # Write the video to a file
    video.write_videofile("data/" + extracted_topic + "/video/" + "output.mp4", fps=24)