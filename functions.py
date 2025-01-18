from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from moviepy import VideoFileClip, AudioFileClip, concatenate_audioclips
from dotenv import load_dotenv
load_dotenv()
import os
import moviepy as mp
import whisper  
import subprocess
from langdetect import detect
from gtts import gTTS
from pydub import AudioSegment


def extract_audio(video_path, audio_output_path):
    
    try : 
        command = [
            "ffmpeg",
            "-i", video_path,
            "-q:a", "0",  # Highest quality
            "-map", "a",  # Only process audio                                       
            audio_output_path           
        ]
        subprocess.run(command, check=True, text=True, encoding='utf-8')
        print(f"Audio successfully extracted to {audio_output_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


def transcribe_audio(audio_path, model_size="base"):
    try:
        model = whisper.load_model(model_size)
        result = model.transcribe(audio_path)
        return result["text"] 

    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def translate_to_text(transcription_text, translated_text_path, target_language):
    target_lang = target_language
    prompt = """"
    Trasnlate the following text file to {target_lang}
    Text in File : {input_text}
    No need for further explanation just give translation to the given te   xt
    Translation :
    """
    
    llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")

    prompt_template = PromptTemplate(template = prompt, input_variables = [ "target_lang", "input_text"])
    
    chain = prompt_template | llm | StrOutputParser()

    try :
        with open(transcription_text, 'r', encoding='utf-8') as file:
            input_text = file.read().strip()
    except FileNotFoundError:
        print("File not found")
    
    output = chain.invoke({"target_lang": target_lang, "input_text": input_text})
    
    with open(translated_text_path, "w", encoding='utf-8') as file:
        file.write(output)


def text_to_speech(translated_text_path, language):

    try:

        gtts_language_map = {
            "en": "English",  # English
            "fr": "French",  # French
            "es": "Spanish",  # Spanish
            "de": "German",  # German
            "zh-cn": "Chinese",  # Chinese (Simplified)
            "hi": "Hindi",  # Hindi
        }

        gtts_lang = gtts_language_map.get(language)
        if not gtts_lang:
            raise ValueError(f"Unsupported language for gTTS: {language}")

        # Read the translated text file
        with open(translated_text_path, 'r', encoding='utf-8') as file:
            text = file.read()
            print(f"Text for TTS: {text}")  # Debug: Check text for TTS

        # Generate speech using gTTS
        tts = gTTS(text=text, lang=language)
        output_file = os.path.join("static/outputs", "translated_audio.mp3")
        tts.save(output_file)

        # Validate that the audio file was created
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            raise ValueError("gTTS failed to generate a valid audio file.")

        print(f"Audio file saved as {output_file}")  # Debug: Confirm file saved
    except Exception as e:
        print(f"Error in text_to_speech: {e}")
        raise



from pydub import AudioSegment
import subprocess

def integrate_audio_to_video(speech_audio_path, video_path, final_video_path):
    
    command = ['ffmpeg', '-i', video_path]
    result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    duration=0
    # Find the duration in the FFmpeg output
    for line in result.stderr.splitlines():
        if "Duration" in line:
            duration_str = line.split("Duration:")[1].split(",")[0].strip()
            hours, minutes, seconds = map(float, duration_str.split(":"))
            duration =  hours * 3600 + minutes * 60 + seconds

    # Load the audio using Pydub
    audio = AudioSegment.from_file(speech_audio_path)

    # Extract video and audio durations
    video_duration = duration
    audio_duration = len(audio) / 1000.0  # Convert from milliseconds to seconds

    # Adjust the audio duration by trimming or padding it
    if audio_duration < video_duration:
        # Pad the audio with silence if it's shorter than the video
        silence = AudioSegment.silent(duration=(video_duration - audio_duration) * 1000)
        audio = audio + silence
    elif audio_duration > video_duration:
        # Trim the audio if it's longer than the video
        audio = audio[:int(video_duration * 1000)]  # Keep only as much audio as the video length

    # Export the adjusted audio to a temporary file in WAV format (FFmpeg works well with WAV)
    temp_audio_path = "temp_audio.wav"
    audio.export(temp_audio_path, format="wav")

    # Use FFmpeg to integrate the audio with the video and synchronize
    command = [
        'ffmpeg',
        '-i', video_path,          # Input video file
        '-i', temp_audio_path,     # Input audio file
        '-c:v', 'copy',            # Copy the video codec
        '-c:a', 'aac',             # Use AAC codec for audio
        '-strict', 'experimental', # Allow experimental features (for audio codec)
        '-map', '0:v:0',           # Map the video stream
        '-map', '1:a:0',           # Map the audio stream
        '-shortest',               # Ensure the output video duration matches the shortest stream
        final_video_path           # Output video path
    ]
    
    # Run the command
    subprocess.run(command)

    # Optionally, delete the temporary audio file
    subprocess.run(["rm", temp_audio_path])

    print(f"Video with integrated audio (synchronized) saved at {final_video_path}")















