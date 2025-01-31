from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import whisper
import subprocess
from pydub import AudioSegment
import json
import edge_tts
import asyncio
load_dotenv()

def extract_audio(video_path, audio_output_path):
    try:
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

def transcribe_audio_with_timestamps(audio_path, model_size="base"):
    """
    Transcribe audio and return both text and timestamp information
    """
    try:
        model = whisper.load_model(model_size)
        result = model.transcribe(audio_path)

        transcription_file = "static/outputs/transcribed_text.txt"
        if "text" in result and result["text"]:
            with open(transcription_file, "w", encoding='utf-8') as f:
                f.write(result["text"])  # Extract text before writing
            print(f"Transcription saved to {transcription_file}")
        else:
            print("Transcription failed.")

        return result  # Returns full result including segments with timestamps
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def generate_srt_from_segments(segments, translated_text=None):
    """
    Generate SRT format subtitles from whisper segments
    """
    srt_content = ""
    for i, segment in enumerate(segments, 1):
        start = format_timestamp(segment['start'])
        end = format_timestamp(segment['end'])
        text = segment['text'].strip()
        
        srt_content += f"{i}\n{start} --> {end}\n{text}\n\n"
    
    return srt_content

def format_timestamp(seconds):
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def translate_to_text(transcription_result, translated_text_path, target_language):
    target_lang = target_language
    # Modified prompt to preserve timing information
    prompt = """Translate these text segments to {target_lang} EXACTLY AS IS:
    - Maintain EXACT same number of lines
    - Preserve ALL punctuation and timing markers
    - NO additional text or explanations
    - ONE translated segment per line
    - STRICT FORMAT: [translation] (NO numbering)
    
    Input Segments:
    {input_text}
    
    Translations (ONLY TRANSLATIONS, ONE PER LINE):"""
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    prompt_template = PromptTemplate(template=prompt, input_variables=["target_lang", "input_text"])
    chain = prompt_template | llm | StrOutputParser()
    
    # Extract segments and create input text
    segments_text = "\n".join([segment['text'] for segment in transcription_result['segments']])
    
    output = chain.invoke({"target_lang": target_lang, "input_text": segments_text})
    
    # Save translated text
    with open(translated_text_path, "w", encoding='utf-8') as file:
        file.write(output)
    
    return output.split('\n')  # Return translated segments

async def generate_speech(text, language, output_file):
    try:
        voice_map = {
            "en": "en-US-AriaNeural",      # More expressive female (great inflection)
            "fr": "fr-FR-BrigitteNeural",  # Warmer French tone
            "es": "es-ES-AlvaroNeural",    # Natural male voice with good pacing
            "de": "de-DE-AmalaNeural",     # Softer German articulation
            "zh-cn": "zh-CN-YunyangNeural",# More emotive Chinese voice
            "hi": "hi-IN-MadhurNeural",    # Natural male Hindi voice
            "af": "af-ZA-WillemNeural",    # Smoother Afrikaans delivery
            "it": "it-IT-DiegoNeural",     # Expressive Italian male
            "ja": "ja-JP-KeitaNeural",     # Natural Japanese male
            "pt": "pt-BR-AntonioNeural",   # Friendly Brazilian Portuguese
            "ru": "ru-RU-DmitryNeural",    # Warm Russian baritone
            "sv": "sv-SE-MattiasNeural",   # Clear Swedish articulation
            "th": "th-TH-NiwatNeural",     # Natural Thai male
        }

        voice = voice_map.get(language)
        if not voice:
            raise ValueError(f"Unsupported language for Edge-TTS: {language}")

        tts = edge_tts.Communicate(text, voice)
        await tts.save(output_file)

        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            raise ValueError("Edge-TTS failed to generate a valid audio file.")

        print(f"Audio file saved as {output_file}")
    except Exception as e:
        print(f"Error in text_to_speech: {e}")
        raise

def text_to_speech(translated_text_path, language):
    try:
        # Read the translated text file
        with open(translated_text_path, 'r', encoding='utf-8') as file:
            text = file.read()
            print(f"Text for TTS: {text}")  # Debug: Check text for TTS

        output_file = os.path.join("static/outputs", "translated_audio.mp3")
        asyncio.run(generate_speech(text, language, output_file))

    except Exception as e:
        print(f"Error in text_to_speech: {e}")
        raise

def create_subtitled_video(video_path, srt_path, output_path):
    """
    Burn subtitles into the video using FFmpeg with improved sync settings
    """
    try:
        # Enhanced FFmpeg command with subtitle sync settings
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'subtitles={srt_path}:force_style=\'FontSize=24,Alignment=2,MarginV=25\'',
            '-c:a', 'copy',
            '-vsync', 'cfr',  # Constant frame rate for better sync
            '-max_muxing_queue_size', '1024',  # Prevent muxing errors
            output_path
        ]
        subprocess.run(command, check=True)
        print(f"Successfully created subtitled video at {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating subtitled video: {e}")

def integrate_audio_and_subtitles(speech_audio_path, video_path, srt_path, final_video_path):
    """
    Integrate translated audio and synchronized subtitles with the video
    """
    try:
        # Step 1: Get video duration
        probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        video_duration = float(subprocess.run(probe_cmd, capture_output=True, text=True).stdout.strip())

        # Step 2: Load and adjust audio duration
        audio = AudioSegment.from_file(speech_audio_path)
        audio_duration = len(audio) / 1000.0  # Convert to seconds

        # Step 3: Create temp audio with matched duration
        if audio_duration != video_duration:
            # Adjust audio speed to match video duration
            speed_factor = audio_duration / video_duration
            if speed_factor > 1:
                # Slow down audio
                adjusted_audio = audio._spawn(audio.raw_data, overrides={
                    "frame_rate": int(audio.frame_rate * speed_factor)
                })
            else:
                # Speed up audio
                adjusted_audio = audio._spawn(audio.raw_data, overrides={
                    "frame_rate": int(audio.frame_rate / speed_factor)
                })
            adjusted_audio = adjusted_audio.set_frame_rate(audio.frame_rate)
        else:
            adjusted_audio = audio

        # Save temporary adjusted audio
        temp_audio_path = "temp_adjusted_audio.wav"
        adjusted_audio.export(temp_audio_path, format="wav")

        # Step 4: Create subtitled video with timing adjustments
        temp_subtitled_video = "temp_subtitled_video.mp4"
        subtitle_command = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'subtitles={srt_path}:force_style=\'FontSize=24,Alignment=2,MarginV=25\'',
            '-c:a', 'copy',
            '-vsync', 'cfr',
            temp_subtitled_video
        ]
        subprocess.run(subtitle_command, check=True)

        # Step 5: Combine everything with precise timing
        final_command = [
            'ffmpeg',
            '-i', temp_subtitled_video,
            '-i', temp_audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-async', '1',  # Audio sync adjustment
            '-vsync', 'cfr',
            '-max_muxing_queue_size', '1024',
            final_video_path
        ]
        subprocess.run(final_command, check=True)

        # Cleanup
        if os.path.exists(temp_subtitled_video):
            os.remove(temp_subtitled_video)
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        print(f"Successfully created final video with synchronized audio and subtitles")
    except subprocess.CalledProcessError as e:
        print(f"Error in final video creation: {e}")
        # Cleanup on error
        if os.path.exists(temp_subtitled_video):
            os.remove(temp_subtitled_video)
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
    except Exception as e:
        print(f"Unexpected error: {e}")

def process_video_with_subtitles(video_path, target_language):
    """
    Main function to process video with synchronized subtitles and translated audio
    """
    # Extract audio
    
    audio_path = "static/outputs/extracted_audio.mp3"
    extract_audio(video_path, audio_path)
    
    # Transcribe with timestamps
    transcription_result = transcribe_audio_with_timestamps(audio_path)
    
    # Save original subtitles
    original_srt = generate_srt_from_segments(transcription_result['segments'])
    with open("static/outputs/original.srt", "w", encoding='utf-8') as f:
        f.write(original_srt)
    
    # Translate text while preserving segment structure
    translated_segments = translate_to_text(transcription_result, 
                                         "static/outputs/translated_text.txt",
                                         target_language)
    
    # Generate translated audio
    text_to_speech("static/outputs/translated_text.txt", target_language)
    
    # Create translated SRT file using original timing with translated text
    translated_srt_content = ""
    for i, (segment, translated_text) in enumerate(zip(transcription_result['segments'], translated_segments), 1):
        start = format_timestamp(segment['start'])
        end = format_timestamp(segment['end'])
        translated_srt_content += f"{i}\n{start} --> {end}\n{translated_text.strip()}\n\n"
    
    with open("static/outputs/translated.srt", "w", encoding='utf-8') as f:
        f.write(translated_srt_content)
    
    # Integrate everything together
    integrate_audio_and_subtitles(
        "static/outputs/translated_audio.mp3",
        video_path,
        "static/outputs/translated.srt",
        "static/outputs/final_video.mp4"
    )