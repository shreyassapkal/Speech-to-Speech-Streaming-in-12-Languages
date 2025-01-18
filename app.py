from flask import Flask, render_template, request, url_for
import os
from functions import (
    extract_audio,
    transcribe_audio,
    translate_to_text,
    text_to_speech,
    integrate_audio_to_video,
)

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html", uploaded_video_url=None, video_url=None)

@app.route("/process_video", methods=["POST"])
def process_video():
    try:
        if "video_file" not in request.files:
            return "No file uploaded.", 400

        video_file = request.files["video_file"]
        if video_file.filename == "":
            return "No selected file.", 400
        
        # Save uploaded video
        video_path = os.path.join("static/uploads", video_file.filename)
        video_file.save(video_path)
        
        if os.path.exists("static/outputs/extracted_audio.mp3"):
            os.remove("static/outputs/extracted_audio.mp3")
            print(f"{"extracted_audio.mp3"} deleted.")

        # Extract audio using the function from functions.py
        audio_path = os.path.join(OUTPUT_FOLDER, "extracted_audio.mp3")
        extract_audio(video_path, audio_path)  # Using extract_audio function

        # Extract audio, transcribe, translate, generate audio, and integrate audio to video
        audio_path = "static/outputs/extracted_audio.mp3"
        transcription_text = transcribe_audio(audio_path)
        transcription_file = "static/outputs/transcribed_text.txt"
        with open(transcription_file, "w", encoding='utf-8') as file:
            file.write(transcription_text)

        target_language = request.form.get("target_language")
        if not target_language:
            return "Target language not selected.", 400

        translated_file = "static/outputs/translated_text.txt"
        translate_to_text(transcription_file, translated_file, target_language)

        speech_audio_path = "static/outputs/translated_audio.mp3"
        text_to_speech(translated_file, target_language)

        if os.path.exists("static/outputs/final_video.mp4"):
            os.remove("static/outputs/final_video.mp4")
            print(f"{"final_video.mp4"} deleted.")

        final_video_path = "static/outputs/final_video.mp4"
        integrate_audio_to_video(speech_audio_path, video_path, final_video_path)

        uploaded_video_url = url_for("static", filename=f"uploads/{video_file.filename}")
        processed_video_url = url_for("static", filename="outputs/final_video.mp4")

        return render_template(
            "index.html",
            uploaded_video_url=uploaded_video_url,
            video_url=processed_video_url,
        )

    #    return render_template('index.html', video_url=url_for('static', filename='outputs/final_video.mp4'))
    
    except Exception as e:
        return f"Error: {str(e)}", 500


if __name__ == "__main__":
    app.run(debug=True)
