from flask import Flask, render_template, request, url_for, send_from_directory
import os
from functions import process_video_with_subtitles
import glob

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/download_video/<filename>")
def download_video(filename):
    return send_from_directory(
        'static/outputs',  # Folder containing the final video
        filename,
        as_attachment=True  # Will prompt a download
    )

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
        
        # Clean up existing output files
        output_files = [
            "extracted_audio.mp3",
            "translated_audio.mp3",
            "final_video.mp4",
            "original.srt",
            "translated.srt"
        ]

        for file in output_files:
            file_path = os.path.join(OUTPUT_FOLDER, file)
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"{file} deleted.")
            except Exception as e:
                print(f"Error deleting {file}: {e}")

        # Remove dynamic files (optional)
        for file_pattern in ["*.mp3", "*.mp4", "*.srt"]:
            for file_path in glob.glob(os.path.join(OUTPUT_FOLDER, file_pattern)):
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        
        target_language = request.form.get("target_language")
        if not target_language:
            return "Target language not selected.", 400
        
        # Process video with synchronized subtitles
        process_video_with_subtitles(video_path, target_language)
        
        uploaded_video_url = url_for("static", filename=f"uploads/{video_file.filename}")
        processed_video_url = url_for("static", filename="outputs/final_video.mp4")
        
        transcription_path = os.path.join(OUTPUT_FOLDER, "translated_text.txt")

        # Read the transcription if the file exists
        if os.path.exists(transcription_path):
            with open(transcription_path, "r", encoding="utf-8") as file:
                transcription = file.read()

        return render_template(
            "index.html",
            uploaded_video_url=uploaded_video_url,
            video_url=processed_video_url,
            transcription=transcription,
        )
        
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)