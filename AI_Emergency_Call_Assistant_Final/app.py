from flask import Flask, render_template, request
import os
from models.asr import speech_to_text
from models.emotion import detect_emotion
from models.ner import extract_entities

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

last_results = []   

severity_score = {
    "High": 3,
    "Medium": 2,
    "Low": 1
}

@app.route("/", methods=["GET", "POST"])
def dashboard():
    global last_results

    if request.method == "POST":
        audios = request.files.getlist("audio")

        results = []

        for audio in audios:
            if audio.filename == "":
                continue

            path = os.path.join(UPLOAD_FOLDER, audio.filename)
            audio.save(path)

            transcript = speech_to_text(path)
            emotion = detect_emotion(path)
            entities = extract_entities(transcript)

            results.append({
                "filename": audio.filename,
                "transcript": transcript,
                "emotion": emotion,
                "severity": entities.get("Severity", "Low"),
                "entities": entities
            })

       
        results.sort(
            key=lambda x: severity_score.get(x["severity"], 0),
            reverse=True
        )

        last_results = results

    return render_template("dashboard.html", results=last_results)

if __name__ == "__main__":
    app.run()
