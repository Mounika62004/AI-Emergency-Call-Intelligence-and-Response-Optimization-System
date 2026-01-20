from flask import Flask, request, jsonify, render_template
import os
import librosa
import torch

from speechbrain.inference.classifiers import EncoderClassifier


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("🔄 Loading emotion model (one-time, may take a minute first time)...")

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir=os.path.join(BASE_DIR, "pretrained_emotion_model")
)

print("✅ Emotion model loaded successfully.")

EMOTION_MAP = {
    "anger": "PANIC",
    "fear": "PANIC",
    "sadness": "DISTRESS",
    "disgust": "DISTRESS",
    "neutral": "CALM",
    "happiness": "CALM",
    "surprise": "PANIC"
}

def detect_emotion(audio_path: str) -> str:
    try:
        if not os.path.exists(audio_path):
            return "CALM"

        signal, sr = librosa.load(audio_path, sr=16000)

        if signal is None or len(signal) < 1000:
            return "CALM"

        signal = torch.tensor(signal).unsqueeze(0)

        prediction = classifier.classify_batch(signal)

        predicted_label = prediction[3][0].lower()
        print("🎧 Raw emotion:", predicted_label)

        final_emotion = EMOTION_MAP.get(predicted_label, "CALM")
        print("🚨 Mapped emotion:", final_emotion)

        return final_emotion

    except Exception as e:
        print("❌ Emotion error:", e)
        return "CALM"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]

    if audio_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(file_path)

    emotion = detect_emotion(file_path)

    return jsonify({"emotion": emotion}

if __name__ == "__main__":
    print("🚀 Starting server...")
    app.run(host="0.0.0.0", port=5000, debug=True)

