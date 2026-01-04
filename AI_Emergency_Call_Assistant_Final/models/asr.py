# Automatic Speech Recognition (ASR) using Whisper model
# Converts emergency call audio into readable text
# Already implemented
import whisper
model = whisper.load_model("base")

def speech_to_text(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]
