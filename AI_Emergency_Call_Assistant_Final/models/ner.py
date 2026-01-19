import re
import spacy
import speech_recognition as sr

# =============================
# LOAD NLP MODEL
# =============================
nlp = spacy.load("en_core_web_sm")

# =============================
# TEXT PREPROCESSING
# =============================
STOPWORDS = {
    "um", "uh", "please", "sir", "actually",
    "hello", "hey", "kindly", "ok", "okay"
}

def preprocess_text(text: str) -> str:
    text = text.lower()
    for w in STOPWORDS:
        text = re.sub(rf"\b{w}\b", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


# =============================
# EMERGENCY TYPE EXTRACTION
# =============================
def extract_emergency_type(text: str) -> str:
    text = text.lower()

    # 1️⃣ Fire (highest priority)
    fire_keywords = [
        "fire", "smoke", "burning", "gas leak", "explosion"
    ]
    if any(word in text for word in fire_keywords):
        return "Fire"

    # 2️⃣ Accident
    accident_keywords = [
        "accident", "crash", "collision", "vehicle", "road accident"
    ]
    if any(word in text for word in accident_keywords):
        return "Accident"

    # 3️⃣ Crime
    crime_keywords = [
        "attack", "robbery", "assault", "fight", "threat"
    ]
    if any(word in text for word in crime_keywords):
        return "Crime"

    # 4️⃣ Medical (lowest priority)
    medical_keywords = [
        "injured", "bleeding", "unconscious", "pain", "heart attack"
    ]
    if any(word in text for word in medical_keywords):
        return "Medical"

    return "Unknown"



# =============================
# URGENCY EXTRACTION
# =============================
def extract_urgency(text: str) -> str:
    text = text.lower()

    # 1️⃣ Explicit LOW urgency indicators (highest priority)
    low_indicators = [
        "minor",
        "not serious",
        "no one is injured",
        "no one injured",
        "no serious injury"
    ]

    for phrase in low_indicators:
        if phrase in text:
            return "LOW"

    # 2️⃣ CRITICAL indicators
    critical_words = [
        "dying",
        "unconscious",
        "not breathing",
        "severe bleeding"
    ]

    for word in critical_words:
        if word in text:
            return "CRITICAL"

    # 3️⃣ HIGH urgency indicators
    high_words = [
        "serious",
        "bleeding",
        "emergency",
        "immediately",
        "ambulance"
    ]

    for word in high_words:
        if word in text:
            return "HIGH"

    # 4️⃣ Default
    return "MEDIUM"



# =============================
# LOCATION EXTRACTION (NER)
# =============================
def extract_location(text: str) -> str:
    doc = nlp(text)

    # 1️⃣ Named Entity Recognition (preferred)
    for ent in doc.ents:
        if ent.label_ in {"GPE", "LOC", "FAC"}:
            return ent.text

    # 2️⃣ Rule-based fallback for common places
    fallback_places = [
        "bus stand", "bus stop", "railway station",
        "hospital", "college", "school", "market",
        "road", "highway"
    ]

    text_lower = text.lower()
    for place in fallback_places:
        if place in text_lower:
            return place

    return "Not Found"


# =============================
# CORE FACT EXTRACTION (YOUR ROLE)
# =============================
def extract_emergency_facts(transcribed_text: str) -> dict:
    clean = preprocess_text(transcribed_text)

    return {
        "emergency_type": extract_emergency_type(clean),
        "urgency_level": extract_urgency(clean),
        "location": extract_location(transcribed_text)
    }


# =============================
# AUDIO → TEXT (NO WHISPER)
# =============================
def transcribe_audio(audio_path: str) -> str:
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio)


# =============================
# MAIN (REVIEW DEMO)
# =============================
# =============================
# MAIN (REVIEW DEMO)
# =============================
import os

if __name__ == "__main__":

    base_dir = os.path.dirname(__file__)
    audio_dir = os.path.join(base_dir, "demo_audios")

    if not os.path.exists(audio_dir):
        print("Audio folder not found!")
        exit()

    audio_files = [
        f for f in os.listdir(audio_dir)
        if f.lower().endswith(".wav")
    ]

    if not audio_files:
        print("No audio files found in demo_audios folder!")
        exit()

    print(f"Found {len(audio_files)} audio files\n")

    for audio_file in audio_files:
        audio_path = os.path.join(audio_dir, audio_file)

        print("=" * 50)
        print("Processing:", audio_file)

        try:
            # Step 1: Audio → Text
            text = transcribe_audio(audio_path)
            print("\nTranscribed Text:")
            print(text)

            # Step 2: Text → Emergency Facts
            facts = extract_emergency_facts(text)
            print("\nExtracted Emergency Facts:")
            print(facts)

        except Exception as e:
            print("Error processing file:", audio_file)
            print("Reason:", e)

