import re
import spacy
from sklearn.metrics import f1_score

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# =============================
# 1. TEXT CLEANER
# =============================
FILLER_WORDS = [
    "um", "uh", "please", "sir", "actually",
    "hello", "hey", "kindly", "ok", "okay"
]

def clean_text(text):
    text = text.lower()
    for word in FILLER_WORDS:
        text = re.sub(r"\b" + word + r"\b", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =============================
# 2. EMERGENCY CLASSIFIER
# =============================
EMERGENCY_KEYWORDS = {
    "Medical": ["injured", "unconscious", "heart attack", "bleeding", "dying"],
    "Fire": ["fire", "smoke", "burning", "gas leak", "explosion"],
    "Accident": ["accident", "crash", "collision", "hit"],
    "Crime": ["attack", "robbery", "fight", "threat", "assault"]
}

def classify_emergency(text):
    for emergency_type, keywords in EMERGENCY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                return emergency_type

    # Future enhancement: BERT-based classification
    return "Unknown"




# =============================
# 3. URGENCY SCORING
# =============================
URGENCY_KEYWORDS = {
    "CRITICAL": ["dying", "unconscious", "not breathing"],
    "HIGH": ["serious", "badly", "emergency", "immediately"],
    "MEDIUM": ["help", "injured", "pain"],
    "LOW": ["minor", "small", "not serious"]
}

def detect_urgency(text):
    for level, keywords in URGENCY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                return level
    return "LOW"


# =============================
# 4. LOCATION EXTRACTION
# =============================
def extract_location(text):
    # Try spaCy NER
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC", "FAC"]:
            return ent.text

    # Fallback rule
    words = text.split()
    for i, word in enumerate(words):
        if word in ["in", "near", "at"] and i + 1 < len(words):
            return words[i + 1]

    return "Not Found"


# =============================
# MAIN (RAW SITUATIONS + F1)
# =============================
if __name__ == "__main__":

    # You can add ANY extra values, it will NOT crash
    
    raw_situations = [
    ("burning accident in chittoor please help immediately", "Fire"),
    ("a person is injured at Tirupathi", "Medical"),
    ("vehicle overturned near rayachoty", "Accident")
   
   
]



    y_true = []
    y_pred = []

    for item in raw_situations:
        raw_text = item[0]          # always first
        correct_label = item[1]     # always second

        cleaned_text = clean_text(raw_text)
        predicted_label = classify_emergency(cleaned_text)
        urgency = detect_urgency(cleaned_text)
        location = extract_location(cleaned_text)

        print("\nRaw Text       :", raw_text)
        print("Predicted Type :", predicted_label)
        print("Correct Type   :", correct_label)
        print("Urgency Level  :", urgency)
        print("Location       :", location)

        y_true.append(correct_label)
        y_pred.append(predicted_label)

    f1 = f1_score(y_true, y_pred, average="macro")
    print("\nFinal F1 Score :", round(f1 * 100, 2), "%")
