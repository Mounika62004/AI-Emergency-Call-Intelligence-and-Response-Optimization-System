import re
import spacy
from sklearn.metrics import f1_score


# =============================
# LOAD spaCy MODEL
# =============================
nlp = spacy.load("en_core_web_sm")

# =============================
# TEXT CLEANING
# =============================
FILLER_WORDS = [
    "um", "uh", "please", "sir", "actually",
    "hello", "hey", "kindly", "ok", "okay"
]

def clean_text(text):
    text = text.lower()
    for word in FILLER_WORDS:
     text = re.sub(rf"\b{word}\b", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =============================
# EMERGENCY CLASSIFICATION
# =============================
EMERGENCY_KEYWORDS = {
    "Fire": [
        "fire", "smoke", "burning", "flames",
        "gas leak", "explosion", "blast",
        "short circuit", "caught fire"
    ],
    "Medical": [
        "injured", "unconscious", "bleeding",
        "heart attack", "not breathing",
        "collapsed", "ambulance"
    ],
    "Accident": [
        "accident", "crash", "collision",
        "hit", "overturned", "vehicle"
    ],
    "Crime": [
        "robbery", "attack", "assault",
        "fight", "threat"
    ]
}

def classify_emergency(text):
    for emergency, keywords in EMERGENCY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                return emergency
    return "Unknown"


# =============================
# SEVERITY DETECTION
# =============================
SEVERITY_KEYWORDS = {
    "High": [
        "unconscious", "not breathing",
        "dying", "serious", "emergency",
        "panic", "immediately"
    ],
    "Medium": [
        "injured", "help", "pain",
        "fear", "gas leak"
    ],
    "Low": [
        "minor", "small"
    ]
}

def detect_severity(text):
    for level, keywords in SEVERITY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                return level
    return "Low"


# =============================
# LOCATION EXTRACTION
# =============================
def extract_location(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC", "FAC"]:
            return ent.text.title()

    words = text.split()
    for i, w in enumerate(words):
        if w in ["in", "near", "at"] and i + 1 < len(words):
            return words[i + 1].title()

    return "Not Found"


# =============================
# MAIN FUNCTION
# =============================
def extract_entities(transcript):
    cleaned = clean_text(transcript)
    return {
        "EmergencyType": classify_emergency(cleaned),
        "Location": extract_location(cleaned),
        "Severity": detect_severity(cleaned)
    }


# =============================
#  DEMO OUTPUT 
# =============================
# =============================
# DEMO + F1 SCORE EVALUATION
# =============================
if __name__ == "__main__":

    # Test cases: (text, correct_emergency_type)
    test_cases = [
        ("Smoke is coming from a factory near Rayachoty people are panicking", "Fire"),
        ("Fire broke out in a building near Chittoor please help immediately", "Fire"),
        ("There was a road accident and a vehicle overturned near Kadapa", "Accident"),
        ("A man is unconscious and not breathing please send ambulance", "Medical"),
        ("Minor injury reported near a shop", "Medical")
    ]

    y_true = []
    y_pred = []

    print("\n===== NLP OUTPUT =====")

    for text, actual in test_cases:
        result = extract_entities(text)

        print("\nInput Text     :", text)
        print("Emergency Type :", result["EmergencyType"])
        print("Location       :", result["Location"])
        print("Severity       :", result["Severity"])

        y_true.append(actual)
        y_pred.append(result["EmergencyType"])

    # Calculate F1-score
    f1 = f1_score(y_true, y_pred, average="macro")

    print("\n===== MODEL EVALUATION =====")
    print("F1 Score :", round(f1 * 100, 2), "%")

