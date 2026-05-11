import pandas as pd
import joblib
import re
import os
import difflib
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util

load_dotenv()

from utils.data_loader import get_disease_info
from utils.translator import translate_to_user_lang, to_english
from utils.nlp_extractor import extract_symptoms_nlp

# ----------------------------
# Load model + dataset
# ----------------------------
model = joblib.load("model/disease_model.pkl")

train_df = pd.read_csv("data/training_improved.csv")
train_df.columns = train_df.columns.str.strip().str.replace(" ", "_")

# ----------------------------
# Load symptom severity
# ----------------------------
severity_df = pd.read_csv("data/symptom-severity.csv")
severity_df["Symptom"] = severity_df["Symptom"].str.strip()

severity_map = dict(zip(severity_df["Symptom"], severity_df["weight"]))

symptom_columns = train_df.columns[:-1].tolist()
ALL_DISEASES = set(model.classes_)
ALL_DISEASES_LOWER = {d.lower() for d in ALL_DISEASES}

# Load SentenceTransformer for English symptom matching
try:
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    symptom_phrases = [col.replace("_", " ") for col in symptom_columns]
    symptom_embeddings = st_model.encode(symptom_phrases, convert_to_tensor=True)
    print("✅ SentenceTransformer loaded for English symptom matching")
except Exception as e:
    st_model = None
    symptom_embeddings = None
    print(f"⚠️ SentenceTransformer failed: {e}")

# ----------------------------
# Configure Gemini API
# ----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

model_gemini = None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        model_gemini = genai.GenerativeModel("gemini-2.5-flash")
        print("✅ Gemini API is working")
    except Exception as e:
        print(f"⚠️ Gemini init error: {e}")
        model_gemini = None


# ----------------------------
# Disease Control
# ----------------------------
COMMON_DISEASES = {
    "Common Cold", "Viral Fever", "Flu", "Allergy", "Migraine", "Gastroenteritis"
}

SERIOUS_DISEASES = {
    "AIDS", "Paralysis (brain hemorrhage)", "Tuberculosis", "Cancer", "Heart attack"
}

# ----------------------------
# SYMPTOM MAP (EXPANDED)
# ----------------------------
RAW_SYMPTOM_MAP = {

    # ===================== FEVER =====================
    # English
    "fever": "high_fever",
    "high fever": "high_fever",
    "mild fever": "mild_fever",
    "low fever": "mild_fever",

    # Hindi
    "bukhar": "high_fever",
    "jwaram": "high_fever",
    "jwar": "high_fever",
    "tez bukhar": "high_fever",
    "kam bukhar": "mild_fever",

    # Telugu
    "జ్వరం": "high_fever",
    "jwaram": "high_fever",
    "light fever": "mild_fever",

    # ===================== COUGH =====================
    # English
    "cough": "cough",
    "coughing": "cough",

    # Hindi
    "khansi": "cough",
    "khasi": "cough",

    # Telugu
    "దగ్గు": "cough",
    "daggu": "cough",

    # ===================== COLD / SNEEZING =====================
    # English
    "cold": "continuous_sneezing",
    "sneezing": "continuous_sneezing",
    "continuous sneezing": "continuous_sneezing",

    # Hindi
    "sardi": "continuous_sneezing",
    "jalubu": "continuous_sneezing",
    "thand lagna": "chills",

    # Telugu
    "జలుబు": "continuous_sneezing",
    "cheemidi": "runny_nose",
    "ముక్కు కారుతుంది": "runny_nose",

    # ===================== RUNNY NOSE =====================
    "runny nose": "runny_nose",
    "mukku karutundi": "runny_nose",
    "mukku nundi neeru vastundi": "runny_nose",
    "nasal discharge": "runny_nose",

    # ===================== HEADACHE =====================
    # English
    "headache": "headache",
    "head ache": "headache",
    "pain in head": "headache",

    # Hindi
    "sar dard": "headache",
    "sir dard": "headache",

    # Telugu
    "తలనొప్పి": "headache",
    "tala noppi": "headache",
    "talanoppi": "headache",
    "talanoppi ga undi": "headache",

    # ===================== STOMACH PAIN =====================
    # English
    "stomach pain": "stomach_pain",
    "stomach ache": "stomach_pain",
    "stomach hurts": "stomach_pain",
    "abdominal pain": "abdominal_pain",
    "belly pain": "belly_pain",
    "pain in stomach": "stomach_pain",

    # Hindi
    "pet dard": "stomach_pain",
    "pet mein dard": "stomach_pain",
    "pet me dard":"stomach_pain",
    "pait mein dard": "stomach_pain",
    "pet dard hai": "stomach_pain",
    "mujhe pet dard hai": "stomach_pain",
    "mere pet me dard hai": "stomach_pain",

    # Telugu
    "కడుపు నొప్పి": "stomach_pain",
    "kadupu noppi": "stomach_pain",
    "naaku pet dard undi": "stomach_pain",

    # ===================== VOMITING =====================
    # English
    "vomiting": "vomiting",
    "vomit": "vomiting",
    "throwing up": "vomiting",

    # Hindi
    "ulti": "vomiting",
    "ulti ho rahi": "vomiting",
    "vomiting ho raha": "vomiting",

    # Telugu
    "వాంతులు": "vomiting",
    "vanti": "vomiting",

    # ===================== DIARRHOEA =====================
    # English
    "diarrhea": "diarrhoea",
    "loose motions": "diarrhoea",
    "loose stools": "diarrhoea",

    # Hindi
    "dast": "diarrhoea",
    "loose motion": "diarrhoea",

    # Telugu
    "డయేరియా": "diarrhoea",
    "dayeriya": "diarrhoea",

    # ===================== WEAKNESS =====================
    # English
    "fatigue": "fatigue",
    "weakness": "fatigue",
    "tiredness": "fatigue",

    # Hindi
    "kamzori": "fatigue",
    "thakan": "fatigue",
    "bahut kamzori": "fatigue",

    # Telugu
    "బలహీనత": "fatigue",
    "weakness": "fatigue",

    # ===================== DIZZINESS =====================
    # English
    "dizziness": "dizziness",
    "vertigo": "dizziness",

    # Hindi
    "chakkar": "dizziness",
    "chakkar aana": "dizziness",

    # Telugu
    "తల తిరగడం": "dizziness",
    "chakkar": "dizziness",

    # ===================== CHEST PAIN =====================
    "chest pain": "chest_pain",
    "pain in chest": "chest_pain",

    # ===================== NAUSEA =====================
    "nausea": "nausea",
    "feeling sick": "nausea",

    # ===================== CHILLS =====================
    "chills": "chills",
    "shivering": "shivering",
    "shiverng": "shivering",

    # ===================== JOINT PAIN =====================
    "joint pain": "joint_pain",
    "pain in joints": "joint_pain",

    # ===================== SKIN RASH =====================
    "skin rash": "skin_rash",
    "rash": "skin_rash",

    # ===================== ITCHING =====================
    "itching": "itching",
    "itch": "itching",

    # ===================== CONSTIPATION =====================
    "constipation": "constipation",

    # ===================== BACK PAIN =====================
    "back pain": "back_pain",

    # ===================== BREATHLESSNESS =====================
    "breathlessness": "breathlessness",
    "shortness of breath": "breathlessness",
    "difficulty breathing": "breathlessness",

    # ===================== SWEATING =====================
    "sweating": "sweating",
    "excessive sweating": "sweating",

    # ===================== DEHYDRATION =====================
    "dehydration": "dehydration",

    # ===================== INDIGESTION =====================
    "indigestion": "indigestion",

    # ===================== YELLOWISH SKIN =====================
    "yellowish skin": "yellowish_skin",
    "jaundice": "yellowish_skin",

    # ===================== DARK URINE =====================
    "dark urine": "dark_urine",

    # ===================== LOSS OF APPETITE =====================
    "loss of appetite": "loss_of_appetite",

    # ===================== PAIN BEHIND EYES =====================
    "pain behind eyes": "pain_behind_the_eyes",

    # ===================== SUNKEN EYES =====================
    "sunken eyes": "sunken_eyes",

    # ===================== WEAKNESS IN LIMBS =====================
    "weakness in limbs": "weakness_in_limbs",

    # ===================== FAST HEART RATE =====================
    "fast heart rate": "fast_heart_rate",
    "palpitations": "palpitations",

    # ===================== NECK PAIN =====================
    "neck pain": "neck_pain",

    # ===================== CRAMPS =====================
    "cramps": "cramps",

    # ===================== BRUISING =====================
    "bruising": "bruising",

    # ===================== OBESITY =====================
    "obesity": "obesity",

    # ===================== SWOLLEN LEGS =====================
    "swollen legs": "swollen_legs",

    # ===================== PUFFY FACE =====================
    "puffy face": "puffy_face_and_eyes",

    # ===================== ENLARGED THYROID =====================
    "enlarged thyroid": "enlarged_thyroid",

    # ===================== BRITTLE NAILS =====================
    "brittle nails": "brittle_nails",

    # ===================== EXCESSIVE HUNGER =====================
    "excessive hunger": "excessive_hunger",

    # ===================== DRYING LIPS =====================
    "drying lips": "drying_and_tingling_lips",

    # ===================== SLURRED SPEECH =====================
    "slurred speech": "slurred_speech",

    # ===================== KNEE PAIN =====================
    "knee pain": "knee_pain",

    # ===================== HIP JOINT PAIN =====================
    "hip joint pain": "hip_joint_pain",

    # ===================== MUSCLE WEAKNESS =====================
    "muscle weakness": "muscle_weakness",

    # ===================== STIFF NECK =====================
    "stiff neck": "stiff_neck",

    # ===================== SWELLING JOINTS =====================
    "swelling joints": "swelling_joints",

    # ===================== MOVEMENT STIFFNESS =====================
    "movement stiffness": "movement_stiffness",

    # ===================== SPINNING MOVEMENTS =====================
    "spinning movements": "spinning_movements",

    # ===================== LOSS OF BALANCE =====================
    "loss of balance": "loss_of_balance",

    # ===================== UNSTEADINESS =====================
    "unsteadiness": "unsteadiness",

    # ===================== WEAKNESS OF ONE BODY SIDE =====================
    "weakness of one body side": "weakness_of_one_body_side",

    # ===================== LOSS OF SMELL =====================
    "loss of smell": "loss_of_smell",

    # ===================== BLADDER DISCOMFORT =====================
    "bladder discomfort": "bladder_discomfort",

    # ===================== FOUL SMELL OF URINE =====================
    "foul smell of urine": "foul_smell_of_urine",

    # ===================== CONTINUOUS FEEL OF URINE =====================
    "continuous feel of urine": "continuous_feel_of_urine",

    # ===================== PASSAGE OF GASES =====================
    "passage of gases": "passage_of_gases",

    # ===================== INTERNAL ITCHING =====================
    "internal itching": "internal_itching",

    # ===================== TOXIC LOOK =====================
    "toxic look": "toxic_look_(typhos)",

    # ===================== DEPRESSION =====================
    "depression": "depression",

    # ===================== IRRITABILITY =====================
    "irritability": "irritability",

    # ===================== MUSCLE PAIN =====================
    "muscle pain": "muscle_pain",

    # ===================== ALTERED SENSORIUM =====================
    "altered sensorium": "altered_sensorium",

    # ===================== RED SPOTS OVER BODY =====================
    "red spots over body": "red_spots_over_body",

    # ===================== ABNORMAL MENSTRUATION =====================
    "abnormal menstruation": "abnormal_menstruation",

    # ===================== DISCHROMIC PATCHES =====================
    "dischromic patches": "dischromic__patches",

    # ===================== WATERING FROM EYES =====================
    "watering from eyes": "watering_from_eyes",

    # ===================== INCREASED APPETITE =====================
    "increased appetite": "increased_appetite",

    # ===================== POLYURIA =====================
    "polyuria": "polyuria",

    # ===================== FAMILY HISTORY =====================
    "family history": "family_history",

    # ===================== MUCOID SPUTUM =====================
    "mucoid sputum": "mucoid_sputum",

    # ===================== RUSTY SPUTUM =====================
    "rusty sputum": "rusty_sputum",

    # ===================== LACK OF CONCENTRATION =====================
    "lack of concentration": "lack_of_concentration",

    # ===================== VISUAL DISTURBANCES =====================
    "visual disturbances": "visual_disturbances",

    # ===================== RECEIVING BLOOD TRANSFUSION =====================
    "receiving blood transfusion": "receiving_blood_transfusion",

    # ===================== RECEIVING UNSTERILE INJECTIONS =====================
    "receiving unsterile injections": "receiving_unsterile_injections",

    # ===================== COMA =====================
    "coma": "coma",

    # ===================== STOMACH BLEEDING =====================
    "stomach bleeding": "stomach_bleeding",

    # ===================== DISTENTION OF ABDOMEN =====================
    "distention of abdomen": "distention_of_abdomen",

    # ===================== HISTORY OF ALCOHOL CONSUMPTION =====================
    "history of alcohol consumption": "history_of_alcohol_consumption",

    # ===================== BLOOD IN SPUTUM =====================
    "blood in sputum": "blood_in_sputum",

    # ===================== PROMINENT VEINS ON CALF =====================
    "prominent veins on calf": "prominent_veins_on_calf",

    # ===================== PAINFUL WALKING =====================
    "painful walking": "painful_walking",

    # ===================== PUS FILLED PIMPLES =====================
    "pus filled pimples": "pus_filled_pimples",

    # ===================== BLACKHEADS =====================
    "blackheads": "blackheads",

    # ===================== SCURRING =====================
    "scurring": "scurring",

    # ===================== SKIN PEELING =====================
    "skin peeling": "skin_peeling",

    # ===================== SILVER LIKE DUSTING =====================
    "silver like dusting": "silver_like_dusting",

    # ===================== SMALL DENTS IN NAILS =====================
    "small dents in nails": "small_dents_in_nails",

    # ===================== INFLAMMATORY NAILS =====================
    "inflammatory nails": "inflammatory_nails",

    # ===================== BLISTER =====================
    "blister": "blister",

    # ===================== RED SORE AROUND NOSE =====================
    "red sore around nose": "red_sore_around_nose",

    # ===================== YELLOW CRUST OOZE =====================
    "yellow crust ooze": "yellow_crust_ooze",

    # ===================== WEIGHT GAIN =====================
    "weight gain": "weight_gain",

    # ===================== ANXIETY =====================
    "anxiety": "anxiety",

    # ===================== COLD HANDS AND FEETS =====================
    "cold hands and feets": "cold_hands_and_feets",

    # ===================== MOOD SWINGS =====================
    "mood swings": "mood_swings",

    # ===================== WEIGHT LOSS =====================
    "weight loss": "weight_loss",

    # ===================== RESTLESSNESS =====================
    "restlessness": "restlessness",

    # ===================== LETHARGY =====================
    "lethargy": "lethargy",

    # ===================== PATCHES IN THROAT =====================
    "patches in throat": "patches_in_throat",

    # ===================== IRREGULAR SUGAR LEVEL =====================
    "irregular sugar level": "irregular_sugar_level",
    "acidity": "acidity",
    "ulcers on tongue": "ulcers_on_tongue",
    "muscle wasting": "muscle_wasting",
    "burning micturition": "burning_micturition",
    "spotting urination": "spotting__urination",
    "nodal skin eruptions": "nodal_skin_eruptions",
    "pain during bowel movements": "pain_during_bowel_movements",
    "pain in anal region": "pain_in_anal_region",
    "bloody stool": "bloody_stool",
    "irritation in anus": "irritation_in_anus",
    "swollen blood vessels": "swollen_blood_vessels",
    "swollen extremities": "swollen_extremeties",
    "extra marital contacts": "extra_marital_contacts",
    "phlegm": "phlegm",
    "throat irritation": "throat_irritation",
    "sore throat": "throat_irritation",
    "sorethroat": "throat_irritation",
    "redness of eyes": "redness_of_eyes",
    "sinus pressure": "sinus_pressure",
    "congestion": "congestion",
    "malaise": "malaise",
    "blurred vision": "blurred_and_distorted_vision",
    "swelled lymph nodes": "swelled_lymph_nodes",
    "swelling of stomach": "swelling_of_stomach",
    "fluid overload": "fluid_overload",
    "acute liver failure": "acute_liver_failure",
    "yellowing of eyes": "yellowing_of_eyes",
    "yellow urine": "yellow_urine",
    "urinary tract infection": "urinary_tract_infection",

    # Additional common phrases
    "pet me dard hai": "stomach_pain",
    "pet mein dard hai": "stomach_pain",
    "pet dard hai": "stomach_pain",
    "aur": None,
    "ulti ho rahi hai": "vomiting",
    "ulti aa rahi hai": "vomiting",
    "pet me dard ho raha hai": "stomach_pain",
    "pet dard ho raha hai": "stomach_pain",
    "mere pet me dard hai": "stomach_pain",
    "mujhe ulti ho rahi hai": "vomiting",
}

def build_symptom_aliases(columns):
    aliases = {}

    for col in columns:
        phrase = col.replace("_", " ")
        if phrase not in aliases:
            aliases[phrase] = col
        aliases[col] = col

    return aliases


SYMPTOM_MAP = {k: v for k, v in RAW_SYMPTOM_MAP.items() if v in symptom_columns}
SYMPTOM_MAP.update(build_symptom_aliases(symptom_columns))
SYMPTOM_ALIAS_KEYS = list(SYMPTOM_MAP.keys())

SYMPTOM_WORDS = set()
for phrase in SYMPTOM_ALIAS_KEYS + [col.replace("_", " ") for col in symptom_columns]:
    SYMPTOM_WORDS.update(phrase.split())


COMMON_MISSPELLINGS = {
    "feeever": "fever",
    "coold": "cold",
    "coough": "cough",
    "vommiting": "vomiting",
    "vommit": "vomiting",
    "weaknes": "weakness",
    "stmach": "stomach",
    "cama": "coma",
    "talanoppi": "talanoppi",
    "shiverng": "shivering",
    "sorethroat": "sore throat",
    "itchingg": "itching",
    "rashh": "rash",
}

def get_best_symptom_match(text, cutoff=0.75):
    if not text:
        return None

    normalized = text.lower().strip()
    if normalized in SYMPTOM_MAP:
        return SYMPTOM_MAP[normalized]

    matches = difflib.get_close_matches(normalized, SYMPTOM_ALIAS_KEYS, n=1, cutoff=cutoff)
    if matches:
        match = matches[0]
        return SYMPTOM_MAP.get(match)

    return None


def correct_symptom_spelling(text, cutoff=0.8):
    if not text:
        return text

    corrected_tokens = []
    for token in text.split():
        if token in SYMPTOM_WORDS or len(token) <= 2:
            corrected_tokens.append(token)
            continue

        if token in COMMON_MISSPELLINGS:
            corrected_tokens.append(COMMON_MISSPELLINGS[token])
            continue

        matches = difflib.get_close_matches(token, SYMPTOM_WORDS, n=1, cutoff=cutoff)
        corrected_tokens.append(matches[0] if matches else token)

    return " ".join(corrected_tokens)


def split_concatenated_symptoms(text):
    """Insert spaces between concatenated symptom words."""
    if not text:
        return text
    
    # Sort by length descending to match longer phrases first
    for symptom_word in sorted(SYMPTOM_WORDS, key=len, reverse=True):
        if len(symptom_word) <= 2:
            continue
        
        # Pattern: insert space before symptom if preceded by alphanumeric
        text = re.sub(
            r'([a-z0-9])' + re.escape(symptom_word) + r'(?=[a-z])',
            r'\1 ' + symptom_word + ' ',
            text,
            flags=re.IGNORECASE
        )
        
        # Pattern: insert space after symptom if followed by alphanumeric
        text = re.sub(
            r'(?<=[a-z])' + re.escape(symptom_word) + r'([a-z0-9])',
            symptom_word + r' \1',
            text,
            flags=re.IGNORECASE
        )
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ----------------------------
# SEMANTIC SYMPTOM MATCHING (FOR ENGLISH)
# ----------------------------
def match_symptoms_semantic(text, threshold=0.7):
    if st_model is None or symptom_embeddings is None:
        return []

    # Split text into potential symptom phrases
    sentences = re.split(r'[,.!?;]', text)
    matched_symptoms = set()

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Encode the sentence
        sentence_embedding = st_model.encode(sentence, convert_to_tensor=True)

        # Compute cosine similarities
        similarities = util.cos_sim(sentence_embedding, symptom_embeddings)[0]

        # Find matches above threshold
        for i, sim in enumerate(similarities):
            if sim > threshold:
                matched_symptoms.add(symptom_columns[i])

    return list(matched_symptoms)


# ----------------------------
# NORMALIZATION
# ----------------------------
def normalize_input(text, lang="en"):
    text = text.lower()

    # Remove common prefixes
    prefixes = ["i have", "i am having", "i feel", "i am feeling", "my", "mere", "mujhe", "nenu", "naaku"]
    for prefix in prefixes:
        if text.startswith(prefix + " "):
            text = text[len(prefix) + 1:]

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = correct_symptom_spelling(text)
    text = split_concatenated_symptoms(text)

    symptoms = set()

    # For English, use semantic matching first
    if lang == "en" and st_model:
        symptoms.update(match_symptoms_semantic(text))

    # split on common separators and conjunctions
    parts = re.split(r"[\,\;\+]|\b(?:aur|and|or|with|also|along with|plus|besides)\b", text)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # direct alias or phrase matches
        for key, value in SYMPTOM_MAP.items():
            if value and key in part:
                symptoms.add(value)

        # match column-like phrases by word set to handle varied prompt order
        part_words = set(part.split())
        for col in symptom_columns:
            col_words = set(col.replace("_", " ").split())
            if col_words and col_words.issubset(part_words):
                symptoms.add(col)

        if "pet" in part and "dard" in part:
            symptoms.add("stomach_pain")

    # catch symptoms anywhere in full text using actual dataset column phrases
    for col in symptom_columns:
        if col.replace("_", " ") in text:
            symptoms.add(col)

    # NLP fallback using the same symptom columns
    try:
        symptoms.update(extract_symptoms_nlp(text, symptom_columns))
    except:
        pass

    return list(symptoms)


# ----------------------------
# RULE ENGINE
# ----------------------------
def apply_medical_rules(symptoms):
    s = set(symptoms)

    if "runny_nose" in s or "continuous_sneezing" in s:
        return ["Common Cold", "Allergy"]

    if "high_fever" in s and "cough" in s:
        return ["Viral Fever", "Flu"]

    if "headache" in s and "vomiting" in s:
        return ["Migraine"]

    # 🔥 NEW CRITICAL RULE
    if "stomach_pain" in s and "vomiting" in s:
        return ["Gastroenteritis", "Food Poisoning"]

    if ("stomach_pain" in s or "abdominal_pain" in s) and "diarrhoea" in s:
        return ["Gastroenteritis"]

    if "chest_pain" in s and "breathlessness" in s:
        return ["Heart attack"]

    return []

# ----------------------------
# ML PREDICTION
# ----------------------------
def predict(symptoms_list):
    input_data = pd.DataFrame([[0] * len(symptom_columns)], columns=symptom_columns)

    for s in symptoms_list:
        if s in symptom_columns:
            input_data.loc[0, s] = severity_map.get(s, 1)

    probs = model.predict_proba(input_data)[0]
    classes = model.classes_

    results = []

    for i, disease in enumerate(classes):
        confidence = probs[i] * 100

        if disease == "Heart attack":
            if not any(x in symptoms_list for x in ["chest_pain", "breathlessness"]):
                continue

        if disease in SERIOUS_DISEASES and confidence < 65:
            continue

        if disease in COMMON_DISEASES:
            confidence += 10

        if confidence >= 20:
            results.append({
                "disease": disease,
                "confidence": round(min(confidence, 100), 2)
            })

    return sorted(results, key=lambda x: x["confidence"], reverse=True)[:3]


# ----------------------------
# DOCTOR ADVICE
# ----------------------------
def get_doctor_advice(symptoms, predictions):
    s = set(symptoms)

        # ----------------------------
    # ✅ NEW: Severity-based check
    # ----------------------------
    severity_score = sum(severity_map.get(sym, 1) for sym in symptoms)

    if severity_score > 20:
        return "🚨 Symptoms look serious. Consult doctor immediately."
    
    if "chest_pain" in s or "breathlessness" in s:
        return "🚨 Seek immediate medical attention."

    if "high_fever" in s:
        return "⚠️ If fever lasts more than 2-3 days, consult a doctor."

    if "vomiting" in s and "diarrhoea" in s:
        return "⚠️ Risk of dehydration."

    if predictions and predictions[0]["confidence"] < 60:
        return "⚠️ Diagnosis uncertain."

    return "✅ Monitor symptoms."


# ----------------------------
# FINAL RESPONSE BUILDER
# ----------------------------
def build_original_response(predictions, doctor_advice, lang, gemini_text=None):
    response = ""

    if gemini_text:
        response += f"""💬 AI Response:
{gemini_text}

----------------------------------------

"""

    for p in predictions:
        info = get_disease_info(p["disease"])

        response += f"""🦠 Disease: {p['disease']}
📊 Confidence: {p['confidence']}%

🧾 Description:
{info['description']}

💊 Medication:
{info['medication']}

🥗 Diet:
{info['diet']}

🛡️ Precautions:
{', '.join(info['precautions'])}

🏃 Workout:
{', '.join(info['workout'])}

----------------------------------------

"""

    response += f"""
🩺 Doctor Recommendation:
{doctor_advice}
"""

    return translate_to_user_lang(response, lang)
# ----------------------------
# SYMPTOM EXTRACTOR (FOR TELEGRAM + DB)
# ----------------------------
def extract_all_symptoms(user_input):
    # Step 1: original text
    translated_text, lang = to_english(user_input)
    symptoms = normalize_input(translated_text if translated_text else user_input.lower(), lang)

    return symptoms

# ----------------------------
# MAIN CHATBOT
# ----------------------------
def run_chatbot(user_input):

    # =========================
    # STEP 1: Detect + Translate
    # =========================
    translated_text, lang = to_english(user_input)

    if not translated_text:
        translated_text = user_input.lower()

    print("🌐 Language:", lang)
    print("📝 English Text:", translated_text)

    # =========================
    # STEP 2: Extract Symptoms (ONLY ENGLISH)
    # =========================
    symptoms = normalize_input(translated_text, lang)

    print("🧾 Symptoms:", symptoms)

    if not symptoms:
        return translate_to_user_lang("Could not understand symptoms.", lang)

    if len(symptoms) < 2:
        return translate_to_user_lang("Please provide more symptoms.", lang)

    # =========================
    # STEP 3: Rule Engine
    # =========================
    rule = apply_medical_rules(symptoms)

    if rule:
        predictions = [{"disease": d, "confidence": 70 - i * 5} for i, d in enumerate(rule)]
    else:
        predictions = predict(symptoms)

    if not predictions:
        predictions = [{"disease": "General Viral Infection", "confidence": 40}]

    # =========================
    # STEP 4: Doctor Advice
    # =========================
    doctor_advice = get_doctor_advice(symptoms, predictions)

    # =========================
    # STEP 5: Gemini Response
    # =========================
    gemini_text = None

    if model_gemini:
        prompt = f"""
You are a medical assistant.

Explain the condition in simple 3-4 lines.

User: {translated_text}
Symptoms: {', '.join(symptoms)}
Diseases: {', '.join([p['disease'] for p in predictions])}
"""

        try:
            response = model_gemini.generate_content(prompt)
            gemini_text = response.text if hasattr(response, "text") else None
        except Exception as e:
            print("Gemini error:", e)

    # =========================
    # STEP 6: Build Structured Response
    # =========================
    final_response = build_original_response(
        predictions,
        doctor_advice,
        lang,
        gemini_text
    )

    # =========================
    # STEP 7: Translate BACK
    # =========================
    return translate_to_user_lang(final_response, lang)



# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    print("💬 AI Health Assistant Ready")

    while True:
        user_input = input("Enter symptoms: ")

        if user_input.lower() == "exit":
            break

        print(run_chatbot(user_input))