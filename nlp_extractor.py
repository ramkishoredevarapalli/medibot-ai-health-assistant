import spacy
import re
from utils.translator import to_english

nlp = spacy.load("en_core_web_sm")


SYMPTOM_EXPANSION = {
    "sir dard": "headache",
    "headache": "headache",
    "pet dard": "stomach_pain",
    "ulti": "vomiting",
    "bukhar": "high_fever",
    "fever": "high_fever",
    "khansi": "cough",
    "cough": "cough",
    "sardi": "continuous_sneezing",
    "jalubu": "continuous_sneezing",
    "chakkar": "dizziness",
    "dizziness": "dizziness",
    "kamzori": "fatigue",
    "weakness": "fatigue",
    "fatigue": "fatigue",
    "chest pain": "chest_pain",
    "nausea": "nausea",
    "vomiting": "vomiting",
    "diarrhea": "diarrhoea",
    "loose motion": "diarrhoea",
    "runny nose": "runny_nose",
    "joint pain": "joint_pain",
    "skin rash": "skin_rash",
    "itching": "itching",
    "constipation": "constipation",
    "back pain": "back_pain",
    "breathlessness": "breathlessness",
    "sweating": "sweating",
    "dehydration": "dehydration",
    "indigestion": "indigestion",
    "yellowish skin": "yellowish_skin",
    "dark urine": "dark_urine",
    "loss of appetite": "loss_of_appetite",
    "pain behind eyes": "pain_behind_the_eyes",
    "sunken eyes": "sunken_eyes",
    "weakness in limbs": "weakness_in_limbs",
    "fast heart rate": "fast_heart_rate",
    "neck pain": "neck_pain",
    "cramps": "cramps",
    "bruising": "bruising",
    "obesity": "obesity",
    "swollen legs": "swollen_legs",
    "puffy face": "puffy_face_and_eyes",
    "enlarged thyroid": "enlarged_thyroid",
    "brittle nails": "brittle_nails",
    "excessive hunger": "excessive_hunger",
    "drying lips": "drying_and_tingling_lips",
    "slurred speech": "slurred_speech",
    "knee pain": "knee_pain",
    "hip joint pain": "hip_joint_pain",
    "muscle weakness": "muscle_weakness",
    "stiff neck": "stiff_neck",
    "swelling joints": "swelling_joints",
    "movement stiffness": "movement_stiffness",
    "spinning movements": "spinning_movements",
    "loss of balance": "loss_of_balance",
    "unsteadiness": "unsteadiness",
    "weakness of one body side": "weakness_of_one_body_side",
    "loss of smell": "loss_of_smell",
    "bladder discomfort": "bladder_discomfort",
    "foul smell of urine": "foul_smell_of_urine",
    "continuous feel of urine": "continuous_feel_of_urine",
    "passage of gases": "passage_of_gases",
    "internal itching": "internal_itching",
    "toxic look": "toxic_look_(typhos)",
    "depression": "depression",
    "irritability": "irritability",
    "muscle pain": "muscle_pain",
    "altered sensorium": "altered_sensorium",
    "red spots over body": "red_spots_over_body",
    "belly pain": "belly_pain",
    "abnormal menstruation": "abnormal_menstruation",
    "dischromic patches": "dischromic__patches",
    "watering from eyes": "watering_from_eyes",
    "increased appetite": "increased_appetite",
    "polyuria": "polyuria",
    "family history": "family_history",
    "mucoid sputum": "mucoid_sputum",
    "rusty sputum": "rusty_sputum",
    "lack of concentration": "lack_of_concentration",
    "visual disturbances": "visual_disturbances",
    "receiving blood transfusion": "receiving_blood_transfusion",
    "receiving unsterile injections": "receiving_unsterile_injections",
    "coma": "coma",
    "stomach bleeding": "stomach_bleeding",
    "distention of abdomen": "distention_of_abdomen",
    "history of alcohol consumption": "history_of_alcohol_consumption",
    "blood in sputum": "blood_in_sputum",
    "prominent veins on calf": "prominent_veins_on_calf",
    "palpitations": "palpitations",
    "painful walking": "painful_walking",
    "pus filled pimples": "pus_filled_pimples",
    "blackheads": "blackheads",
    "scurring": "scurring",
    "skin peeling": "skin_peeling",
    "silver like dusting": "silver_like_dusting",
    "small dents in nails": "small_dents_in_nails",
    "inflammatory nails": "inflammatory_nails",
    "blister": "blister",
    "red sore around nose": "red_sore_around_nose",
    "yellow crust ooze": "yellow_crust_ooze",
    "weight gain": "weight_gain",
    "anxiety": "anxiety",
    "cold hands and feets": "cold_hands_and_feets",
    "mood swings": "mood_swings",
    "weight loss": "weight_loss",
    "restlessness": "restlessness",
    "lethargy": "lethargy",
    "patches in throat": "patches_in_throat",
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
}


def phrase_exists(text, phrase):
    return phrase in text


def extract_symptoms_nlp(text, symptom_columns):

    if not text:
        return []

    # translate
    text, lang = to_english(text)
    text = re.sub(r'\s+', ' ', text.lower().strip())

    found = set()

    # ----------------------------
    # 1. EXPANSION (FIXED)
    # ----------------------------
    for key, value in SYMPTOM_EXPANSION.items():
        if phrase_exists(text, key):
            found.add(value)

    # ----------------------------
    # 2. DIRECT COLUMN MATCH
    # ----------------------------
    text_tokens = set(text.split())
    for col in symptom_columns:
        col_phrase = col.replace("_", " ")
        if col_phrase in text:
            found.add(col)

        col_tokens = set(col_phrase.split())
        if col_tokens and col_tokens.issubset(text_tokens):
            found.add(col)

    # ----------------------------
    # 3. NLP fallback
    # ----------------------------
    doc = nlp(text)

    for token in doc:
        if token.text in symptom_columns:
            found.add(token.text)

    return list(found)