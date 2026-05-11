import telebot
import os
from dotenv import load_dotenv

from main import run_chatbot, extract_all_symptoms
from utils.db import save_user_query
from utils.translator import detect_language_custom

# ----------------------------
# Load environment
# ----------------------------
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not TOKEN:
    raise ValueError("❌ TELEGRAM_BOT_TOKEN not found")

bot = telebot.TeleBot(TOKEN)

LANGUAGE_GREETINGS = {
    "en": "👋 Hello! I'm your AI Health Assistant 🤖\n\nJust type your symptoms:\n• fever and cough\n• jwaram and talanoppi\n• bukhar aur khansi",
    "hi": "👋 नमस्ते! मैं आपका AI हेल्थ असिस्टेंट हूँ 🤖\n\nकृपया अपने लक्षण लिखें:\n• बुखार और खांसी\n• पेट दर्द\n• चक्कर आना",
    "hinglish": "👋 नमस्ते! मैं आपका AI हेल्थ असिस्टेंट हूँ 🤖\n\nअपने लक्षण लिखें:\n• bukhar aur khansi\n• pet dard\n• sar dard",
    "te": "👋 నమస్తే! నేను మీ AI ఆరోగ్య సహాయకుడు 🤖\n\nదయచేసి మీ లక్షణాలను టైప్ చేయండి:\n• జ్వరం మరియు దగ్గు\n• పొట్ట నొప్పి\n• తలనొప్పి",
    "tenglish": "👋 Namaskaram! Nenu mee AI aarogya sahayakudu 🤖\n\nMee lakshanaalu type cheyandi:\n• jwaram and daggu\n• pet dard\n• thala noppi",
}

HELP_MESSAGES = {
    "en": "💡 Need help? Send your symptoms like:\n• fever and cough\n• itching and skin rash\n• pet dard aur bukhar",
    "hi": "💡 मदद चाहिए? अपने लक्षण भेजें जैसे:\n• बुखार और खांसी\n• खुजली और दाने\n• पेट दर्द और बुखार",
    "hinglish": "💡 Help chahiye? apne symptoms bheje jaise:\n• bukhar aur khansi\n• itching aur skin rash\n• pet dard aur bukhar",
    "te": "💡 సహాయం కావాలా? మీ లక్షణాలను పంపండి ఇలా:\n• జ్వరం మరియు దగ్గు\n• చర్మంపై బొజ్జలు మరియు కొంకరిపులం\n• పొట్ట నొప్పి మరియు జ్వరం",
    "tenglish": "💡 Help kavala? mee symptoms pampandi ila:\n• jwaram and daggu\n• itching and skin rash\n• pet dard and jwaram",
}

FAREWELL_MESSAGES = {
    "en": "🙂 Take care! If you have more symptoms, just send them anytime.",
    "hi": "🙂 ध्यान रखें! यदि आपके और लक्षण हों, तो कभी भी भेजें।",
    "hinglish": "🙂 Dhyan rakho! agar aur symptoms ho to kabhi bhi bhejo.",
    "te": "🙂 జాగ్రత్తగా ఉండండి! మీకు ఇంకు లక్షణాలు ఉంటే ఎప్పుడైనా పంపండి.",
    "tenglish": "🙂 Jagratta ga undandi! inka symptoms unna eppudaina pampandi.",
}

GREETINGS = {
    "en": ["hello", "hi", "hey", "good morning", "good evening", "good afternoon"],
    "hi": ["नमस्ते", "हैलो", "हेलो", "नमस्कार", "हाय", "सुप्रभात", "शुभ संध्या"],
    "hinglish": ["namaste", "hello", "hi", "hey", "namaskar", "salaam"],
    "te": ["నమస్తే", "హాయ్", "హలో", "నమస్కారం", "శుభోదయం", "శుభసాయంకాలం"],
    "tenglish": ["namaste", "namaskaram", "hello", "hi", "hey"],
}

FAREWELLS = {
    "en": ["bye", "goodbye", "see you", "thanks", "thank you"],
    "hi": ["बाय", "अलविदा", "धन्यवाद", "शुक्रिया"],
    "hinglish": ["bye", "goodbye", "thanks", "thank you"],
    "te": ["బై", "అలవిడా", "ధన్యవాదాలు", "థ్యాన్‌క్యూ"],
    "tenglish": ["bye", "goodbye", "thanks", "thank you"],
}


def get_localized_message(message, lang):
    return message if lang == "en" else message


def get_start_message(lang):
    return LANGUAGE_GREETINGS.get(lang, LANGUAGE_GREETINGS["en"])


def get_help_message(lang):
    return HELP_MESSAGES.get(lang, HELP_MESSAGES["en"])


def get_farewell_message(lang):
    return FAREWELL_MESSAGES.get(lang, FAREWELL_MESSAGES["en"])


import re

def detect_simple_message_type(text, lang):
    lower_text = text.lower().strip()

    for greeting in GREETINGS.get(lang, GREETINGS["en"]):
        if re.search(rf"\b{re.escape(greeting)}\b", lower_text):
            return "greeting"

    for farewell in FAREWELLS.get(lang, FAREWELLS["en"]):
        if re.search(rf"\b{re.escape(farewell)}\b", lower_text):
            return "farewell"

    if re.search(r"\b(help|/help|सहायता|help me|సహాయం|help please)\b", lower_text):
        return "help"

    return None


# ----------------------------
# Start Command
# ----------------------------
@bot.message_handler(commands=['start'])
def start(message):
    lang = detect_language_custom(message.text or "")
    bot.send_message(message.chat.id, get_start_message(lang))


# ----------------------------
# MAIN HANDLER
# ----------------------------
@bot.message_handler(func=lambda message: True)
def handle_message(message):

    user_input = message.text.strip()

    if not user_input:
        bot.send_message(message.chat.id, "❌ Please enter symptoms.")
        return

    user_id = message.from_user.id
    username = message.from_user.username or "NoUsername"
    full_name = message.from_user.first_name or "Unknown"

    try:
        print("\n==============================")
        print("📩 INPUT:", user_input)

        lang = detect_language_custom(user_input)
        intent = detect_simple_message_type(user_input, lang)

        if intent == "greeting":
            bot.send_message(message.chat.id, get_start_message(lang))
            return

        if intent == "help":
            bot.send_message(message.chat.id, get_help_message(lang))
            return

        if intent == "farewell":
            bot.send_message(message.chat.id, get_farewell_message(lang))
            return

        # ----------------------------
        # STEP 1: AI PIPELINE (MAIN LOGIC)
        # ----------------------------
        response = run_chatbot(user_input)

        if not response:
            response = "😕 Unable to process symptoms. Try again."

        print("📤 RESPONSE GENERATED")

        # ----------------------------
        # STEP 2: Send response immediately
        # ----------------------------
        bot.send_chat_action(message.chat.id, "typing")
        bot.send_message(message.chat.id, response)

        # ----------------------------
        # STEP 3: Extract symptoms (after response)
        # ----------------------------
        symptoms = extract_all_symptoms(user_input)

        print("🧾 SYMPTOMS:", symptoms)

        # ----------------------------
        # STEP 4: Save to DB safely
        # ----------------------------
        try:
            save_user_query(
                user_id=user_id,
                username=username,
                full_name=full_name,
                symptoms=symptoms,
                response=response
            )
            print("💾 DB SAVED SUCCESSFULLY")

        except Exception as db_error:
            print("❌ DB ERROR (ignored):", db_error)

        print("==============================\n")

    except Exception as e:
        print("❌ BOT ERROR:", e)

        bot.send_message(
            message.chat.id,
            "⚠️ Server error occurred. Please try again."
        )


# ----------------------------
# RUN BOT
# ----------------------------
print("🤖 Telegram Bot Running...")
bot.infinity_polling(skip_pending=True)