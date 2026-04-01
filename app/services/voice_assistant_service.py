"""
Voice assistant service for post-prediction disease Q&A.

This module is intentionally rule-based and lightweight so responses are fast,
predictable, and reliable in hackathon/demo environments.
"""

from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger("leaflens")

SupportedLanguage = Literal["en", "hi", "or"]

_LANGUAGE_CODE_MAP = {
    "en": "en",
    "english": "en",
    "hi": "hi",
    "hindi": "hi",
    "हिंदी": "hi",
    "hindī": "hi",
    "hindī (hindi)": "hi",
    "od": "or",
    "or": "or",
    "odia": "or",
    "oriya": "or",
    "ଓଡ଼ିଆ": "or",
}


def normalize_language(language: str | None) -> SupportedLanguage:
    """Normalize API language input to one of: en, hi, or."""
    if not language:
        return "en"

    normalized = _LANGUAGE_CODE_MAP.get(str(language).strip().lower(), "en")
    return normalized  # type: ignore[return-value]


def detect_intent(question: str) -> str:
    """Detect user intent using multilingual keyword heuristics."""
    if not question:
        return "general"

    text = question.strip().lower()

    treatment_terms = [
        "treatment", "cure", "medicine", "spray", "control", "manage",
        "इलाज", "उपचार", "दवा", "स्प्रे",
        "ଚିକିତ୍ସା", "ଔଷଧ", "ସ୍ପ୍ରେ",
    ]
    cause_terms = [
        "cause", "why", "reason", "source", "how came",
        "कारण", "क्यों", "वजह",
        "କାରଣ", "କାହିଁକି",
    ]
    prevention_terms = [
        "prevent", "prevention", "avoid", "protect", "stop",
        "रोकथाम", "बचाव", "रोकें",
        "ପ୍ରତିରୋଧ", "ବଞ୍ଚିବା", "ରୋକ",
    ]
    symptom_terms = [
        "symptom", "symptoms", "sign", "signs", "identify", "how to identify",
        "लक्षण", "निशान", "कैसे पहचानें", "pehchan", "lakshan", "lakhyan",
        "ଲକ୍ଷଣ", "ଚିହ୍ନ", "କେମିତି ଚିହ୍ନିବେ", "lakshana",
    ]

    if any(token in text for token in treatment_terms):
        return "treatment"
    if any(token in text for token in cause_terms):
        return "cause"
    if any(token in text for token in symptom_terms):
        return "symptoms"
    if any(token in text for token in prevention_terms):
        return "prevention"
    return "general"


_DISEASE_ALIAS_MAP = {
    "cercospora_leaf_spot": "corn_cercospora_leaf_spot",
    "common_rust": "corn_common_rust",
    "northern_leaf_blight": "corn_northern_leaf_blight",
    "corn_healthy": "corn_healthy",
    "bacterial_blight": "rice_bacterial_blight",
    "blast": "rice_blast",
    "brown_spot": "rice_brown_spot",
    "tungro": "rice_tungro",
    "black_rust": "wheat_black_rust",
    "brown_rust": "wheat_brown_rust",
    "wheat_healthy": "wheat_healthy",
    "yellow_rust": "wheat_yellow_rust",
    "healthy": "generic_healthy",
}

_DISEASE_KNOWLEDGE = {
    "corn_cercospora_leaf_spot": {
        "treatment": "For corn Cercospora leaf spot, remove infected lower leaves, improve field airflow, and use recommended fungicide only when disease pressure is high.",
        "cause": "Cercospora leaf spot in corn spreads in warm, humid weather and survives on infected crop residue.",
        "prevention": "Use crop rotation, residue management, balanced nutrition, and avoid prolonged leaf wetness to prevent Cercospora leaf spot.",
        "general": "Corn Cercospora leaf spot can reduce photosynthesis and yield; early monitoring and clean field practice are important.",
    },
    "corn_common_rust": {
        "treatment": "For corn common rust, monitor pustules early, remove heavily affected leaves if possible, and apply need-based fungicide as per local advisory.",
        "cause": "Corn common rust is caused by fungal spores that spread quickly through wind under moderate temperatures and humidity.",
        "prevention": "Use resistant hybrids, maintain proper spacing, avoid excess nitrogen, and scout fields regularly for early rust symptoms.",
        "general": "Corn common rust can spread fast in favorable weather, so frequent scouting and timely intervention are key.",
    },
    "corn_northern_leaf_blight": {
        "treatment": "For northern leaf blight in corn, remove severely infected leaves, manage residue, and follow fungicide recommendations when lesions expand rapidly.",
        "cause": "Northern leaf blight develops from fungal inoculum in residues, especially during cool to moderately warm humid conditions.",
        "prevention": "Rotate crops, use tolerant varieties, and keep field sanitation strong to reduce northern leaf blight risk.",
        "general": "Northern leaf blight can damage leaf area significantly, so early detection helps protect yield.",
    },
    "corn_healthy": {
        "treatment": "Your corn leaf appears healthy; no treatment is needed right now. Continue routine crop monitoring.",
        "cause": "No disease symptoms are detected in this corn sample.",
        "prevention": "To keep corn healthy, maintain balanced fertilizer, proper irrigation, weed control, and regular field scouting.",
        "general": "Great news, corn is healthy. Keep preventive agronomy practices and periodic observation.",
    },
    "rice_bacterial_blight": {
        "treatment": "For rice bacterial blight, avoid excess nitrogen, drain stagnant water when possible, and use recommended bactericide practices from local extension guidance.",
        "cause": "Rice bacterial blight is caused by bacterial infection that spreads through rain splash, irrigation water, and wounds on leaves.",
        "prevention": "Use resistant varieties, clean seed, proper spacing, and avoid field-to-field spread through contaminated water.",
        "general": "Rice bacterial blight can spread quickly in wet weather, so strict field hygiene and balanced nutrition are essential.",
    },
    "rice_blast": {
        "treatment": "For rice blast, remove infected leaves and use need-based fungicide at the correct growth stage as advised locally.",
        "cause": "Rice blast is fungal and favors high humidity, leaf wetness, dense canopy, and imbalanced nitrogen use.",
        "prevention": "Use clean seed, resistant varieties, proper spacing, and balanced nutrients to reduce rice blast outbreaks.",
        "general": "Rice blast can cause serious yield loss if unmanaged; early stage monitoring is very important.",
    },
    "rice_brown_spot": {
        "treatment": "For rice brown spot, improve plant nutrition, especially potassium balance, and use suitable fungicide only when needed.",
        "cause": "Brown spot in rice is often linked to fungal infection under nutrient stress, drought stress, and poor field conditions.",
        "prevention": "Use healthy seed, balanced fertilization, and proper water management to prevent rice brown spot.",
        "general": "Rice brown spot is associated with stress-prone fields; improving crop vigor greatly reduces risk.",
    },
    "rice_tungro": {
        "treatment": "For rice tungro, rogue infected plants early and manage vector insects like leafhoppers under integrated pest management guidance.",
        "cause": "Rice tungro is a viral disease transmitted mainly by leafhoppers, not by fungus.",
        "prevention": "Use resistant rice varieties, synchronized planting, vector control, and removal of volunteer host plants.",
        "general": "Rice tungro can spread rapidly through vectors, so early vector management is crucial.",
    },
    "wheat_black_rust": {
        "treatment": "For wheat black rust, monitor severity and apply recommended fungicide if threshold is crossed; remove volunteer hosts around fields.",
        "cause": "Wheat black rust is a fungal rust disease that spreads through airborne spores under favorable moisture and temperature.",
        "prevention": "Grow resistant wheat varieties, avoid late sowing where possible, and monitor rust-prone periods closely.",
        "general": "Wheat black rust can progress quickly once established, so timely scouting is important.",
    },
    "wheat_brown_rust": {
        "treatment": "For wheat brown rust, use need-based fungicide strategy and maintain crop nutrition to reduce disease impact.",
        "cause": "Wheat brown rust is caused by fungal spores that spread by wind and thrive under moderate humidity.",
        "prevention": "Use resistant cultivars, monitor early rust pustules, and follow integrated rust management practices.",
        "general": "Wheat brown rust can reduce grain filling if unmanaged; regular field checks help prevent losses.",
    },
    "wheat_healthy": {
        "treatment": "Your wheat leaf appears healthy, so no disease treatment is needed now.",
        "cause": "No disease symptoms are currently detected in this wheat sample.",
        "prevention": "Maintain healthy wheat through balanced nutrition, proper irrigation, and regular scouting.",
        "general": "Wheat looks healthy. Continue preventive practices and periodic observation.",
    },
    "wheat_yellow_rust": {
        "treatment": "For wheat yellow rust, start early control based on local advisory and apply recommended fungicide at correct stage if needed.",
        "cause": "Wheat yellow rust is a fungal rust favored by cool, humid weather and can spread rapidly by wind.",
        "prevention": "Use resistant varieties, timely sowing, and early surveillance during cool humid periods to prevent yellow rust spread.",
        "general": "Wheat yellow rust can spread in patches very fast; early action protects yield.",
    },
    "generic_healthy": {
        "treatment": "The crop appears healthy, so no treatment is needed right now.",
        "cause": "No disease symptoms are detected in the current sample.",
        "prevention": "Continue balanced nutrition, proper irrigation, and routine scouting to keep the crop healthy.",
        "general": "Great news, this leaf appears healthy. Continue preventive field management and regular observation.",
    },
}


def _normalize_disease_key(disease: str) -> str:
    normalized = "_".join(str(disease or "").strip().lower().replace("-", " ").split())
    return _DISEASE_ALIAS_MAP.get(normalized, normalized)


def generate_answer(disease: str, question: str) -> str:
    """Generate professional, eco-friendly, farmer-friendly answers."""
    safe_disease = (disease or "this disease").strip()
    intent = detect_intent(question)

    disease_key = _normalize_disease_key(safe_disease)
    disease_data = _DISEASE_KNOWLEDGE.get(disease_key)
    if disease_data:
        base = disease_data.get(intent, disease_data["general"])
        eco_suffix = (
            " Eco-friendly focus: keep field sanitation strong, prefer bio-solutions "
            "such as neem or Trichoderma options first, and use chemical spray only when necessary "
            "as per local agriculture guidance."
        )
        if intent in {"treatment", "prevention", "general", "symptoms"}:
            return f"{base}{eco_suffix}"
        return base

    if intent == "symptoms":
        return (
            f"For {safe_disease}, monitor visual signs like spots, discoloration, rust-like pustules, leaf drying, "
            "and unusual spreading patterns. Compare new symptoms every 2 to 3 days to track progression. "
            "Eco-friendly focus: remove infected leaves early and avoid overhead irrigation to reduce spread."
        )

    if intent == "treatment":
        return (
            f"For {safe_disease}, remove heavily infected leaves first. "
            "Use bio-fungicides like neem-based or Trichoderma products, keep proper plant spacing, "
            "and spray only as recommended by your local agriculture advisor. "
            "Start with eco-friendly options and reserve stronger chemical control for severe infection only."
        )

    if intent == "cause":
        return (
            f"{safe_disease} usually spreads due to infected seed or crop residue, high humidity, "
            "and poor air movement in the field. Wet leaves for long hours increase disease risk."
        )

    if intent == "prevention":
        return (
            f"To prevent {safe_disease}, use certified seed, rotate crops, avoid over-irrigation, "
            "and keep the field clean from old infected plant parts. Early monitoring helps a lot. "
            "Eco-friendly focus: improve airflow, soil health, and balanced nutrition to naturally reduce disease pressure."
        )

    return (
        f"{safe_disease} can reduce yield if ignored. Start with field sanitation, balanced nutrition, "
        "and timely eco-friendly management. For better support, ask about symptoms, treatment, cause, or prevention."
    )


def translate_text(text: str, language: SupportedLanguage) -> str:
    """
    Translate text to target language for voice output.
    Supports: en (English), hi (Hindi), or (Odia)
    Returns original text if translation fails.
    """
    if not text:
        return ""
    if language == "en":
        return text

    logger.info(f"Translating voice response to language: {language}")

    translator_target_map = {
        "en": "en",
        "hi": "hi",
        "or": "or",
    }
    target_code = translator_target_map.get(language, "en")

    # Try deep-translator for dynamic translation (works with Python 3.13+)
    try:
        from deep_translator import GoogleTranslator  # type: ignore

        logger.debug(f"Attempting translation to '{language}' for {len(text)} characters")

        # deep_translator language codes: 'en', 'hi', 'or' (Odia)
        translator = GoogleTranslator(source='en', target=target_code)
        translated = translator.translate(text)
        if translated:
            logger.info(f"✓ Translation successful: {language} - {len(translated)} characters")
            return translated
    except ImportError as e:
        logger.warning(f"⚠️ deep-translator not installed: {e}")
    except Exception as exc:
        logger.error(f"❌ Translation error for language '{language}': {type(exc).__name__}: {exc}")

    # Secondary fallback: googletrans (keeps behavior resilient when one provider fails)
    try:
        from googletrans import Translator  # type: ignore

        translator = Translator()
        translated = translator.translate(text, src="en", dest=target_code)
        translated_text = getattr(translated, "text", "")
        if isinstance(translated_text, str) and translated_text.strip():
            logger.info(f"✓ googletrans fallback successful: {language} - {len(translated_text)} characters")
            return translated_text.strip()
    except Exception as exc:
        logger.error(f"❌ googletrans fallback error for language '{language}': {type(exc).__name__}: {exc}")

    logger.warning(f"⚠️ Returning English text as translation fallback for language: {language}")
    return text
