from __future__ import annotations
import json
import logging
import os
from typing import Any, Dict
from urllib import error, request

from sqlalchemy.orm import Session

from database.models import LLMExplanation
from i18n.language_manager import LanguageManager

logger = logging.getLogger("leaflens.llm")


GEMINI_API_URL: str = (
	"https://generativelanguage.googleapis.com/v1beta/"
	"models/gemini-1.5-pro:generateContent"
)
GEMINI_MODEL_NAME: str = "gemini-1.5-pro"
language_manager = LanguageManager()


def get_detailed_explanation(
	db: Session,
	crop: str,
	disease: str,
	base_explanation: Dict[str, str],
	language: str = "English",
) -> str:
	"""Return a detailed, farmer-friendly explanation using Gemini with DB caching.

	The function first checks the LLMExplanation cache table using the
	(crop, disease, language, model_name) tuple. If a cached explanation is found, it
	is returned immediately without calling the external API.

	If no cache entry exists, the function attempts to call the Gemini REST API
	using the GEMINI_API_KEY environment variable. On success, the generated
	text is stored in the cache table and returned. If the API key is missing,
	the HTTP request fails, or the response is malformed, a deterministic
	fallback explanation is generated purely from the structured knowledge base
	data and returned without creating a cache entry.

	Args:
		db: SQLAlchemy Session.
		crop: Crop identifier (e.g., "rice").
		disease: Disease name/label.
		base_explanation: Structured explanation dictionary from the knowledge
			base with keys: summary, cause, symptoms, spread, treatment, prevention.
		language: Language for the explanation (code or display name). Defaults to English.

	Returns:
		A single-paragraph explanation string, either from cache, Gemini, or
		a deterministic fallback.
	"""
	crop_key = crop.strip().lower()
	disease_key = disease.strip()
	_, language_key = language_manager.resolve_language(language)

	# 1) Try cache first (no external call if hit).
	cached = (
		db.query(LLMExplanation)
		.filter(
			LLMExplanation.crop == crop_key,
			LLMExplanation.disease == disease_key,
			LLMExplanation.language == language_key,
			LLMExplanation.model_name == GEMINI_MODEL_NAME,
		)
		.first()
	)
	if cached is not None:
		logger.info(
			"Returning cached LLM explanation for crop=%s, disease=%s, language=%s, model=%s",
			crop_key,
			disease_key,
			language_key,
			GEMINI_MODEL_NAME,
		)
		return cached.explanation_text

	# 2) Attempt Gemini call; on failure, fall back to deterministic expansion.
	api_key = os.getenv("GEMINI_API_KEY")
	if not api_key:
		logger.warning(
			"GEMINI_API_KEY is not set; returning deterministic fallback explanation "
			"without calling external LLM."
		)
		return _build_fallback_explanation(crop_key, disease_key, base_explanation)

	try:
		generated_text = _call_gemini_api(
			api_key=api_key,
			crop=crop_key,
			disease=disease_key,
			base_explanation=base_explanation,
			language=language_key,
		)
	except Exception:
		# Never propagate LLM errors to the API layer; always fall back safely.
		logger.exception(
			"Gemini API call failed for crop=%s, disease=%s; using deterministic fallback.",
			crop_key,
			disease_key,
		)
		return _build_fallback_explanation(crop_key, disease_key, base_explanation)

	if not generated_text:
		logger.warning(
			"Gemini returned empty text for crop=%s, disease=%s; using deterministic fallback.",
			crop_key,
			disease_key,
		)
		return _build_fallback_explanation(crop_key, disease_key, base_explanation)

	# Guard against short or malformed responses
	if len(generated_text) < 50:
		logger.warning(
			"Gemini returned text too short (%d chars) for crop=%s, disease=%s; using deterministic fallback.",
			len(generated_text),
			crop_key,
			disease_key,
		)
		return _build_fallback_explanation(crop_key, disease_key, base_explanation)

	# 3) Persist successful Gemini result to cache.
	try:
		entry = LLMExplanation(
			crop=crop_key,
			disease=disease_key,
			language=language_key,
			model_name=GEMINI_MODEL_NAME,
			explanation_text=generated_text,
		)
		db.add(entry)
		db.commit()
		db.refresh(entry)
		logger.info(
			"Cached new LLM explanation for crop=%s, disease=%s, language=%s, model=%s",
			crop_key,
			disease_key,
			language_key,
			GEMINI_MODEL_NAME,
		)
	except Exception:
		# Cache write failures must not break the API; log and continue.
		db.rollback()
		logger.exception(
			"Failed to cache LLM explanation for crop=%s, disease=%s", crop_key, disease_key
		)

	return generated_text


def _build_system_instruction(language: str) -> str:
    """
    Build a strict language-enforced system instruction for Gemini.

    This version strongly forces the model to respond ONLY in the requested
    language and prevents language mixing.
    """
    return f"""
You are a professional agricultural plant disease expert helping farmers.

CRITICAL LANGUAGE RULE:
You MUST respond strictly and ONLY in {language}.
Do NOT use English words unless the requested language is English.
Do NOT mix languages.
Do NOT translate your answer.
Do NOT provide the answer in multiple languages.

If the requested language is:
- Hindi → respond fully in Hindi script.
- Odia → respond fully in Odia script.
- Telugu → respond fully in Telugu script.
- Tamil → respond fully in Tamil script.
- Bengali → respond fully in Bengali script.
- Gujarati → respond fully in Gujarati script.
- Marathi → respond fully in Marathi script.
- Kannada → respond fully in Kannada script.
- Malayalam → respond fully in Malayalam script.
- English → respond fully in English.

CONTENT RULES:
- Provide clear, practical, farmer-friendly advice.
- Do not invent new facts.
- Use short and simple sentences.
- Keep explanations structured and easy to understand.

Your entire response must follow the requested language strictly.
"""

def _build_user_prompt(
	crop: str,
	disease: str,
	summary: str,
	cause: str,
	symptoms: str,
	spread: str,
	treatment: str,
	prevention: str,
	language: str,
) -> str:
	"""Build a language-specific user prompt for disease explanation."""
	base_content = (
		f"Crop: {crop}\n"
		f"Disease: {disease}\n"
		f"Summary: {summary}\n"
		f"Cause: {cause}\n"
		f"Symptoms: {symptoms}\n"
		f"Spread: {spread}\n"
		f"Treatment: {treatment}\n"
		f"Prevention: {prevention}\n\n"
	)

	return (
		base_content
		+ "Provide a comprehensive agricultural explanation structured as:\n"
		+ "- Disease Name\n"
		+ "- Cause\n"
		+ "- Symptoms (Visual signs)\n"
		+ "- Prevention (Practical steps)\n"
		+ "- Treatment (Recommended actions)\n"
		+ "- Government Schemes (if applicable)\n\n"
		+ f"Write the complete response strictly in {language}. "
		+ "Use a simple, farmer-friendly style with short practical sentences."
	)


def _call_gemini_api(
    api_key: str,
    crop: str,
    disease: str,
    base_explanation: Dict[str, str],
    language: str = "English",
    timeout_seconds: float = 10.0,
) -> str:
    """Call the Gemini REST API and return the generated text in the specified language.

    This helper is intentionally narrow in scope: it constructs the request
    payload, performs an HTTPS POST with a sane timeout, and parses the first
    candidate's first text part. Any network or parsing errors are raised to
    the caller to be handled centrally.

    Args:
        api_key: Gemini API key.
        crop: Crop name.
        disease: Disease name.
        base_explanation: Structured explanation dictionary from knowledge base.
		language: Language for explanation. Defaults to English.
        timeout_seconds: HTTP request timeout in seconds.

    Returns:
        Generated explanation text in the specified language.
    """
    # Build structured prompt content from deterministic knowledge.
    summary = base_explanation.get("summary", "")
    cause = base_explanation.get("cause", "")
    symptoms = base_explanation.get("symptoms", "")
    spread = base_explanation.get("spread", "")
    treatment = base_explanation.get("treatment", "")
    prevention = base_explanation.get("prevention", "")

    # Create language-specific system instruction and user prompt
    system_instruction = _build_system_instruction(language)
    user_content = _build_user_prompt(crop, disease, summary, cause, symptoms, spread, treatment, prevention, language)

    body: Dict[str, Any] = {
        "system_instruction": {
            "role": "system",
            "parts": [{"text": system_instruction}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_content}],
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 512,
        },
    }

    # Use API key via query parameter; never log the full URL to avoid leaking it.
    url = f"{GEMINI_API_URL}?key={api_key}"
    data = json.dumps(body).encode("utf-8")

    http_request = request.Request(
        url=url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
        },
    )

    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as resp:
            resp_body = resp.read().decode("utf-8")
    except error.HTTPError as e:
        # Avoid logging response body, which could contain sensitive information.
        logger.error(
            "Gemini HTTP error: status=%s, reason=%s", e.code, getattr(e, "reason", "")
        )
        raise
    except error.URLError as e:
        logger.error("Gemini request failed: %s", e.reason)
        raise

    try:
        payload = json.loads(resp_body)
    except json.JSONDecodeError as e:
        logger.error("Failed to decode Gemini response JSON: %s", e)
        raise

    text = _extract_text_from_gemini_response(payload)
    if not text:
        logger.error("Gemini response did not contain any text candidates.")
        raise RuntimeError("Gemini response missing text candidates")

    return text.strip()

def _extract_text_from_gemini_response(payload: Dict[str, Any]) -> str:
	"""Extract the first text candidate from a Gemini generateContent response."""
	try:
		candidates = payload.get("candidates") or []
		if not candidates:
			return ""
		first = candidates[0] or {}
		content = first.get("content") or {}
		parts = content.get("parts") or []
		if not parts:
			return ""
		text = parts[0].get("text") or ""
		return text
	except Exception:
		# Be defensive: never let a parsing bug propagate.
		logger.exception("Unexpected structure in Gemini response payload.")
		return ""


def _build_fallback_explanation(
	crop: str,
	disease: str,
	base_explanation: Dict[str, str],
) -> str:
	"""Build a deterministic, human-readable explanation from base knowledge.

	This function never calls external services and is safe to use when the
	LLM is unavailable. It preserves the same high-level content as the
	knowledge base but in a single-paragraph narrative form.
	"""
	summary = base_explanation.get("summary", "").strip()
	cause = base_explanation.get("cause", "").strip()
	symptoms = base_explanation.get("symptoms", "").strip()
	spread = base_explanation.get("spread", "").strip()
	treatment = base_explanation.get("treatment", "").strip()
	prevention = base_explanation.get("prevention", "").strip()

	parts = [
		f"For {crop} leaves affected by {disease}, farmers should understand the full picture.",
	]
	if summary:
		parts.append(summary)
	if cause:
		parts.append(f"The main cause of this disease is: {cause}")
	if symptoms:
		parts.append(f"Typical symptoms you may see in the field include: {symptoms}")
	if spread:
		parts.append(f"The disease can spread in the following ways: {spread}")
	if treatment:
		parts.append(f"For treatment, consider the following practical steps: {treatment}")
	if prevention:
		parts.append(
			f"To prevent future outbreaks, focus on these preventive practices: {prevention}"
		)

	return " ".join(part for part in parts if part)

