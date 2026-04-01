from typing import Dict, Any, Literal, Optional
from copy import deepcopy

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import io
import base64
import logging
import hashlib
import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from i18n.language_manager import LanguageManager

from preprocessing.image_transforms import preprocess_image
from database.db import get_db
from database.models import AppUser, Prediction
from services.llm_service import get_detailed_explanation
from services.alert_service import can_send_alert, get_alert_threshold_percent
from services.email_service import send_alert_email
from services.treatment_service import get_treatment
from app.services.voice_assistant_service import (
	normalize_language,
	generate_answer,
	detect_intent,
	translate_text,
)

logger = logging.getLogger(__name__)

router = APIRouter()
language_manager = LanguageManager()

_DASHBOARD_TO_TRANSLATOR_CODE = {
	"en": "en",
	"hi": "hi",
	"od": "or",
	"or": "or",
	"ta": "ta",
	"te": "te",
	"bn": "bn",
	"gu": "gu",
	"mr": "mr",
	"kn": "kn",
	"ml": "ml",
}


def _resolve_translator_language_code(language_code: str) -> str:
	"""Map dashboard language code to deep-translator language code."""
	code = (language_code or "en").strip().lower()
	return _DASHBOARD_TO_TRANSLATOR_CODE.get(code, "en")


def _translate_to_english_if_needed(text: str, language_code: str) -> str:
	"""Translate user input text to English for stable intent detection.

	Falls back to original text if translation is unavailable.
	"""
	if not text:
		return ""
	translator_lang = _resolve_translator_language_code(language_code)
	if translator_lang == "en":
		return text
	try:
		from deep_translator import GoogleTranslator  # type: ignore
		translated = GoogleTranslator(source=translator_lang, target="en").translate(text)
		if isinstance(translated, str) and translated.strip():
			return translated.strip()
		translated_auto = GoogleTranslator(source="auto", target="en").translate(text)
		if isinstance(translated_auto, str) and translated_auto.strip():
			return translated_auto.strip()
	except Exception:
		logger.debug("Input translation to English failed; using original text.", exc_info=True)

	try:
		from googletrans import Translator  # type: ignore
		translator = Translator()
		translated = translator.translate(text, src=translator_lang, dest="en")
		translated_text = getattr(translated, "text", "")
		if isinstance(translated_text, str) and translated_text.strip():
			return translated_text.strip()
	except Exception:
		logger.debug("googletrans input translation failed; using original text.", exc_info=True)
	return text


def _translate_from_english(text: str, language_code: str) -> str:
	"""Translate English response text to requested dashboard language code.

	Uses generic language-code translation for full global dashboard support.
	Falls back to original English text when translation is unavailable.
	"""
	if not text:
		return ""
	translator_lang = _resolve_translator_language_code(language_code)
	if translator_lang == "en":
		return text
	try:
		from deep_translator import GoogleTranslator  # type: ignore
		translated = GoogleTranslator(source="en", target=translator_lang).translate(text)
		if isinstance(translated, str) and translated.strip():
			return translated.strip()
	except Exception:
		logger.debug("Output translation failed for language_code=%s; using English fallback.", language_code, exc_info=True)

	try:
		from googletrans import Translator  # type: ignore
		translator = Translator()
		translated = translator.translate(text, src="en", dest=translator_lang)
		translated_text = getattr(translated, "text", "")
		if isinstance(translated_text, str) and translated_text.strip():
			return translated_text.strip()
	except Exception:
		logger.debug("googletrans output translation failed for language_code=%s; using English fallback.", language_code, exc_info=True)
	return text


def _looks_mostly_english(text: str) -> bool:
	"""Heuristic: detect if text is predominantly ASCII/English-like."""
	if not text:
		return True
	ascii_chars = sum(1 for ch in text if ord(ch) < 128)
	ratio = ascii_chars / max(len(text), 1)
	return ratio > 0.85


def _native_language_fallback_answer(
	language_code: str,
	intent: str,
	disease: str,
	knowledge_answer: str,
) -> str:
	"""Generate native fallback answer for critical languages when translation fails."""
	code = (language_code or "en").lower()
	name = disease or "इस रोग"

	if code == "hi":
		intent_lines = {
			"treatment": f"{name} के लिए खेत की स्वच्छता रखें, संक्रमित पत्तियाँ हटाएँ, और जरूरत होने पर ही अनुशंसित दवा का उपयोग करें।",
			"cause": f"{name} का मुख्य कारण आमतौर पर नमी, संक्रमित अवशेष और अनुकूल मौसम होता है।",
			"symptoms": f"{name} में धब्बे, रंग बदलना, पत्ती सूखना या जंग जैसे लक्षण दिख सकते हैं।",
			"prevention": f"{name} की रोकथाम के लिए फसल चक्र, संतुलित पोषण और जल प्रबंधन अपनाएँ।",
		}
		base = knowledge_answer.strip() if knowledge_answer else intent_lines.get(
			intent,
			f"{name} के लिए नियमित निगरानी और समय पर प्रबंधन बहुत जरूरी है।",
		)
		return (
			"मैं आपकी चिंता समझता हूँ। "
			f"{base} "
			"पर्यावरण-अनुकूल सलाह: पहले जैविक विकल्प (नीम/ट्राइकोडर्मा), खेत की स्वच्छता और संतुलित पोषण अपनाएँ; "
			"रासायनिक उपाय केवल गंभीर स्थिति में विशेषज्ञ सलाह से करें।"
		)

	if code in {"od", "or"}:
		intent_lines = {
			"treatment": f"{name} ପାଇଁ ଖେତ ସଫା ରଖନ୍ତୁ, ଆକ୍ରାନ୍ତ ପତ୍ର କାଢ଼ନ୍ତୁ, ଏବଂ ଆବଶ୍ୟକ ହେଲେମାତ୍ର ସୁପାରିଶିତ ଔଷଧ ବ୍ୟବହାର କରନ୍ତୁ।",
			"cause": f"{name} ର ମୁଖ୍ୟ କାରଣ ସାଧାରଣତଃ ଆର୍ଦ୍ରତା, ଆକ୍ରାନ୍ତ ଅବଶିଷ୍ଟ ଏବଂ ଅନୁକୂଳ ପାଗ ଅଟେ।",
			"symptoms": f"{name} ରେ ଦାଗ, ରଙ୍ଗ ପରିବର୍ତ୍ତନ, ପତ୍ର ଶୁଖିବା କିମ୍ବା ରଷ୍ଟ ଭଳି ଲକ୍ଷଣ ଦେଖାଯାଏ।",
			"prevention": f"{name} ପ୍ରତିରୋଧ ପାଇଁ ଫସଲ ପରିବର୍ତ୍ତନ, ସନ୍ତୁଳିତ ପୋଷଣ ଏବଂ ଜଳ ପରିଚାଳନା କରନ୍ତୁ।",
		}
		base = knowledge_answer.strip() if knowledge_answer else intent_lines.get(
			intent,
			f"{name} ପାଇଁ ନିୟମିତ ନିରୀକ୍ଷଣ ଏବଂ ସମୟୋଚିତ ପରିଚାଳନା ଅତ୍ୟାବଶ୍ୟକ।",
		)
		return (
			"ମୁଁ ଆପଣଙ୍କ ଚିନ୍ତା ବୁଝୁଛି। "
			f"{base} "
			"ପରିବେଶ-ବନ୍ଧୁ ପରାମର୍ଶ: ପ୍ରଥମେ ଜୈବିକ ବିକଳ୍ପ (ନିମ୍/ଟ୍ରାଇକୋଡର୍ମା), ଖେତ ସଫାସୁଥା ଏବଂ ସନ୍ତୁଳିତ ପୋଷଣ ଅନୁସରଣ କରନ୍ତୁ; "
			"ରାସାୟନିକ ପଦ୍ଧତି କେବଳ ଗୁରୁତର ସ୍ଥିତିରେ ନିପୁଣ ପରାମର୍ଶରେ କରନ୍ତୁ।"
		)

	return knowledge_answer or "I understand your concern. Please continue with eco-friendly disease management practices."


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]
    gradcam_image: str
    model_version: Optional[str] = Field(
        None,
        description="Model version identifier used for this prediction (e.g. 'v1.0').",
    )
    device: Optional[str] = Field(
        None,
        description="Computation device used for inference (e.g. 'cuda', 'cpu').",
    )
    image_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash of the uploaded image bytes (hex-encoded).",
    )
    prediction_id: Optional[int] = Field(
        None,
        description="ID of the logged prediction. Use this ID to submit feedback via POST /feedback/{prediction_id}"
    )
    crop: Optional[str] = Field(
        None,
        description="Crop type used for prediction."
    )
    prediction: Optional[str] = Field(
        None,
        description="Disease prediction label."
    )
    eco_recommendation: Optional[Dict[str, Any]] = Field(
        None,
        description="Eco-friendly recommendations for the predicted disease."
    )

class FeedbackRequest(BaseModel):
	feedback: Literal["correct", "incorrect"] = Field(
		...,
		description="User feedback on prediction accuracy. Must be 'correct' or 'incorrect'."
	)


class FeedbackResponse(BaseModel):
	message: str
	prediction_id: int
	feedback: str


class ExplanationResponse(BaseModel):
	"""Structured farmer-friendly explanation for a predicted disease.

	By default this contains only deterministic knowledge base fields.
	When `detailed=true` is passed to the endpoint, an additional
	`detailed_explanation` field may be populated using the hybrid
	explanation engine (Gemini + DB cache).
	"""

	summary: str
	cause: str
	symptoms: str
	spread: str
	treatment: str
	prevention: str
	prediction_id: int
	crop: str
	disease: str
	detailed_explanation: Optional[str] = Field(
		default=None,
		description="Optional detailed LLM-generated explanation when ?detailed=true",
	)


class VoiceQueryRequest(BaseModel):
	question: str = Field(..., description="Farmer question captured from voice transcript")
	disease: str = Field(..., description="Already predicted disease label from previous step")
	language: str = Field("en", description="Target language: en, hi, or")


class VoiceQueryResponse(BaseModel):
	response: str


class ChatbotQueryRequest(BaseModel):
	question: str = Field(..., description="User chat question")
	disease: str = Field(..., description="Predicted disease label")
	crop: Optional[str] = Field(None, description="Predicted crop label")
	language_code: str = Field("en", description="Dashboard language code")


class ChatbotQueryResponse(BaseModel):
	response: str
	language_code: str
	disease: str


@router.get("/system-status")
async def system_status(request: Request) -> Dict[str, bool]:
	"""Report high-level system status for UI indicators.

	Returns JSON describing whether core components are available. This endpoint
	is read-only and safe for frequent polling from the UI.
	"""
	app = request.app
	registry = getattr(app.state, "registry", None)
	knowledge_base = getattr(app.state, "knowledge_base", None)

	model_loaded = bool(registry and getattr(registry, "is_loaded", lambda: False)())
	knowledge_base_loaded = knowledge_base is not None
	llm_enabled = bool(os.getenv("GEMINI_API_KEY"))

	return {
		"model_loaded": model_loaded,
		"knowledge_base_loaded": knowledge_base_loaded,
		"llm_enabled": llm_enabled,
	}

@router.get("/crops")
async def list_crops(request: Request) -> Dict[str, Any]:
	"""Return available crops for the UI dropdown.

	This endpoint is safe for production and does not modify application state.
	"""
	registry = getattr(request.app.state, "registry", None)
	if registry is None:
		# Keep response JSON-only and defensive: the UI can fall back to a default list.
		return {"crops": [], "warning": "Model registry not loaded"}
	try:
		return {"crops": registry.list_available_crops()}
	except Exception as e:
		logger.exception("Failed to list available crops: %s", e)
		return {"crops": [], "warning": "Failed to read available crops"}


@router.get("/i18n/languages")
async def list_languages() -> Dict[str, Any]:
	"""Return supported UI languages for dashboard selectors."""
	return {
		"default_language": language_manager.default_language,
		"languages": language_manager.get_supported_languages_detailed(),
	}


@router.get("/i18n/translations")
async def get_translations(
	language_code: str = Query("en", alias="language_code")
) -> Dict[str, Any]:
	"""Return translation dictionary for the requested language code/name."""
	resolved_code = language_manager.resolve_language_code(language_code)
	base = deepcopy(language_manager.get_language(language_manager.default_language) or {})
	selected = language_manager.get_language(resolved_code) or {}

	def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
		for key, value in src.items():
			if isinstance(value, dict) and isinstance(dst.get(key), dict):
				_deep_merge(dst[key], value)
			else:
				dst[key] = value
		return dst

	merged = _deep_merge(base, selected)
	return {
		"language_code": resolved_code,
		"translations": merged,
	}


@router.post("/predict/{crop}", response_model=PredictionResponse)
async def predict(
	crop: str,
	request: Request,
	background_tasks: BackgroundTasks,
	file: UploadFile = File(...),
	db: Session = Depends(get_db)
) -> Any:
	"""
	Predict disease from uploaded image using multi-crop model registry.

	Args:
		crop: Crop type from path (e.g., "rice", "corn", "potato", "wheat")
		request: FastAPI request object
		file: Uploaded image file
		db: Database session dependency

	Returns:
		PredictionResponse with label, confidence, probabilities, and GradCAM visualization
	
	Raises:
		HTTPException: If registry not loaded, crop unsupported, image invalid, or inference fails
	"""
	from app.recommendation import get_recommendation

	app = request.app

	# Get registry from app state
	registry = getattr(app.state, "registry", None)
	if registry is None:
		logger.error("Model registry not initialized")
		raise HTTPException(status_code=500, detail="Model registry not loaded")

	# Get model bundle for requested crop
	try:
		model_bundle = registry.get(crop)
		predictor = model_bundle["predictor"]
		gradcam = model_bundle["gradcam"]
	except ValueError as e:
		logger.error(f"Unsupported crop in path: {crop}")
		available_crops = registry.list_available_crops()
		return JSONResponse(
			status_code=400,
			content={
				"error": "Unsupported crop",
				"available_crops": available_crops
			}
		)

	# Read bytes in-memory
	try:
		content = await file.read()
		img = Image.open(io.BytesIO(content)).convert("RGB")
	except Exception as e:
		logger.exception("Failed to read uploaded image")
		raise HTTPException(status_code=400, detail="Invalid image")

	# Compute SHA-256 hash of image bytes
	image_hash = hashlib.sha256(content).hexdigest()
	logger.debug(f"Computed image hash: {image_hash[:16]}...")

	# Use predictor's clean API for main prediction
	try:
		prediction_result = predictor.predict(img)
		label = prediction_result["label"]
		top_prob = prediction_result["confidence"]
	except Exception:
		logger.exception("Model inference failed")
		raise HTTPException(status_code=500, detail="Inference error")

	# Prepare tensor for Grad-CAM visualization (needed separately)
	try:
		device_str = str(getattr(predictor, "device", "cpu"))
		tensor = preprocess_image(img, input_size=224, device=device_str)
	except Exception:
		logger.exception("Failed to preprocess image for Grad-CAM")
		raise HTTPException(status_code=500, detail="Preprocessing error")

	# Compute all probabilities for response (needed for full probability distribution)
	try:
		predictor.model.eval()
		with torch.no_grad():
			outputs = predictor.model(tensor)
			if isinstance(outputs, (list, tuple)):
				logits = outputs[0]
			else:
				logits = outputs
			probs = F.softmax(logits, dim=1).cpu().numpy()[0]
	except Exception:
		logger.exception("Failed to compute probability distribution")
		raise HTTPException(status_code=500, detail="Probability computation error")

	# Find top index for Grad-CAM
	top_idx = int(np.argmax(probs))

	# Build probabilities dict
	probabilities = {predictor.labels[i] if i < len(predictor.labels) else str(i): float(p) for i, p in enumerate(probs)}

	# Generate gradcam heatmap and overlay
	try:
		heatmap, predicted_class = gradcam.generate(tensor, target_class=top_idx)
		# Need original image as numpy uint8 matching model input size
		# Resize original to model input size
		orig_np = np.array(img.resize((224, 224), Image.Resampling.LANCZOS), dtype=np.uint8)
		overlay = gradcam.overlay_heatmap(orig_np, heatmap, alpha=0.4)

		# Encode overlay to JPEG in-memory
		is_success, buffer = cv2.imencode('.jpg', overlay)
		if not is_success:
			raise RuntimeError('Failed to encode gradcam image')
		gradcam_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
	except Exception:
		logger.exception("Grad-CAM generation failed")
		# Fallback: return blank transparent 1x1 image
		blank = Image.new('RGB', (1, 1), (255, 255, 255))
		buf = io.BytesIO()
		blank.save(buf, format='JPEG')
		gradcam_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

	# Save image and log prediction to database
	upload_dir = Path("uploads") / crop
	upload_dir.mkdir(parents=True, exist_ok=True)
	image_path = upload_dir / f"{image_hash}.jpg"
	
	# Save image only if it doesn't already exist
	if not image_path.exists():
		try:
			img.save(image_path, "JPEG", quality=85)
			logger.info(f"Saved image to {image_path}")
		except Exception as e:
			logger.warning(f"Failed to save image to {image_path}: {e}")
	else:
		logger.debug(f"Image already exists at {image_path}, skipping save")

	# Insert prediction record into database
	prediction = None
	try:
		prediction = Prediction(
			crop=crop,
			image_path=str(image_path),
			image_hash=image_hash,
			predicted_label=label,
			confidence=top_prob,
			model_version="v1.0",
			user_feedback=None
		)
		db.add(prediction)
		db.commit()
		db.refresh(prediction)
		logger.info(f"Logged prediction to database: {prediction}")
	except Exception as e:
		db.rollback()
		logger.error(f"Failed to log prediction to database: {e}", exc_info=True)
		# Continue execution - don't fail the request if logging fails

	# Surface analytics-friendly metadata to the UI without breaking existing contract.
	model_version = "v1.0"
	device_str = str(getattr(predictor, "device", "cpu"))
	predicted_label = label
	confidence = top_prob
	eco_data = get_recommendation(predicted_label.lower().replace(" ", "_"))

	try:
		session_email = str(request.session.get("email", "")).strip().lower()
		target_user = None
		if session_email:
			target_user = db.query(AppUser).filter(AppUser.email == session_email).first()

		if target_user and target_user.email:
			confidence_percent = float(confidence * 100.0 if confidence <= 1 else confidence)
			threshold = get_alert_threshold_percent()
			if confidence_percent >= threshold:
				if can_send_alert(target_user.id):
					treatment_info = get_treatment(predicted_label)
					background_tasks.add_task(
						send_alert_email,
						to_email=target_user.email,
						disease=predicted_label,
						confidence=confidence_percent,
						treatment=treatment_info["treatment"],
						prevention=treatment_info["prevention"],
					)
					logger.info(
						"Scheduled alert email for user_id=%s disease=%s confidence=%.2f%%",
						target_user.id,
						predicted_label,
						confidence_percent,
					)
				else:
					logger.info("Alert skipped due to cooldown for user_id=%s", target_user.id)
	except Exception:
		logger.exception("Alert scheduling failed; continuing prediction response")

	response = {
		"label": label,
		"confidence": confidence,
		"probabilities": probabilities,
		"gradcam_image": gradcam_b64,
		"model_version": model_version,
		"device": device_str,
		"image_hash": image_hash,
		"prediction_id": prediction.id if prediction else None,
		"crop": crop,
		"prediction": predicted_label,
		"eco_recommendation": eco_data,
	}

	logger.info(f"Prediction successful for crop '{crop}': {label} (confidence: {top_prob:.4f})")
	return JSONResponse(content=response)


@router.post("/feedback/{prediction_id}", response_model=FeedbackResponse)
async def submit_feedback(
	prediction_id: int,
	feedback_request: FeedbackRequest,
	db: Session = Depends(get_db)
) -> FeedbackResponse:
	"""
	Submit user feedback for a specific prediction.

	Args:
		prediction_id: ID of the prediction to provide feedback for
		feedback_request: Feedback request containing feedback value ("correct" or "incorrect")
		db: Database session dependency

	Returns:
		FeedbackResponse with success message, prediction ID, and feedback value

	Raises:
		HTTPException: If prediction not found or database operation fails
	"""
	logger.info(f"Received feedback request for prediction_id={prediction_id}, feedback={feedback_request.feedback}")

	# Retrieve prediction by ID
	prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
	
	if prediction is None:
		logger.warning(f"Prediction with id={prediction_id} not found")
		raise HTTPException(
			status_code=404,
			detail=f"Prediction with id {prediction_id} not found"
		)

	# Prevent double feedback overwrite
	if prediction.user_feedback is not None:
		logger.warning(
			f"Attempted to overwrite existing feedback for prediction_id={prediction_id}. "
			f"Current feedback: {prediction.user_feedback}"
		)
		raise HTTPException(
			status_code=400,
			detail="Feedback already submitted for this prediction."
		)

	# Update user_feedback field
	try:
		prediction.user_feedback = feedback_request.feedback
		db.commit()
		db.refresh(prediction)
		logger.info(
			f"Successfully updated feedback for prediction_id={prediction_id}: "
			f"feedback={feedback_request.feedback}"
		)
	except Exception as e:
		db.rollback()
		logger.error(
			f"Failed to update feedback for prediction_id={prediction_id}: {e}",
			exc_info=True
		)
		raise HTTPException(
			status_code=500,
			detail="Failed to record feedback. Please try again later."
		)

	return FeedbackResponse(
		message="Feedback recorded successfully",
		prediction_id=prediction.id,
		feedback=feedback_request.feedback
	)


@router.get("/explain-advanced/{prediction_id}", response_model=ExplanationResponse)
async def explain_advanced(
	prediction_id: int,
	request: Request,
	detailed: bool = Query(
		False,
		description=(
			"Set to true to request a detailed, LLM-generated explanation in "
			"addition to the deterministic knowledge base fields."
		),
	),
	language: str = Query(
		"English",
		description=(
			"Language for detailed AI explanation. Accepts language name or code (e.g., English/en, Hindi/hi). "
			"Defaults to English. Only used when detailed=true."
		),
	),
	db: Session = Depends(get_db)
) -> ExplanationResponse:
	"""
	Return structured disease explanation for a stored prediction (knowledge base only).

	Does not re-run model inference, reload models, or modify the registry.
	Fetches the prediction from the database and looks up explanation by crop and
	predicted_label. Designed for future extension (e.g. ?detailed=true with LLM).

	Args:
		prediction_id: ID of the logged prediction.
		request: FastAPI request (for app.state.knowledge_base).
		language: Language for detailed AI explanation (name or code). Defaults to English.
		db: Database session dependency.

	Returns:
		ExplanationResponse with summary, cause, symptoms, spread, treatment, prevention.

	Raises:
		HTTPException 404: Prediction not found.
		HTTPException 400: Knowledge not found for crop/disease.
		HTTPException 503: Knowledge base not loaded.
	"""
	knowledge_base = getattr(request.app.state, "knowledge_base", None)
	if knowledge_base is None:
		logger.error("Knowledge base not initialized")
		raise HTTPException(
			status_code=503,
			detail="Explanation service unavailable (knowledge base not loaded)."
		)

	prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
	if prediction is None:
		logger.warning("Prediction not found for explain-advanced: id=%s", prediction_id)
		raise HTTPException(
			status_code=404,
			detail=f"Prediction with id {prediction_id} not found."
		)

	crop = prediction.crop
	disease = prediction.predicted_label
	try:
		explanation = knowledge_base.get_explanation(crop, disease)
	except ValueError as e:
		logger.warning("Knowledge not found for explain-advanced: prediction_id=%s, crop=%s, disease=%s: %s", prediction_id, crop, disease, e)
		raise HTTPException(
			status_code=400,
			detail=str(e)
		) from e
	detailed_text: Optional[str] = None
	if detailed:
		try:
			_, resolved_llm_language = language_manager.resolve_language(language)
			detailed_text = get_detailed_explanation(
				db=db,
				crop=crop,
				disease=disease,
				base_explanation=explanation,
				language=resolved_llm_language,
			)
		except Exception:
			# Service layer is already defensive, but guard against any unexpected
			# failures to ensure we never break the endpoint.
			logger.exception(
				"Failed to generate detailed explanation for prediction_id=%s; "
				"returning base explanation only.",
				prediction_id,
			)
			detailed_text = None

	return ExplanationResponse(
		summary=explanation["summary"],
		cause=explanation["cause"],
		symptoms=explanation["symptoms"],
		spread=explanation["spread"],
		treatment=explanation["treatment"],
		prevention=explanation["prevention"],
		prediction_id=prediction.id,
		crop=crop,
		disease=disease,
		detailed_explanation=detailed_text,
	)


@router.post("/voice-query", response_model=VoiceQueryResponse)
async def voice_query(payload: VoiceQueryRequest) -> VoiceQueryResponse:
	"""Voice assistant endpoint for post-prediction disease Q&A."""
	question = (payload.question or "").strip()
	disease = (payload.disease or "").strip()
	lang = normalize_language(payload.language)

	if not question:
		raise HTTPException(status_code=400, detail="Question is required")
	if not disease:
		raise HTTPException(status_code=400, detail="Disease is required")

	try:
		base_answer = await run_in_threadpool(generate_answer, disease, question)
		translated_answer = await run_in_threadpool(translate_text, base_answer, lang)
		return VoiceQueryResponse(response=translated_answer)
	except HTTPException:
		raise
	except Exception as exc:
		logger.exception("Voice query processing failed: %s", exc)
		raise HTTPException(
			status_code=500,
			detail="Voice assistant failed to generate response. Please try again.",
		)


@router.post("/chatbot/query", response_model=ChatbotQueryResponse)
async def chatbot_query(request: Request, payload: ChatbotQueryRequest) -> ChatbotQueryResponse:
	"""Text chatbot endpoint for post-prediction disease Q&A.

	This endpoint is intentionally lightweight and reuses stable voice assistant
	logic for disease-aware responses while honoring dashboard language.
	"""
	question = (payload.question or "").strip()
	disease = (payload.disease or "").strip()
	language_code = language_manager.resolve_language_code(payload.language_code or "en")

	if not question:
		raise HTTPException(status_code=400, detail="Question is required")
	if not disease:
		raise HTTPException(status_code=400, detail="Disease is required")

	try:
		question_for_intent = await run_in_threadpool(_translate_to_english_if_needed, question, language_code)
		intent = detect_intent(question_for_intent)
		knowledge_base = getattr(request.app.state, "knowledge_base", None)
		friendly_intro = "I understand your concern."

		knowledge_answer = ""
		if knowledge_base is not None and payload.crop:
			try:
				explanation = knowledge_base.get_explanation(payload.crop, disease)
				if intent == "treatment":
					knowledge_answer = explanation.get("treatment", "")
				elif intent == "cause":
					knowledge_answer = explanation.get("cause", "")
				elif intent == "symptoms":
					knowledge_answer = explanation.get("symptoms", "")
				elif intent == "prevention":
					knowledge_answer = explanation.get("prevention", "")
				else:
					knowledge_answer = explanation.get("summary", "")
			except Exception:
				knowledge_answer = ""

		base_answer = knowledge_answer or await run_in_threadpool(generate_answer, disease, question_for_intent)
		eco_suffix = (
			" Eco-friendly advice: start with sanitation, balanced nutrition, biocontrol options "
			"(like neem/Trichoderma), and only use stronger chemicals when disease pressure is high "
			"and local experts recommend it."
		)
		friendly_answer = f"{friendly_intro} {base_answer}{eco_suffix}".strip()
		translated_answer = await run_in_threadpool(_translate_from_english, friendly_answer, language_code)

		# Strict language enforcement: avoid returning English when non-English is selected.
		if language_code != "en":
			failed_language_enforcement = (
				not translated_answer
				or translated_answer.strip().lower() == friendly_answer.strip().lower()
				or _looks_mostly_english(translated_answer)
			)
			if failed_language_enforcement:
				translated_answer = _native_language_fallback_answer(
					language_code=language_code,
					intent=intent,
					disease=disease,
					knowledge_answer=knowledge_answer,
				)

		return ChatbotQueryResponse(
			response=translated_answer or friendly_answer,
			language_code=language_code,
			disease=disease,
		)
	except HTTPException:
		raise
	except Exception as exc:
		logger.exception("Chatbot query processing failed: %s", exc)
		raise HTTPException(
			status_code=500,
			detail="Chatbot failed to generate response. Please try again.",
		)

