import logging
import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from app.recommendation import get_recommendation
from app.api import router as api_router
from app.registry import ModelRegistry
from app.routers.iot import router as iot_router
from app.routers.weather import router as weather_router
from app.ui.routes import router as ui_router
from database.db import init_db
from explainability.knowledge_base import KnowledgeBase

logger = logging.getLogger("leaflens")
logging.basicConfig(level=logging.INFO)


def create_app() -> FastAPI:
    """Create and configure FastAPI application with DB and model registry."""
    app = FastAPI(title="LeafLens - Crop Disease Detection")
    app.add_middleware(
        SessionMiddleware,
        secret_key=os.getenv("LEAFLENS_SESSION_SECRET", "leaflens-dev-session-secret"),
        same_site="lax",
        https_only=False,
    )

    @app.middleware("http")
    async def add_no_cache_headers(request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/static/") or path in {"/insight", "/weather", "/chatbot", "/"}:
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    # UI configuration (static + templates). Keep UI and JSON APIs separated.
    app.mount("/static", StaticFiles(directory="app/ui/static"), name="static")
    templates = Jinja2Templates(directory="app/ui/templates")
    app.state.templates = templates

    @app.on_event("startup")
    async def startup_load_resources():
        """Initialize database and load all crop models at startup."""
        logger.info("Initializing database...")
        try:
            init_db()
            logger.info("✓ Database initialized successfully.")
        except Exception as e:
            logger.exception(f"✗ Failed to initialize database: {e}")
            raise

        logger.info("Initializing model registry and loading all crop models...")
        try:
            registry = ModelRegistry()
            registry.load_models()
            app.state.registry = registry

            available_crops = registry.list_available_crops()
            logger.info(
                f"✓ Model registry loaded successfully. "
                f"Available crops: {available_crops}"
            )
        except Exception as e:
            logger.exception(f"✗ Failed to load models at startup: {e}")
            app.state.registry = None
            raise

        # Knowledge base logic (UNCHANGED)
        knowledge_path = Path("config/disease_knowledge.json")
        kb_loader_path = knowledge_path

        knowledge_dir = Path("knowledge")
        if knowledge_dir.exists() and knowledge_dir.is_dir():
            merged = {}

            if knowledge_path.exists():
                try:
                    with open(knowledge_path, "r", encoding="utf-8") as f:
                        base = json.load(f)
                        if isinstance(base, dict):
                            merged.update(base)
                except Exception:
                    logger.warning("Failed to read base disease_knowledge.json.")

            for crop_dir in knowledge_dir.iterdir():
                if not crop_dir.is_dir():
                    continue
                crop_key = crop_dir.name
                diseases = {}

                for jf in crop_dir.glob("*.json"):
                    try:
                        with open(jf, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        diseases[jf.stem] = data
                    except Exception:
                        logger.warning("Failed to load knowledge file %s", jf)

                if diseases:
                    if crop_key in merged and isinstance(merged[crop_key], dict) and "diseases" in merged[crop_key]:
                        merged[crop_key]["diseases"].update(diseases)
                    else:
                        merged[crop_key] = {"diseases": diseases}

            if merged:
                try:
                    merged_path = Path("config") / "merged_disease_knowledge.json"
                    with open(merged_path, "w", encoding="utf-8") as f:
                        json.dump(merged, f, indent=2, ensure_ascii=False)
                    kb_loader_path = merged_path
                except Exception:
                    logger.warning("Failed to write merged knowledge config.")

        try:
            app.state.knowledge_base = KnowledgeBase(kb_loader_path)
            logger.info("✓ Knowledge base loaded successfully from %s.", kb_loader_path)
        except (FileNotFoundError, ValueError) as e:
            logger.warning("✗ Knowledge base not loaded: %s.", e)
            app.state.knowledge_base = None

    # ---------------- ROUTES ----------------

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/background-video")
    async def background_video():
        video_path = Path("app/ui/templates/InShot_20260321_092840276.mp4")
        if not video_path.exists():
            return Response(status_code=404)
        return FileResponse(video_path, media_type="video/mp4")

    @app.get("/dashboard-background")
    async def dashboard_background():
        image_path = Path("app/ui/templates/Dashboard.jpg")
        if not image_path.exists():
            return Response(status_code=404)
        return FileResponse(image_path, media_type="image/jpeg")

    @app.get("/recommend/{disease}")
    async def recommend(disease: str):
        return get_recommendation(disease)

    # ---------------- INCLUDE ROUTERS ----------------

    app.include_router(ui_router)
    app.include_router(api_router)
    app.include_router(weather_router)
    app.include_router(iot_router)

    return app

app = create_app()


if __name__ == "__main__":
	import uvicorn

	uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
