from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Form, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from database.db import get_db
from database.models import AppUser

logger = logging.getLogger("leaflens.ui")

router = APIRouter()

ADMIN_USERNAME = "Admin"
ADMIN_PASSWORD = "Admin123"
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _get_templates(request: Request) -> Jinja2Templates:
	"""Get the shared Jinja templates instance from app state.

	`main.py` configures this in production. This fallback keeps the UI resilient
	in tests or alternate startup paths.
	"""
	templates = getattr(request.app.state, "templates", None)
	if isinstance(templates, Jinja2Templates):
		return templates
	logger.warning("Jinja2Templates not found in app.state; using fallback directory")
	return Jinja2Templates(directory="app/ui/templates")


def _is_authenticated(request: Request) -> bool:
	return bool(request.session.get("authenticated"))


def _is_admin(request: Request) -> bool:
	return request.session.get("role") == "admin"


def _login_redirect() -> RedirectResponse:
	return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)


def _dashboard_redirect() -> RedirectResponse:
	return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> Any:
	if not _is_authenticated(request):
		return _login_redirect()
	templates = _get_templates(request)
	return templates.TemplateResponse(
		"index.html",
		{
			"request": request,
			"app_name": "LeafLens",
			"tagline": "AI-Powered Crop Disease Detection & Smart Explanation Engine",
			"is_admin": _is_admin(request),
			"auth_user": request.session.get("username", "User"),
		},
	)


@router.get("/weather", response_class=HTMLResponse)
async def weather(request: Request) -> Any:
	if not _is_authenticated(request):
		return _login_redirect()
	templates = _get_templates(request)
	return templates.TemplateResponse(
		"weather.html",
		{
			"request": request,
			"app_name": "LeafLens",
			"tagline": "AI-Powered Crop Disease Detection & Smart Explanation Engine",
			"is_admin": _is_admin(request),
			"auth_user": request.session.get("username", "User"),
		},
	)


@router.get("/insight", response_class=HTMLResponse)
async def insight(request: Request) -> Any:
	if not _is_authenticated(request):
		return _login_redirect()
	templates = _get_templates(request)
	return templates.TemplateResponse(
		"leaflens_insight.html",
		{
			"request": request,
			"app_name": "LeafLens",
			"tagline": "AI-Powered Crop Disease Detection & Smart Explanation Engine",
			"is_admin": _is_admin(request),
			"auth_user": request.session.get("username", "User"),
		},
	)


@router.get("/chatbot", response_class=HTMLResponse)
async def chatbot(request: Request) -> Any:
	if not _is_authenticated(request):
		return _login_redirect()
	templates = _get_templates(request)
	return templates.TemplateResponse(
		"chatbot.html",
		{
			"request": request,
			"app_name": "LeafLens",
			"tagline": "AI-Powered Crop Disease Detection & Smart Explanation Engine",
			"is_admin": _is_admin(request),
			"auth_user": request.session.get("username", "User"),
		},
	)


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> Any:
	if _is_authenticated(request):
		return _dashboard_redirect()
	templates = _get_templates(request)
	return templates.TemplateResponse(
		"login.html",
		{
			"request": request,
			"error": None,
		},
	)


@router.post("/login", response_class=HTMLResponse)
async def login_submit(
	request: Request,
	role: str = Form("user"),
	username: str = Form(""),
	password: str = Form(""),
	email: str = Form(""),
	db: Session = Depends(get_db),
) -> Any:
	role_value = (role or "user").strip().lower()
	username_value = (username or "").strip()
	email_value = (email or "").strip().lower()
	templates = _get_templates(request)

	if role_value == "admin":
		if username_value != ADMIN_USERNAME or (password or "") != ADMIN_PASSWORD:
			return templates.TemplateResponse(
				"login.html",
				{
					"request": request,
					"error": "Invalid admin username or password.",
				},
				status_code=status.HTTP_401_UNAUTHORIZED,
			)

		request.session.clear()
		request.session.update(
			{
				"authenticated": True,
				"role": "admin",
				"username": ADMIN_USERNAME,
			}
		)
		return _dashboard_redirect()

	if not username_value:
		return templates.TemplateResponse(
			"login.html",
			{
				"request": request,
				"error": "Username is required for user login.",
			},
			status_code=status.HTTP_400_BAD_REQUEST,
		)

	if not email_value or not EMAIL_PATTERN.match(email_value):
		return templates.TemplateResponse(
			"login.html",
			{
				"request": request,
				"error": "A valid email is required for user login.",
			},
			status_code=status.HTTP_400_BAD_REQUEST,
		)

	existing_user = db.query(AppUser).filter(AppUser.email == email_value).first()
	if existing_user is None:
		existing_user = AppUser(username=username_value, email=email_value)
		db.add(existing_user)
	else:
		existing_user.username = username_value

	db.commit()

	request.session.clear()
	request.session.update(
		{
			"authenticated": True,
			"role": "user",
			"username": username_value,
			"email": email_value,
		}
	)

	return _dashboard_redirect()


@router.get("/logout")
async def logout(request: Request) -> RedirectResponse:
	request.session.clear()
	return _login_redirect()


@router.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request, db: Session = Depends(get_db)) -> Any:
	if not _is_authenticated(request):
		return _login_redirect()
	if not _is_admin(request):
		return _dashboard_redirect()

	users = db.query(AppUser).order_by(AppUser.created_at.desc()).all()
	templates = _get_templates(request)
	return templates.TemplateResponse(
		"admin_dashboard.html",
		{
			"request": request,
			"users": users,
			"user_count": len(users),
			"app_name": "LeafLens",
			"tagline": "AI-Powered Crop Disease Detection & Smart Explanation Engine",
			"is_admin": True,
			"auth_user": request.session.get("username", ADMIN_USERNAME),
		},
	)


@router.get("/login-photo")
async def login_photo() -> Response:
	photo_path = Path("app/ui/templates/Login.avif")
	if not photo_path.exists():
		return Response(status_code=status.HTTP_404_NOT_FOUND)
	return FileResponse(photo_path, media_type="image/avif")

