from __future__ import annotations

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv

logger = logging.getLogger("leaflens.email")

load_dotenv()

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587


def send_alert_email(
	to_email: str,
	disease: str,
	confidence: float,
	treatment: str,
	prevention: str,
) -> bool:
	"""Send high-confidence disease alert email.

	Returns True on success and False on failure.
	"""
	email_user = os.getenv("EMAIL_USER")
	email_pass = os.getenv("EMAIL_PASS")

	if not email_user or not email_pass:
		logger.error("Email credentials are not configured (EMAIL_USER/EMAIL_PASS).")
		return False

	subject = f"🚨 LeafLens Alert: {disease}"
	confidence_str = f"{confidence:.2f}%"

	plain_body = (
		"Hello from LeafLens AI,\n\n"
		"A high-confidence disease prediction was detected for your recent upload.\n\n"
		f"Disease: {disease}\n"
		f"Confidence: {confidence_str}\n\n"
		"Eco-friendly treatment:\n"
		f"- {treatment}\n\n"
		"Prevention:\n"
		f"- {prevention}\n\n"
		"Please inspect your crop and take preventive action as soon as possible.\n\n"
		"Stay safe,\n"
		"LeafLens AI"
	)

	html_body = f"""
	<html>
	  <body style=\"font-family: Arial, sans-serif; line-height: 1.6;\">
	    <p>Hello from <strong>LeafLens AI</strong>,</p>
	    <p>A high-confidence disease prediction was detected for your recent upload.</p>
	    <ul>
	      <li><strong>Disease:</strong> {disease}</li>
	      <li><strong>Confidence:</strong> {confidence_str}</li>
	    </ul>
	    <p><strong>Eco-friendly treatment</strong><br>{treatment}</p>
	    <p><strong>Prevention</strong><br>{prevention}</p>
	    <p>Please inspect your crop and take preventive action as soon as possible.</p>
	    <p>Stay safe,<br>LeafLens AI</p>
	  </body>
	</html>
	""".strip()

	msg = MIMEMultipart("alternative")
	msg["From"] = email_user
	msg["To"] = to_email
	msg["Subject"] = subject
	msg.attach(MIMEText(plain_body, "plain", "utf-8"))
	msg.attach(MIMEText(html_body, "html", "utf-8"))

	try:
		with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
			server.ehlo()
			server.starttls()
			server.ehlo()
			server.login(email_user, email_pass)
			server.sendmail(email_user, [to_email], msg.as_string())
		logger.info("Alert email sent to %s for disease=%s confidence=%.2f%%", to_email, disease, confidence)
		return True
	except Exception:
		logger.exception("Failed to send alert email to %s", to_email)
		return False
