from __future__ import annotations

import os
import threading
import time
from typing import Dict

_ALERT_COOLDOWN_SECONDS = 300
_last_alert_sent_at: Dict[int, float] = {}
_lock = threading.Lock()


def can_send_alert(user_id: int) -> bool:
	"""Return True if user can receive an alert now; enforce 5-minute cooldown."""
	now = time.time()
	with _lock:
		last_sent = _last_alert_sent_at.get(user_id)
		if last_sent is None or (now - last_sent) >= _ALERT_COOLDOWN_SECONDS:
			_last_alert_sent_at[user_id] = now
			return True
		return False


def get_alert_threshold_percent() -> float:
	"""Read confidence threshold from environment; default to 95%."""
	raw = os.getenv("ALERT_CONFIDENCE_THRESHOLD", "95")
	try:
		threshold = float(raw)
		if threshold < 0:
			return 95.0
		return threshold
	except (TypeError, ValueError):
		return 95.0
